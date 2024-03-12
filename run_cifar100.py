#!/usr/bin/env python3

import argparse
import copy
import json
import os
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import tqdm 
from torch import nn
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import Subset

from RejectionEnsemble import RejectionEnsemble
from RejectionEnsembleWithOnlineCalibration import RejectionEnsembleWithOnlineCalibration #, predict_batch, predict_batch_optimized, train_pytorch
from utils import benchmark_torch_batchprocessing, benchmark_torch_realtimeprocessing

class CIFARModelWrapper():
    def __init__(self, model_name):
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True, verbose=False)
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def features(self, X):
        if self.model.__class__.__name__ == "ShuffleNetV2":
            x = self.model.conv1(X)
            x = self.model.stage2(x)
            x = self.model.stage3(x)
            x = self.model.stage4(x)
            x = self.model.conv5(x)
            x = x.mean([2, 3])  # globalpool
            return x
        elif self.model.__class__.__name__ == "MobileNetV2":
            x = self.model.features(X)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        
            return x
        else:
            return X.flatten()
    
    def classifier(self, X):
        if self.model.__class__.__name__ == "ShuffleNetV2":
            return self.model.fc(X)
        elif self.model.__class__.__name__ == "MobileNetV2":
            return self.model.classifier(X)
        else:
            return self.model(X)

    def predict_single(self, x, return_cnt = False):
        return self.predict_batch(x, return_cnt) #.unsqueeze(0)

    def predict_batch(self, T, return_cnt = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            preds = self.model(T.to(device))
            if return_cnt:
                return preds, T.shape[0]
            else:
                return preds

def main(args):
    # Define transformation for the validation data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    kf = KFold(n_splits=args["x"], shuffle=True)
    dataset = datasets.CIFAR100(root=args["data"], train=False, download=True, transform=transform)

    # Use a pre-trained Wide ResNet model
    fbig = CIFARModelWrapper(args["big"])
    fsmall = CIFARModelWrapper(args["small"])
    
    if not ("cifar100_mobilenetv2" in args["small"] or "cifar100_shufflenetv2" in args["small"]):
        print("Warning: This script only supports mobilenetv2 and shufflenetv2 variants as the small model for feature extraction. The rejector will now be traiend on the raw data.")
    
    metrics = []
    
    if not isinstance(args["p"], list):
        Ps = [float(args["p"])]
    else:
        Ps = [float(p) for p in args["p"]]

    rejectors = [
        {
            "model":"LogisticRegression"
        },
        {
            "model":"DecisionTreeClassifier",
            "max_depth":2
        },
        {
            "model":"DecisionTreeClassifier",
            "max_depth":5
        },
        {
            "model":"DecisionTreeClassifier",
            "max_depth":10
        },
        {
            "model":"DecisionTreeClassifier",
            "max_depth":None
        }
    ]
    measure_jetson_power = args["e"]
    n_data = len(dataset)

    cfgs = []
    n_experiments = 0
    for i, (train_idx, test_idx) in enumerate(kf.split(range(n_data))):
        for k, r in enumerate(rejectors):  
            rname = "_".join([str(v) for v in r.values()]) 
            for tm in ["confidence", "virtual-labels"]:
                for p in Ps:
                    cfgs.append(
                        {
                            "model":RejectionEnsemble(fsmall, fbig, p=p, rejector_cfg=copy.copy(r), return_cnt=True, train_method=tm, calibration=False),
                            "train":train_idx,
                            "test":test_idx,
                            "i":i,
                            "rejector":f"{rname}",
                            "train_method":tm,
                            "calibration":False,
                            "p":p
                        }
                    )
                    n_experiments += 1
                cfgs.append(
                    {
                        "model":RejectionEnsemble(fsmall, fbig, p=0, rejector_cfg=copy.copy(r), return_cnt=True, train_method=tm, calibration=True),
                        "train":train_idx,
                        "test":test_idx,
                        "i":i,
                        "rejector":f"{rname}",
                        "train_method":tm,
                        "calibration":True
                    }
                )
                n_experiments += len(Ps)
        cfgs.append(
            {
                "model":fsmall,
                "train":None,
                "test":test_idx,
                "i":i,
                "big":False
            }
        )
        cfgs.append(
            {
                "model":fbig,
                "train":None,
                "test":test_idx,
                "i":i,
                "big":True
            }
        )
        n_experiments += 2

    with tqdm.tqdm(total=n_experiments, desc = "Overall progress") as pb:
        for cfg in cfgs:
            train_dataset = Subset(dataset, cfg["train"])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["b"], shuffle=False, pin_memory=True, num_workers = 6)
            test_dataset = Subset(dataset, cfg["test"])

            if cfg["train"] is None:
                pb.set_description(f"(BATCH) Applying {'big' if cfg['big'] else 'small'} model for {i+1}/{args['x']}")
                metrics.append({
                    "model":"big" if cfg["big"] else "small",
                    "batch":True,
                    "rejector":None,
                    "train_method":None,
                    "calibration":None,
                    "run":cfg["i"],
                    "p":None,
                    **benchmark_torch_batchprocessing(test_dataset, cfg["model"], args["b"], f"{i+1}/{args['x']} Applying {'big' if cfg['big'] else 'small'} model", jetson=measure_jetson_power,verbose=False)
                })
                pb.update(1)
            else:
                if cfg["calibration"]:
                    pb.set_description(f"Training rejection ensemble for p = {p} and r = {rname} and run {i+1}/{args['x']} ")
                    re = cfg["model"]

                    re.train_pytorch(train_loader, f"{i+1}/{args['x']}", False)
                    for p in Ps:
                        re.p = p
                        metrics.append({
                            "model":"RE",
                            "batch":True,
                            "train_method":cfg["train_method"],
                            "calibration":cfg["calibration"],
                            "rejector":cfg["rejector"],
                            "run":cfg["i"],
                            "p":p,
                            **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} (BATCH) Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power, verbose=False)
                        })
                        pb.update(1)
                else:
                    pb.set_description(f"Training rejection ensemble for p = {p} and r = {rname} and run {i+1}/{args['x']} ")
                    re = cfg["model"]

                    re.train_pytorch(train_loader, f"{i+1}/{args['x']}", False)
                    metrics.append({
                        "model":"RE",
                        "batch":True,
                        "train_method":cfg["train_method"],
                        "calibration":cfg["calibration"],
                        "rejector":cfg["rejector"],
                        "run":cfg["i"],
                        "p":cfg["p"],
                        **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} (BATCH) Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power, verbose=False)
                    })
                    pb.update(1)

    with open(args["out"], "w") as outfile:
        json.dump(metrics, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument("--data", help='Path to CIFAR100 data. Downloads CIFAR100 automaticall to that folder if it is not found.', required=False, type=str, default="/mnt/ssd/data/cifar100")
    parser.add_argument("--small", help='Small model to be used. See chenyaofo/pytorch-cifar-models for a list of available models.', required=False, type=str, default="cifar100_shufflenetv2_x0_5")
    parser.add_argument("--big", help='Big model to be used. See chenyaofo/pytorch-cifar-models for a list of available models.', required=False, type=str, default="cifar100_repvgg_a2")
    parser.add_argument("-e", help='If true, energy is measured.', action='store_true')
    parser.add_argument("-b", help='Batch size.', required=False, type=int, default=64)
    parser.add_argument("-x", help='Number of x-val splits.', required=False, type=int, default=5)
    parser.add_argument("-p", help='Budgets to try.', required=False, nargs='+', default=list(np.arange(0.0, 1.05, 0.05)))
    parser.add_argument("--out", help='Name / Path of output json.', required=False, type=str, default="cifar100.json")
    args = vars(parser.parse_args())
    
    main(args)