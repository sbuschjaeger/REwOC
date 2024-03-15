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
from torchvision.models import get_model
import tqdm 
from torch import nn
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import Subset

#from jtop import jtop
from TorchRejectionEnsemble import TorchRejectionEnsemble
#from RejectionEnsembleWithOnlineCalibration import RejectionEnsembleWithOnlineCalibration #, predict_batch, predict_batch_optimized, train_pytorch
from utils import benchmark_torch_batchprocessing

class ImageNetModelWrapper():
    def __init__(self, model_name):
        self.model = get_model(model_name, weights="DEFAULT")
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def features(self, X):
        if self.model.__class__.__name__ == "MobileNetV3":
            x = self.model.features(X)
            x = self.model.avgpool(x)
            return x.flatten(1)
        else:
            return X.flatten()
    
    def classifier(self, X):
        if self.model.__class__.__name__ == "MobileNetV3":
            return self.model.classifier(X)
        else:
            return self.model(X)

    def predict_single(self, x, return_cnt = False):
        return self.predict_batch(x, return_cnt)

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    kf = KFold(n_splits=args["x"], shuffle=True)
    dataset = datasets.ImageFolder(os.path.join(args["data"], "val"), transform=transform)

    # Use a pre-trained Wide ResNet model
    fbig = ImageNetModelWrapper(args["big"])
    fsmall = ImageNetModelWrapper(args["small"])
    
    if not ("mobilenet_v3_small" in args["small"]):
        print("Warning: This script only supports MobileNetV3 as the small model for feature extraction. The rejector will now be traiend on the raw data.")
    
    metrics = []
    
    if not isinstance(args["p"], list):
        Ps = [float(args["p"])]
    else:
        Ps = [float(p) for p in args["p"]]

    rejectors = [
        # {
        #     "model":"LogisticRegression"
        # },
        # {
        #     "model":"DecisionTreeClassifier",
        #     "max_depth":2
        # },
        # {
        #     "model":"DecisionTreeClassifier",
        #     "max_depth":5
        # },
        # {
        #     "model":"DecisionTreeClassifier",
        #     "max_depth":10
        # },
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
                for c in [True, False]:
                    cfgs.append(
                        {
                            "model":TorchRejectionEnsemble(fsmall, fbig, p=0, rejector_cfg=copy.copy(r), return_cnt=True, train_method=tm, calibration=c),
                            "train":train_idx,
                            "test":test_idx,
                            "i":i,
                            "rejector":f"{rname}",
                            "train_method":tm,
                            "calibration":c
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
                    pb.set_description(f"Training rejection ensemble for r = {rname} and run {i+1}/{args['x']} ")
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
                    pb.set_description(f"Training rejection ensemble for r = {rname} and run {i+1}/{args['x']} ")
                    re = cfg["model"]

                    re.train_pytorch(train_loader, f"{i+1}/{args['x']}", False)
                    for p in Ps:
                        re.p = p
                        re._train_rejector()

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

                    
                    # metrics.append({
                    #     "model":"RE",
                    #     "batch":True,
                    #     "train_method":cfg["train_method"],
                    #     "calibration":cfg["calibration"],
                    #     "rejector":cfg["rejector"],
                    #     "run":cfg["i"],
                    #     "p":cfg["p"],
                    #     **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} (BATCH) Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power, verbose=False)
                    # }) 
                    # pb.update(1)


    with open(args["out"], "w") as outfile:
        json.dump(metrics, outfile)
    # df = pd.DataFrame(metrics)
    # df.to_csv(args["out"], index = False)
    # df.groupby(["model"])["time", "f1 macro", "f1 micro", "accuracy"].mean()
    # df.groupby(["model"])["time", "f1 macro", "f1 micro", "accuracy"].std()
    # print(pd.DataFrame(metrics))
    
    # dawd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-label classification problem on a series of patients. Training and evaluation are performed on a per-patient basis, i.e. we train on patients {1,2,3} and test on patient 4.')
    parser.add_argument("--data", help='Path to CIFAR100 data.', required=False, type=str, default="/mnt/ssd/data/ImageNet")
    parser.add_argument("--small", help='Path to ImageNet data.', required=False, type=str, default="mobilenet_v3_small")
    parser.add_argument("--big", help='Path to ImageNet data.', required=False, type=str, default="vit_b_32")
    parser.add_argument("-b", help='Batch size.', required=False, type=int, default=32)
    parser.add_argument("-x", help='Number of x-val splits.', required=False, type=int, default=5)
    parser.add_argument("-e", help='If true, energy is measured.', action='store_true')
    # parser.add_argument("--rejector", help='Rejector.', required=False, type=str, default="DecisionTreeClassifier")
    # parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=[0,0.5,1])
    parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=list(np.arange(0.0, 1.1, 0.1)))
    parser.add_argument("--out", help='Name / Path of output csv.', required=False, type=str, default="imagenet.json")
    args = vars(parser.parse_args())
    
    main(args)
