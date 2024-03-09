#!/usr/bin/env python3

import argparse
import copy
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

from jtop import jtop
from RejectionEnsemble import RejectionEnsemble
from RejectionEnsembleWithOnlineCalibration import RejectionEnsembleWithOnlineCalibration #, predict_batch, predict_batch_optimized, train_pytorch
from utils import benchmark_torch_batchprocessing

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
        return self.predict_batch(x.unsqueeze(0), return_cnt)

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
        Ps = [args["p"]]
    else:
        Ps = args["p"]

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
    measure_jetson_power = False
    for i, (train_idx, test_idx) in enumerate(kf.split(dataset.data)):
        train_dataset = Subset(dataset, train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["b"], shuffle=False, pin_memory=True, num_workers = 6)
        test_dataset = Subset(dataset, test_idx)

        for k, r in enumerate(rejectors):   
            rname = "_".join([str(v) for v in r.values()])
            # p does not depend on the training, so we only have to train one rewoc per P
            # Hence, we set p to a dummy value here, because we set it to the corret value later in the loop for evaluation
            rewoc = RejectionEnsembleWithOnlineCalibration(fsmall, fbig, p=0, rejector_cfg=copy.copy(r), return_cnt=True)
            rewoc.train_pytorch(train_loader, f"{i+1}/{args['x']}")

            for p in Ps:
                rewoc.p = p 

                metrics.append({
                    "model":f"REwOC",
                    "rejector":f"{rname}",
                    "run":i,
                    "p":p,
                    **benchmark_torch_batchprocessing(test_dataset, rewoc, args["b"], f"{i+1}/{args['x']} Applying rejection ensemble with online calibration for p = {p} and r = {rname}", jetson=measure_jetson_power)
                })

                re = RejectionEnsemble(fsmall, fbig, rejector_cfg=copy.copy(r), p=p, return_cnt=True)
                re.train_pytorch(train_loader, f"{i+1}/{args['x']}")

                metrics.append({
                    "model":f"RE",
                    "rejector":f"{rname}",
                    "run":i,
                    "p":p,
                    **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power)
                })

                if p == Ps[0] and k == 0:
                    metrics.append({
                        "model":"small",
                        "rejector":None,
                        "run":i,
                        "p":p,
                        **benchmark_torch_batchprocessing(test_dataset, fsmall, args["b"], f"{i+1}/{args['x']} Applying small model", jetson=measure_jetson_power)
                    })

                    metrics.append({
                        "model":"big",
                        "rejector":None,
                        "run":i,
                        "p":p,
                        **benchmark_torch_batchprocessing(test_dataset, fbig, args["b"], f"{i+1}/{args['x']} Applying big model", jetson=measure_jetson_power)
                    })

    df = pd.DataFrame(metrics)
    df.to_csv(args["out"], index = False)
    # df.groupby(["model"])["time", "f1 macro", "f1 micro", "accuracy"].mean()
    # df.groupby(["model"])["time", "f1 macro", "f1 micro", "accuracy"].std()
    # print(pd.DataFrame(metrics))
    
    # dawd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-label classification problem on a series of patients. Training and evaluation are performed on a per-patient basis, i.e. we train on patients {1,2,3} and test on patient 4.')
    parser.add_argument("--data", help='Path to CIFAR100 data.', required=False, type=str, default="/mnt/ssd/data/cifar100")
    parser.add_argument("--small", help='Path to ImageNet data.', required=False, type=str, default="cifar100_shufflenetv2_x0_5")
    parser.add_argument("--big", help='Path to ImageNet data.', required=False, type=str, default="cifar100_repvgg_a2")
    parser.add_argument("-b", help='Batch size.', required=False, type=int, default=64)
    parser.add_argument("-x", help='Number of x-val splits.', required=False, type=int, default=5)
    # parser.add_argument("--rejector", help='Rejector.', required=False, type=str, default="DecisionTreeClassifier")
    parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=[0, 0.25, 0.5, 0.75, 1])
    #parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=list(np.arange(0.0, 1.05, 0.05)))
    parser.add_argument("--out", help='Name / Path of output csv.', required=False, type=str, default="cifar100.csv")
    args = vars(parser.parse_args())
    
    main(args)