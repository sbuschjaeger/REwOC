#!/usr/bin/env python3

import argparse
import copy
import json
import os
import time
import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
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

from TorchRejectionEnsemble import TorchRejectionEnsemble, get_predictions
from utils import JetsonMonitor, benchmark_torch_batchprocessing

class CIFARModelWrapper():
    # A wrapper around the CIFAR100 models to provide the common interface featuring a features() and classifier() function
    def __init__(self, model_name):
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True, verbose=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

    def __call__(self, x):
        return self.model(x.to(self.device))

class ImageNetModelWrapper():
    # A wrapper around the ImageNet models to provide the common interface featuring a features() and classifier() function
    def __init__(self, model_name):
        self.model = get_model(model_name, weights="DEFAULT")
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def features(self, X):
        if self.model.__class__.__name__ == "MobileNetV3" or self.model.__class__.__name__ == "EfficientNet":
            x = self.model.features(X)
            x = self.model.avgpool(x)
            return x.flatten(1)
        elif self.model.__class__.__name__ == "ShuffleNetV2":
            x = self.model.conv1(X)
            x = self.model.maxpool(x)
            x = self.model.stage2(x)
            x = self.model.stage3(x)
            x = self.model.stage4(x)
            x = self.model.conv5(x)
            x = x.mean([2, 3])  # globalpool
            return x
        else:
            return X.flatten()
    
    def classifier(self, X):
        if self.model.__class__.__name__ == "MobileNetV3" or self.model.__class__.__name__ == "EfficientNet":
            return self.model.classifier(X)
        elif self.model.__class__.__name__ == "ShuffleNetV2":
            return self.model.fc(X)
        else:
            return self.model(X)

    def __call__(self, T):
        return self.model(T.to(self.device))

def main(args):
    if args["data"] == "cifar100":
        # Prepare CIFAR 100 data and small/big model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

        dataset = datasets.CIFAR100(root=args["tmp"], train=False, download=True, transform=transform)

        fbig = CIFARModelWrapper(args["big"])
        fsmall = CIFARModelWrapper(args["small"])
        
        if not ("cifar100_mobilenetv2" in args["small"] or "cifar100_shufflenetv2" in args["small"]):
            print("Warning: This script only supports mobilenetv2 and shufflenetv2 variants as the small model for feature extraction. The rejector will now be traiend on the raw data.")
    elif args["data"] == "imagenet":
        # Prepare ImageNet Data and small/big model
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(os.path.join(args["tmp"], "val"), transform=transform)

        fbig = ImageNetModelWrapper(args["big"])
        fsmall = ImageNetModelWrapper(args["small"])
        
        if args["small"] != "mobilenet_v3_small" and args["small"] != "efficientnet_b0":
            print("Warning: This script only supports MobileNetV3 as the small model for feature extraction. The rejector will now be traiend on the raw data.")
    else:
        raise ValueError(f"Received wrong dataset. Currently supported are `cifar100' and `imagenet', but I got {args['data']}")
    
    # Prepare rejector
    if args["rejector"] == "dt":
        rejector = DecisionTreeClassifier(max_depth=None)
    elif args["rejector"] == "rf":
        rejector = RandomForestClassifier(n_estimators = 16, max_depth=None)
    elif args["rejector"] == "linear":
        # We use liblinear, because its binary classification problem and lfbgs sometimes leads to convergence issues 
        rejector = LogisticRegression(solver="liblinear")
    else:
        raise ValueError(f"Given rejector not supported. Currently supported are dt,linear and rf, but I got {args['rejector']}")
    rejector_name = args["rejector"]

    # prepare cross validation
    kf = KFold(n_splits=args["x"], shuffle=True)
    metrics = []
    
    # Prepare budgets to be used during experiments
    if not isinstance(args["p"], list):
        Ps = [float(args["p"])]
    else:
        Ps = [float(p) for p in args["p"]]

    # Measure energy?
    if args["e"]:
        jetson = JetsonMonitor()
        jetson.start()
    else:
        jetson = None

    n_data = len(dataset)

    for i, (train_idx, test_idx) in enumerate(kf.split(range(n_data))):
        train_dataset = Subset(dataset, train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["M"], shuffle=False, pin_memory=True, num_workers = 4)
        test_dataset = Subset(dataset, test_idx)

        # Cache the predictions of the small and big model for faster training of the rejectors. This is not part of the benchmarking because we benchmark model application and not rejector training
        small_preds, X, Y = get_predictions(fsmall, train_loader, True, True, pbar_desc=f"[{i+1}]/[{args['x']}] Getting predictions of small model")
        big_preds = get_predictions(fbig, train_loader, False, False, pbar_desc=f"[{i+1}]/[{args['x']}] Getting predictions of big model")

        # Benchmark all four combinations
        for tm in ["confidence", "virtual-labels"]:
            for c in [True, False]:
                for p in tqdm.tqdm(Ps, total=len(Ps), desc=f"[{i+1}]/[{args['x']}] Running experiments for {tm} {'with calibration' if c else 'without calibration'}"):
                    r = clone(rejector)
                    re = TorchRejectionEnsemble(fsmall, fbig, p=p, rejector = r, train_method=tm, calibration=c)
                    re._fit(X, Y, small_preds, big_preds)

                    metrics.append({
                            "model":"RE",
                            "small":args["small"],
                            "big":args["big"],
                            "batch":True,
                            "rejector":rejector_name,
                            "train_method":tm,
                            "calibration":c,
                            "run":i,
                            "p":p,
                            **benchmark_torch_batchprocessing(test_dataset, re, args["M"], "", jetson=jetson,verbose=False)
                        }
                    )
        
        # Benchmark small and big model as well
        print(f"[{i+1}]/[{args['x']}] Benchmarking small model")
        metrics.append({
                "model":"small",
                "small":args["small"],
                "big":args["big"],
                "batch":True,
                "rejector":None,
                "train_method":None,
                "calibration":None,
                "run":i,
                "p":None,
                **benchmark_torch_batchprocessing(test_dataset, fsmall, args["M"], f"{i+1}/{args['x']} Applying small model", jetson=jetson,verbose=False)
            }
        )

        print(f"[{i+1}]/[{args['x']}] Benchmarking big model")
        metrics.append({
                "model":"big",
                "small":args["small"],
                "big":args["big"],
                "batch":True,
                "rejector":None,
                "train_method":None,
                "calibration":None,
                "run":i,
                "p":None,
                **benchmark_torch_batchprocessing(test_dataset, fbig, args["M"], f"{i+1}/{args['x']} Applying big model", jetson=jetson,verbose=False)
            }
        )
        print("")

    with open(os.path.join(args["out"], f"{args['data']}.json"), "w") as outfile:
        json.dump(metrics, outfile)
    
    if jetson:
        jetson.stop()
        jetson.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running exeriments on torch-based rejectors, i.e. on CIFAR100 and ImageNet.')
    parser.add_argument("--data", help='Dataset used for the experiments. Currently cifar100 and imagenet are supported.', required=False, type=str, default="cifar100")
    parser.add_argument("--tmp", help='Path to the data. In case of cifar100, data will automatically be downloaded to the given folder if not found. In case of imagenet, it is assuemd that the validation data is already donwnlaoded to the given folder.', required=False, type=str, default=".")
    parser.add_argument("--small", help='Small model to be used. For cifar100, models form chenyaofo/pytorch-cifar-models are supported. For imagenet, models form torchvision.models are supported.', required=False, type=str, default="cifar100_shufflenetv2_x0_5")
    parser.add_argument("--big", help='Big model to be used. For cifar100, models form chenyaofo/pytorch-cifar-models are supported. For imagenet, models form torchvision.models are supported.', required=False, type=str, default="cifar100_repvgg_a2")
    parser.add_argument("--rejector", help='Rejector to be used. Currently dt (DecisionTreeClassifier with max_depth = None), rf (RandomForestClassifier with 16 trees and max_depth = None) and linear LogisticRegression are supported', required=False, type=str, default="dt")
    parser.add_argument("-e", help='If true, energy is measured. This only works on Jetson Boards.', action='store_true')
    parser.add_argument("-M", help='Batch size during deployment.', required=False, type=int, default=32)
    parser.add_argument("-x", help='Number of cross-validation splits.', required=False, type=int, default=5)
    parser.add_argument("-p", help='Budgets to try.', required=False, nargs='+', default=list(np.arange(0.0, 1.05, 0.05)))
    parser.add_argument("--out", help='Folder in which to store the output file. Name will be the same as the dataset name, e.g. cifar100.json.', required=False, type=str, default=".")

    args = vars(parser.parse_args())
    
    main(args)