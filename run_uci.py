#!/usr/bin/env python3

import argparse
import copy
import json
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import tqdm 
from torch import nn
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from torch.utils.data import Subset
from sklearn.base import clone

from Datasets import get_dataset
from SKRejectionEnsemble import SKRejectionEnsemble
from utils import benchmark_batchprocessing, benchmark_torch_batchprocessing, benchmark_torch_realtimeprocessing

def main(args):
    big_model = RandomForestClassifier(n_estimators=128, max_depth=None)
    small_model = DecisionTreeClassifier(max_depth = 5)
    rejector = DecisionTreeClassifier(max_depth=None)

    X,Y = get_dataset(args["data"])

    kf = KFold(n_splits=args["x"], shuffle=True)
    
    metrics = []
    
    if not isinstance(args["p"], list):
        Ps = [float(args["p"])]
    else:
        Ps = [float(p) for p in args["p"]]

    measure_jetson_power = args["e"]
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        fbig = clone(big_model)
        fsmall = clone(small_model)

        X_train, y_train = X[train_idx], Y[train_idx]
        print("Fitting big model")
        fbig.fit(X_train,y_train)
        print("Fitting small model")
        fsmall.fit(X_train,y_train)

        X_train_re, Y_train_re, X_test, Y_test = train_test_split(X[test_idx], Y[test_idx], test_size = 0.5)

        for tm in ["confidence", "virtual-labels"]:
            for c in [True, False]:
                print(f"Running experiments for tm = {tm} and c = {c}")
                
                if c:
                    r = clone(rejector)
                    re = SKRejectionEnsemble(fsmall, fbig, p=0, rejector = r, train_method=tm, calibration=c)
                    re.fit(X_train_re,Y_train_re)
                    for p in Ps:
                        re.p = p
                        metrics.append({
                                "model":"RE",
                                "batch":True,
                                "rejector":"DecisionTreeClassifier",
                                "train_method":tm,
                                "calibration":c,
                                "run":i,
                                "p":p,
                                **benchmark_batchprocessing(X_test, Y_test, re, args["b"], f"{i+1}/{args['x']} Applying rejection ensemble for p = {p}", jetson=measure_jetson_power,verbose=False)
                            }
                        )
                else:
                    for p in Ps:
                        r = clone(rejector)
                        re = SKRejectionEnsemble(fsmall, fbig, p=p, rejector = r, train_method=tm, calibration=c)
                        r.fit(X_train_re,Y_train_re)

                        metrics.append( {
                                "model":"RE",
                                "batch":True,
                                "rejector":"DecisionTreeClassifier",
                                "train_method":tm,
                                "calibration":c,
                                "run":i,
                                "p":p,
                                **benchmark_batchprocessing(X_test, Y_test, re, args["b"], f"{i+1}/{args['x']} Applying rejection ensemble for p = {p}", jetson=measure_jetson_power,verbose=False)
                            }
                        )

        metrics.append({
                "model":"small",
                "batch":True,
                "rejector":None,
                "train_method":None,
                "calibration":None,
                "run":i,
                "p":None,
                **benchmark_batchprocessing(X_test, Y_test, fsmall, args["b"], f"{i+1}/{args['x']} Applying small model", jetson=measure_jetson_power,verbose=False)
            }
        )

        metrics.append({
                "model":"big",
                "batch":True,
                "rejector":None,
                "train_method":None,
                "calibration":None,
                "run":i,
                "p":None,
                **benchmark_batchprocessing(X_test, Y_test, fbig, args["b"], f"{i+1}/{args['x']} Applying big model", jetson=measure_jetson_power,verbose=False)
            }
        )

    #     rname = "_".join([str(v) for v in r.values()]) 
    #     for tm in ["confidence", "virtual-labels"]:
    #         for p in Ps:
    #             cfgs.append(
    #                 {
    #                     "model":SKRejectionEnsemble(fsmall, fbig, p=p, rejector_cfg=copy.copy(r), return_cnt=True, train_method=tm, calibration=False),
    #                     "train":train_idx,
    #                     "test":test_idx,
    #                     "i":i,
    #                     "rejector":f"{rname}",
    #                     "train_method":tm,
    #                     "calibration":False,
    #                     "p":p
    #                 }
    #             )
    #             n_experiments += 1
    #         cfgs.append(
    #             {
    #                 "model":RejectionEnsemble(fsmall, fbig, p=0, rejector_cfg=copy.copy(r), return_cnt=True, train_method=tm, calibration=True),
    #                 "train":train_idx,
    #                 "test":test_idx,
    #                 "i":i,
    #                 "rejector":f"{rname}",
    #                 "train_method":tm,
    #                 "calibration":True
    #             }
    #         )
    #         n_experiments += len(Ps)
    #     cfgs.append(
    #         {
    #             "model":fsmall,
    #             "train":None,
    #             "test":test_idx,
    #             "i":i,
    #             "big":False
    #         }
    #     )
    #     cfgs.append(
    #         {
    #             "model":fbig,
    #             "train":None,
    #             "test":test_idx,
    #             "i":i,
    #             "big":True
    #         }
    #     )
    #     n_experiments += 2

    # with tqdm.tqdm(total=n_experiments, desc = "Overall progress") as pb:
    #     for cfg in cfgs:
    #         train_dataset = Subset(dataset, cfg["train"])
    #         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["b"], shuffle=False, pin_memory=True, num_workers = 6)
    #         test_dataset = Subset(dataset, cfg["test"])

    #         if cfg["train"] is None:
    #             pb.set_description(f"(BATCH) Applying {'big' if cfg['big'] else 'small'} model for {i+1}/{args['x']}")
    #             metrics.append({
    #                 "model":"big" if cfg["big"] else "small",
    #                 "batch":True,
    #                 "rejector":None,
    #                 "train_method":None,
    #                 "calibration":None,
    #                 "run":cfg["i"],
    #                 "p":None,
    #                 **benchmark_torch_batchprocessing(test_dataset, cfg["model"], args["b"], f"{i+1}/{args['x']} Applying {'big' if cfg['big'] else 'small'} model", jetson=measure_jetson_power,verbose=False)
    #             })
    #             pb.update(1)
    #         else:
    #             if cfg["calibration"]:
    #                 pb.set_description(f"Training rejection ensemble for p = {p} and r = {rname} and run {i+1}/{args['x']} ")
    #                 re = cfg["model"]

    #                 re.train_pytorch(train_loader, f"{i+1}/{args['x']}", False)
    #                 for p in Ps:
    #                     re.p = p
    #                     metrics.append({
    #                         "model":"RE",
    #                         "batch":True,
    #                         "train_method":cfg["train_method"],
    #                         "calibration":cfg["calibration"],
    #                         "rejector":cfg["rejector"],
    #                         "run":cfg["i"],
    #                         "p":p,
    #                         **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} (BATCH) Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power, verbose=False)
    #                     })
    #                     pb.update(1)
    #             else:
    #                 pb.set_description(f"Training rejection ensemble for p = {p} and r = {rname} and run {i+1}/{args['x']} ")
    #                 re = cfg["model"]

    #                 re.train_pytorch(train_loader, f"{i+1}/{args['x']}", False)
    #                 metrics.append({
    #                     "model":"RE",
    #                     "batch":True,
    #                     "train_method":cfg["train_method"],
    #                     "calibration":cfg["calibration"],
    #                     "rejector":cfg["rejector"],
    #                     "run":cfg["i"],
    #                     "p":cfg["p"],
    #                     **benchmark_torch_batchprocessing(test_dataset, re, args["b"], f"{i+1}/{args['x']} (BATCH) Applying rejection ensemble for p = {p} and r = {rname}", jetson=measure_jetson_power, verbose=False)
    #                 })
    #                 pb.update(1)

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