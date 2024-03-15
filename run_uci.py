#!/usr/bin/env python3

import argparse
import copy
import json
import os
import time
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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
from utils import benchmark_batchprocessing

def main(args):
    #big_model = AdaBoostClassifier(n_estimators=128,algorithm="SAMME.R")
    big_model = RandomForestClassifier(n_estimators=128, max_depth=None)
    #small_model = RandomForestClassifier(n_estimators=16, max_depth=None)
    #small_model = LogisticRegression()
    small_model = AdaBoostClassifier(n_estimators=5,algorithm="SAMME.R",estimator=DecisionTreeClassifier(max_depth=2))
    # small_model = DecisionTreeClassifier(max_depth = 5)
    # rejector = LogisticRegression()
    #rejector = RandomForestClassifier(n_estimators=16, max_depth=None)
    rejector = DecisionTreeClassifier(max_depth=None)
    # rejector = DecisionTreeClassifier(max_depth=None)

    if not isinstance(args["data"], list):
        datasets = [args["data"]]
    else:
        datasets = args["data"]

    if not isinstance(args["p"], list):
        Ps = [float(args["p"])]
    else:
        Ps = [float(p) for p in args["p"]]

    for d in datasets:
        X,Y = get_dataset(d, args["tmp"])
        kf = KFold(n_splits=args["x"], shuffle=True)
        
        metrics = []
        measure_jetson_power = args["e"]
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            fbig = clone(big_model)
            fsmall = clone(small_model)

            X_train, y_train = X[train_idx], Y[train_idx]
            print(f"{d}: [{i+1}]/[{args['x']}] Fitting big model")
            fbig.fit(X_train,y_train)
            print(f"{d}: [{i+1}]/[{args['x']}] Fitting small model")
            fsmall.fit(X_train,y_train)

            X_train_re, X_test, Y_train_re, Y_test = train_test_split(X[test_idx], Y[test_idx], test_size = 0.5)

            for tm in ["confidence", "virtual-labels"]:
                for c in [True, False]:
                    print(f"{d}: [{i+1}]/[{args['x']}] Running experiments for tm = {tm} {'with calibration' if c else 'without calibration'}")
                    
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
                            re.fit(X_train_re,Y_train_re)

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

        with open(os.path.join(args["out"], f"{d}.json"), "w") as outfile:
            json.dump(metrics, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-label classification problem on a series of patients. Training and evaluation are performed on a per-patient basis, i.e. we train on patients {1,2,3} and test on patient 4.')
    parser.add_argument("--data", help='Name of dataset to be used', required=False, nargs="+", default=["magic"])
    parser.add_argument("--tmp", help='Temporarly folder', required=False, type=str, default="./data")
    parser.add_argument("-b", help='Batch size.', required=False, type=int, default=32)
    parser.add_argument("-x", help='Number of x-val splits.', required=False, type=int, default=5)
    parser.add_argument("-e", help='If true, energy is measured.', action='store_true')
    parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=[0, 0.5, 1.0])
    parser.add_argument("--out", help='Name / Path of output csv.', required=False, type=str, default=".")
    args = vars(parser.parse_args())
    
    main(args)