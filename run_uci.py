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
from RejectionEnsemble import RejectionEnsemble
from utils import JetsonMonitor, benchmark_batchprocessing

def main(args):
    # Prepare the big model
    if args["big"] == "rf":
        big_model = RandomForestClassifier(n_estimators=128, max_depth=None, n_jobs=8)
    elif args["big"] == "dt2":
        big_model = DecisionTreeClassifier(max_depth=2)
    elif args["big"] == "dt3":
        big_model = DecisionTreeClassifier(max_depth=3)
    else:
        raise ValueError(f"Unknown big model given: rf are supported but received {args['big']} ")
    
    # Prepare the small model
    if args["small"] == "rf":
        small_model = RandomForestClassifier(n_estimators=16, max_depth=None, n_jobs=8)    
    elif args["small"] == "boosting":
        small_model = AdaBoostClassifier(n_estimators=5,algorithm="SAMME.R",estimator=DecisionTreeClassifier(max_depth=2))
    elif args["small"] == "dt":
        small_model = DecisionTreeClassifier(max_depth=1)
    else:
        raise ValueError(f"Unknown small model given: boosting, rf are supported but received {args['small']} ")
    
    # Prepare the rejector
    if args["rejector"] == "dt":
        rejector = DecisionTreeClassifier(max_depth=None)
    elif args["rejector"] == "rf":
        rejector = RandomForestClassifier(n_estimators=16, max_depth=None, n_jobs=8)
    elif args["rejector"] == "linear":
        rejector = LogisticRegression(solver="liblinear")
    else:
        raise ValueError(f"Unknown rejector given: dt, rf, linear are supported but received {args['rejector']} ")

    # Prepare datasets to be used during experiments
    if not isinstance(args["data"], list):
        datasets = [args["data"]]
    else:
        datasets = args["data"]

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

    for d in datasets:
        # Get dataset and prepare cross-validation
        X,Y = get_dataset(d, args["tmp"])
        kf = KFold(n_splits=args["x"], shuffle=True)
        
        metrics = []
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Train small and big model
            fbig = clone(big_model)
            fsmall = clone(small_model)

            X_train, y_train = X[train_idx], Y[train_idx]
            print(f"{d}: [{i+1}]/[{args['x']}] Fitting big model")
            fbig.fit(X_train,y_train)
            print(f"{d}: [{i+1}]/[{args['x']}] Fitting small model")
            fsmall.fit(X_train,y_train)

            # Prepare training data for rejector 
            X_train_re, X_test, Y_train_re, Y_test = train_test_split(X[test_idx], Y[test_idx], test_size = 0.5)

            # Benchmark all four combinations
            for tm in ["confidence", "virtual-labels"]:
                for c in [True, False]:
                    print(f"{d}: [{i+1}]/[{args['x']}] Running experiments for tm = {tm} {'with calibration' if c else 'without calibration'}")
                    
                    if c:
                        r = clone(rejector)
                        re = RejectionEnsemble(fsmall, fbig, p=0, rejector = r, train_method=tm, calibration=c)
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
                                    **benchmark_batchprocessing(X_test, Y_test, re, args["M"], f"{i+1}/{args['x']} Applying rejection ensemble for p = {p}", jetson=jetson,verbose=False)
                                }
                            )
                    else:
                        for p in Ps:
                            r = clone(rejector)
                            re = RejectionEnsemble(fsmall, fbig, p=p, rejector = r, train_method=tm, calibration=c)
                            re.fit(X_train_re,Y_train_re)

                            metrics.append( {
                                    "model":"RE",
                                    "batch":True,
                                    "rejector":"DecisionTreeClassifier",
                                    "train_method":tm,
                                    "calibration":c,
                                    "run":i,
                                    "p":p,
                                    **benchmark_batchprocessing(X_test, Y_test, re, args["M"], f"{i+1}/{args['x']} Applying rejection ensemble for p = {p}", jetson=jetson,verbose=False)
                                }
                            )

            # Benchmark small and big model as well
            metrics.append({
                    "model":"small",
                    "batch":True,
                    "rejector":None,
                    "train_method":None,
                    "calibration":None,
                    "run":i,
                    "p":None,
                    **benchmark_batchprocessing(X_test, Y_test, fsmall, args["M"], f"{i+1}/{args['x']} Applying small model", jetson=jetson,verbose=False)
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
                    **benchmark_batchprocessing(X_test, Y_test, fbig, args["M"], f"{i+1}/{args['x']} Applying big model", jetson=jetson,verbose=False)
                }
            )
            print("")

        
        with open(os.path.join(args["out"], f"{d}.json"), "w") as outfile:
            json.dump(metrics, outfile)
    
    if jetson:
        jetson.stop()
        jetson.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running exeriments on sklearn-based rejectors and the UCI datasets.')
    parser.add_argument("--data", help='Name of dataset to be used', required=False, nargs="+", default=["magic"])
    parser.add_argument("--tmp", help='Path to the data. The data will automatically be downloaded to the given folder if not found.', required=False, type=str, default="./data")
    parser.add_argument("--small", help='Small model to be used. Can be dt, rf, boosting.', required=False, type=str, default="dt")
    parser.add_argument("--big", help="Can be dt2, dt3, rf.", required=False, type=str, default="rf")
    parser.add_argument("--rejector", help='Rejector to be used. Currently dt (DecisionTreeClassifier with max_depth = None), rf (RandomForestClassifier with 16 trees and max_depth = None), linear are supported', required=False, type=str, default="dt")
    parser.add_argument("-M", help='Batch size.', required=False, type=int, default=32)
    parser.add_argument("-x", help='Number of cross-validation splits.', required=False, type=int, default=5)
    parser.add_argument("-e", help='If true, energy is measured. This only works on Jetson Boards.', action='store_true')
    parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=[0, 0.5, 1.0])
    parser.add_argument("--out", help='Folder in which to store the output file. Name will be the same as the dataset name, e.g. magic.json.', required=False, type=str, default=".")
    args = vars(parser.parse_args())
    
    main(args)