import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch

import tqdm

def benchmark_torch_realtimeprocessing(dataset, model, pbardesc = "", jetson = False, verbose = False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers = 6)
    ypred = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if jetson:
        from jtop import jtop
        jstats = jtop()
        jstats.start()

    runtime = []
    power_avg = []
    power = []
    pcnts = []
    targets = []
    with tqdm.tqdm(total=len(dataset), desc = pbardesc, disable=not verbose) as pb:
        for x, y in data_loader:
            start = time.time()
            cur_preds, cnt = model.predict_single(x.to(device), True)
            runtime.append( time.time() - start )
            pcnts.append(cnt)

            ypred.extend(cur_preds.argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
            pb.update(1)

            if jetson:
                power_avg.append(jstats.power['tot']['avg'])
                power.append(jstats.power['tot']['power'])
    
    if jetson:
        jstats.close()

    return {
        "time":runtime,
        "f1 macro":f1_score(targets, ypred, average = "macro"),
        "f1 micro":f1_score(targets, ypred, average = "micro"),
        "accuracy":accuracy_score(targets, ypred),
        "p_per_batch":pcnts,
        "power_per_batch": power if len(power) > 0 else 0,
        "poweravg_per_batch": power_avg if len(power_avg) > 0 else 0
    }

def benchmark_torch_batchprocessing(dataset, model, batch_size = 32, pbardesc = "", jetson = False, verbose = False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 6)
    ypred = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if jetson:
        from jtop import jtop
        jstats = jtop()
        jstats.start()

    runtime = []
    power_avg = []
    power = []
    pcnts = []
    targets = []
    with tqdm.tqdm(total=len(dataset), desc = pbardesc, disable=not verbose) as pb:
        for xbatch, ybatch in data_loader:
            start = time.time()
            cur_preds, cnt = model.predict_batch(xbatch.to(device), True)
            runtime.append( time.time() - start )
            pcnts.append(cnt/xbatch.shape[0])

            ypred.extend(cur_preds.argmax(1).cpu().numpy())
            targets.extend(ybatch.cpu().numpy())

            pb.update(xbatch.shape[0])

            if jetson and jstats.ok():
                power_avg.append(jstats.power['tot']['avg'])
                power.append(jstats.power['tot']['power'])
    
    if jetson:
        jstats.close()

    return {
        "time":runtime,
        "f1 macro":f1_score(targets, ypred, average = "macro"),
        "f1 micro":f1_score(targets, ypred, average = "micro"),
        "accuracy":accuracy_score(targets, ypred),
        "p_per_batch":pcnts,
        "power_per_batch": power if len(power) > 0 else 0,
        "poweravg_per_batch": power_avg if len(power_avg) > 0 else 0
    }
