import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from jtop import jtop

import tqdm

def benchmark_torchmodel(data_loader, model, pbardesc = "", jetson = False):
    dataset = data_loader.dataset
    ypred = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if jetson:
        jstats = jtop()
        jstats.start()

    runtime = []
    power_avg = []
    power = []
    pcnts = []
    with tqdm.tqdm(total=len(dataset), desc = pbardesc) as pb:
        for xbatch, ybatch in data_loader:
            start = time.time()
            cur_preds, cnt = model.predict_batch(xbatch.to(device), True)
            runtime.append( time.time() - start )
            pcnts.append(cnt/xbatch.shape[0])

            ypred.extend(cur_preds.argmax(1).cpu().numpy())
            pb.update(ybatch.shape[0])

            if jetson:
                power_avg.append(jstats.power['tot']['avg'])
                power.append(jstats.power['tot']['power'])
    
    if jetson:
        jstats.close()

    return {
        "time":np.mean(runtime),
        "f1 macro":f1_score(dataset.targets, ypred, average = "macro"),
        "f1 micro":f1_score(dataset.targets, ypred, average = "micro"),
        "accuracy":accuracy_score(dataset.targets, ypred),
        "preal":np.mean(pcnts) if len(pcnts) > 0 else 0,
        "pmin":np.min(pcnts) if len(pcnts) > 0 else 0,
        "pmax":np.max(pcnts) if len(pcnts) > 0 else 0,
        "power":np.mean(power) if len(power) > 0 else 0,
        "power_avg":np.mean(power_avg) if len(power_avg) > 0 else 0
    }