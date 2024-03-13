import threading
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch

import tqdm

class JetsonMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.power = []
        self.power_avg = []
        self.sleep_sec = 0.01

    def run(self):
        from jtop import jtop
        with jtop() as jetson:
            while not self.stop_event.is_set():
                if jetson.ok():
                    self.power_avg.append(jetson.power['tot']['avg'])
                    self.power.append(jetson.power['tot']['power'])
            
            time.sleep(self.sleep_sec)

    def stop(self):
        self.stop_event.set()

    def get_power(self):
        return self.power, self.power_avg

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
        jm = JetsonMonitor()
        jm.start()

    runtime = []
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

    if jetson:
        power, power_avg = jm.get_power()
        jm.stop()
        jm.join()

    return {
        "time":runtime,
        "f1 macro":f1_score(targets, ypred, average = "macro"),
        "f1 micro":f1_score(targets, ypred, average = "micro"),
        "accuracy":accuracy_score(targets, ypred),
        "p_per_batch":pcnts,
        "power_per_batch": power if jetson else 0,
        "poweravg_per_batch": power_avg if jetson else 0
    }
