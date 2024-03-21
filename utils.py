import threading
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch

import tqdm

from RejectionEnsemble import RejectionEnsemble
from TorchRejectionEnsemble import TorchRejectionEnsemble

class JetsonMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.kill = False
        self.lock = threading.Lock()
        self.power = []
        self.power_avg = []

    def run(self):
        from jtop import jtop
        with jtop() as jetson:
            while not self.kill or len(self.power) == 0:
                # make sure to have at-least one measurment
                if jetson.ok():
                    self.lock.acquire()
                    self.power_avg.append(jetson.power['tot']['avg'])
                    self.power.append(jetson.power['tot']['power'])
                    self.lock.release()
                    time.sleep(0.1)
                
    def reset(self):
        self.lock.acquire()
        self.power = []
        self.power_avg = []
        self.lock.release()

    def stop(self):
        self.lock.acquire()
        self.kill = True
        self.lock.release()

    def get_power(self):
        return self.power, self.power_avg

# def benchmark_torch_realtimeprocessing(dataset, model, pbardesc = "", jetson = False, verbose = False):
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers = 6)
#     ypred = []
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if jetson:
#         from jtop import jtop
#         jstats = jtop()
#         jstats.start()

#     runtime = []
#     power_avg = []
#     power = []
#     pcnts = []
#     targets = []
#     with tqdm.tqdm(total=len(dataset), desc = pbardesc, disable=not verbose) as pb:
#         for x, y in data_loader:
#             start = time.time()
#             cur_preds, cnt = model.predict_single(x.to(device), True)
#             runtime.append( time.time() - start )
#             pcnts.append(cnt)

#             ypred.extend(cur_preds.argmax(1).cpu().numpy())
#             targets.append(y.cpu().numpy())
#             pb.update(1)

#             if jetson:
#                 power_avg.append(jstats.power['tot']['avg'])
#                 power.append(jstats.power['tot']['power'])
    
#     if jetson:
#         jstats.close()

#     return {
#         "time":runtime,
#         "f1 macro":f1_score(targets, ypred, average = "macro"),
#         "f1 micro":f1_score(targets, ypred, average = "micro"),
#         "accuracy":accuracy_score(targets, ypred),
#         "p_per_batch":pcnts,
#         "power_per_batch": power if len(power) > 0 else 0,
#         "poweravg_per_batch": power_avg if len(power_avg) > 0 else 0
#     }

def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

def benchmark_batchprocessing(X, y, model, batch_size = 32, pbardesc = "", jetson = None, verbose = False, n_repetitions = 1):
    if jetson:
        jetson.reset()

    for i in range(n_repetitions):
        ypred = []
        runtime = []
        pcnts = []
        with tqdm.tqdm(total=X.shape[0], desc = f"{i}: {pbardesc}", disable=not verbose) as pb:
            for xbatch, ybatch in create_mini_batches(X, y, batch_size):
                start = time.time()
                
                if isinstance(model, RejectionEnsemble):
                    cur_preds, cnt = model.predict_proba(xbatch, True)
                else:
                    cur_preds, cnt = model.predict_proba(xbatch), X.shape[0]

                runtime.append( time.time() - start )
                pcnts.append(cnt/xbatch.shape[0])

                ypred.extend(cur_preds.argmax(1))

                pb.update(xbatch.shape[0])

    if jetson:
        power, power_avg = jetson.get_power()

    return {
        "time":runtime,
        "f1 macro":f1_score(y, ypred, average = "macro"),
        "f1 micro":f1_score(y, ypred, average = "micro"),
        "accuracy":accuracy_score(y, ypred),
        "p_per_batch":pcnts,
        "power_per_batch": power if jetson else 0,
        "poweravg_per_batch": power_avg if jetson else 0
    }

def benchmark_torch_batchprocessing(dataset, model, batch_size = 32, pbardesc = "", jetson = False, verbose = False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 6)
    ypred = []
    
    if jetson:
        jetson.reset()

    runtime = []
    pcnts = []
    targets = []
    with tqdm.tqdm(total=len(dataset), desc = pbardesc, disable=not verbose) as pb:
        for xbatch, ybatch in data_loader:
            start = time.time()
            if isinstance(model, TorchRejectionEnsemble):
                cur_preds, cnt = model.predict_proba(xbatch, True)
            else:
                cur_preds, cnt = model(xbatch), xbatch.shape[0]

            runtime.append( time.time() - start )
            pcnts.append(cnt/xbatch.shape[0])

            ypred.extend(cur_preds.argmax(1).cpu().numpy())
            targets.extend(ybatch.cpu().numpy())

            pb.update(xbatch.shape[0])

    if jetson:
        power, power_avg = jetson.get_power()

    return {
        "time":runtime,
        "f1 macro":f1_score(targets, ypred, average = "macro"),
        "f1 micro":f1_score(targets, ypred, average = "micro"),
        "accuracy":accuracy_score(targets, ypred),
        "p_per_batch":pcnts,
        "power_per_batch": power if jetson else 0,
        "poweravg_per_batch": power_avg if jetson else 0
    }
