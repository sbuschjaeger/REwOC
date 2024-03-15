import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm
from sklearn.tree import DecisionTreeClassifier

class SKRejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector, train_method="confidence", calibration = True):
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector = rejector
        self.p = p
        self.train_method = train_method
        self.calibration = calibration
        assert self.train_method in ["confidence", "virtual-labels"]

    def fit(self, X, Y):
        preds_small = self.fsmall.predict_proba(X)
        self.n_classes_ = len(set(Y))

        if self.train_method == "confidence":
            mask = preds_small.argmax(1) != Y
            y = preds_small.max(1)
            y[mask] = 0
            #y = [preds_small[i].max() if preds_small[i].argmax() == ybatch[i] else 0 for i in range(preds_small.shape[0])]
            targets = y

            P = int(np.floor(self.p * X.shape[0]))
            if P > 0 and P < X.shape[0] and np.unique(Y).shape[0] > 1: 
                # Check if actually use the small and the big model 
                indices_of_bottom_K = np.argsort(Y)[:P]
                targets = np.zeros_like(Y)
                targets[indices_of_bottom_K] = 1
                
                self.rejector.fit(X,targets)
            else:
                self.rejector = None
        else:
            preds_big = self.fbig.predict_proba(X).argmax(1)
            preds_small = preds_small.argmax(1)
            targets = []
            for i in range(preds_big.shape[0]):
                if preds_big[i] == preds_small[i]:
                    targets.append(0)
                else:
                    if preds_big[i] == Y[i]:
                        targets.append(1)
                    else:
                        targets.append(0)
            
            targets = np.array(targets)
            if np.unique(targets).shape[0] > 1:
                self.rejector.fit(X,targets)
            else:
                self.rejector = None
        
        return self.fsmall, self.fbig, self.rejector

    def predict_proba(self, T, return_cnt = False):
        if self.rejector is None:
            if self.p == 1:
                preds, cnt = self.fbig.predict_proba(T), T.shape[0]
            else:
                preds, cnt = self.fsmall.predict_proba(T), 0

            if return_cnt:
                return preds, cnt
            else:
                return preds
        else:
            r_pred = self.rejector.predict_proba(T)

            if self.calibration:
                M = len(T)
                P = int(np.floor(self.p * M))

                Tb_sorted_indices = np.argsort(r_pred[:,1])[::-1]
                Tb_sorted_indices = Tb_sorted_indices[:P]

                Tb_mask = np.zeros(M, dtype=bool)
                Tb_mask[Tb_sorted_indices] = True
                Ts_mask = ~Tb_mask  
            else:
                Ts_mask = r_pred.argmax(1) == 0
                Tb_mask = r_pred.argmax(1) == 1

            ypred = np.zeros((T.shape[0], self.n_classes_))
            if not np.all(Ts_mask == False):
                fsmall_preds = self.fsmall.predict_proba(T[Ts_mask])
                ypred[Ts_mask] = fsmall_preds

            if not np.all(Tb_mask == False):
                fbig_preds = self.fbig.predict_proba(T[Tb_mask])
                ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred