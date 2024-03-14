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

        if self.train_method == "confidence":
            mask = preds_small.argmax(1) != Y
            y,_ = preds_small.max(1)
            y[mask] = 0
            #y = [preds_small[i].max() if preds_small[i].argmax() == ybatch[i] else 0 for i in range(preds_small.shape[0])]
            targets = y
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

        self.X = np.vstack(X)
        targets = np.array(targets)
        
        tmp_cfg = self.rejector_cfg.copy()
        rejector_name = tmp_cfg.pop("model")

        if rejector_name == "DecisionTreeClassifier":
            rejector = DecisionTreeClassifier(**tmp_cfg)
        elif rejector_name == "LogisticRegression":
            rejector = LogisticRegression(**tmp_cfg)
        else:
            raise ValueError(f"I do not know the classifier {rejector_name}. Please use another classifier.")
        
        if self.train_method == "confidence":
            P = int(np.floor(self.p * self.X.shape[0]))
            if P > 0 and P < self.X.shape[0] and np.unique(self.Y).shape[0] > 1: 
                # Check if actually use the small and the big model 
                indices_of_bottom_K = np.argsort(self.Y)[:P]
                targets = np.zeros_like(self.Y)
                targets[indices_of_bottom_K] = 1
                
                rejector.fit(self.X,targets)
            else:
                rejector = None
        else:
            rejector.fit(self.X,self.Y)
            
        self.rejector = rejector

        return self.fsmall, self.fbig, self.rejector

    def predict_proba(self, T, return_cnt = False):
        if self.rejector is None:
            if self.p == 1:
                preds, cnt = self.fbig.predict_batch(T, return_cnt=False), T.shape[0]
            else:
                preds, cnt = self.fsmall.predict_batch(T, return_cnt=False), 0

            if return_cnt:
                return preds, cnt
            else:
                return preds
        else:
            r_pred = self.rejector.predict_proba(T)

            if self.calibration:
                M = len(T)
                P = int(np.floor(self.p * M))

                # Determine indices for Ts and Tb using boolean masks
                _, Tb_sorted_indices = np.sort(r_pred[:, 1], descending=True)
                Tb_sorted_indices = Tb_sorted_indices[:P]

                Tb_mask = np.array(M, dtype=np.bool)
                Tb_mask[Tb_sorted_indices] = True
                Ts_mask = ~Tb_mask  
            else:
                Ts_mask = r_pred.argmax(dim=1) == 0
                Tb_mask = r_pred.argmax(dim=1) == 1

            with torch.no_grad():
                fsmall_preds = self.fsmall.predict_proba(T[Ts_mask])

                ypred = np.empty((T.shape[0], fsmall_preds.shape[1]), dtype=fsmall_preds.dtype)
                ypred[Ts_mask] = fsmall_preds
                if not np.all(Tb_mask == False):
                    fbig_preds = self.fbig.predict_proba(T[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred