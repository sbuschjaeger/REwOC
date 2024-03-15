import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm
from sklearn.tree import DecisionTreeClassifier

def get_predictions(f, data_loader, return_embeddings = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_emebddings = []
    for xbatch, _ in data_loader:
        xbatch = xbatch.to(device)
        with torch.no_grad():
            if return_embeddings:
                emebddings = f.features(xbatch)
                preds = f.classifier(emebddings)
                all_emebddings.append(emebddings)
            else:
                preds = f(xbatch)
            
            all_preds.append(preds)
    
    all_preds = torch.vstack(all_preds).cpu().numpy()

    if return_embeddings:
        all_emebddings = torch.vstack(all_emebddings).cpu().numpy()
        return all_preds, all_emebddings
    else:
        return all_preds

class TorchRejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector, train_method="confidence", calibration = True):
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector = rejector
        self.p = p
        self.train_method = train_method
        self.calibration = calibration
        assert self.train_method in ["confidence", "virtual-labels"]

    def fit(self, dataset_loader):
        preds_small, X = get_predictions(self.fsmall, dataset_loader, True)
        Y = [yi for _, yb in dataset_loader for yi in yb]

        if self.train_method != "confidence":
            preds_big = get_predictions(self.fbig, dataset_loader, False)
        else:
            preds_big= None

        return self._fit(X, Y, preds_small, preds_big)

    def _fit(self, X, Y, preds_small, preds_big):
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
            preds_big = preds_big.argmax(1)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = T.to(device)

        if self.rejector is None:
            with torch.no_grad():
                if self.p == 1:
                    preds, cnt = self.fbig(T), T.shape[0]
                else:
                    preds, cnt = self.fsmall(T), 0

                if return_cnt:
                    return preds, cnt
                else:
                    return preds
        else:
            with torch.no_grad():
                x_features = self.fsmall.features(T)
                r_pred = self.rejector.predict_proba(x_features.cpu().numpy())

            if self.calibration:
                M = len(T)
                P = int(np.floor(self.p * M))

                r_pred_tensor = torch.tensor(r_pred, device=device)
                
                # Determine indices for Ts and Tb using boolean masks
                _, Tb_sorted_indices = torch.sort(r_pred_tensor[:, 1], descending=True)
                Tb_sorted_indices = Tb_sorted_indices[:P]

                Tb_mask = torch.zeros(M, dtype=torch.bool, device=device)
                Tb_mask[Tb_sorted_indices] = True
                Ts_mask = ~Tb_mask  
            else:
                r_pred_tensor = torch.tensor(r_pred, device=device)
                
                Ts_mask = r_pred_tensor.argmax(dim=1) == 0
                Tb_mask = r_pred_tensor.argmax(dim=1) == 1

            with torch.no_grad():
                fsmall_preds = self.fsmall.classifier(x_features[Ts_mask])

                ypred = torch.empty((T.shape[0], fsmall_preds.shape[1]), dtype=fsmall_preds.dtype, device=device)
                ypred[Ts_mask] = fsmall_preds
                if not torch.all(Tb_mask == False):
                    fbig_preds = self.fbig(T[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred