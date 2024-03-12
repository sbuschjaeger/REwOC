import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm
from sklearn.tree import DecisionTreeClassifier

class RejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector_cfg={"model":"DecisionTreeClassifier"}, return_cnt = False, train_method="confidence", calibration = True):
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector_cfg = rejector_cfg
        self.p = p
        self.return_cnt = return_cnt
        self.train_method = train_method
        self.calibration = calibration
        assert self.train_method in ["confidence", "virtual-labels"]

    def train_pytorch(self, dataset_loader, pbar_desc="",verbose=False):
        X = []
        Y = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with tqdm.tqdm(total=len(dataset_loader.dataset), desc = f"{pbar_desc} Preparing training data", disable=not verbose) as pb:
            for i, (xbatch, ybatch) in enumerate(dataset_loader):
                xbatch = xbatch.to(device)
                ybatch = ybatch.to(device)
                with torch.no_grad():
                    tmp = self.fsmall.features(xbatch)
                    preds_small = self.fsmall.classifier(tmp)

                    X.append(tmp.cpu())
                    if self.train_method == "confidence":
                        mask = preds_small.argmax(1) != ybatch
                        y,_ = preds_small.max(1)
                        y[mask] = 0
                        #y = [preds_small[i].max() if preds_small[i].argmax() == ybatch[i] else 0 for i in range(preds_small.shape[0])]
                        Y.extend(y.cpu().numpy())
                    else:
                        preds_big = self.fbig.predict_batch(xbatch).argmax(1)
                        preds_small = preds_small.argmax(1)
                        
                        for i in range(preds_big.shape[0]):
                            if preds_big[i] == preds_small[i]:
                                Y.append(0)
                            else:
                                if preds_big[i] == ybatch[i]:
                                    Y.append(1)
                                else:
                                    Y.append(0)

                pb.update(ybatch.shape[0])
        X = np.vstack(X)
        Y = np.array(Y)

        rejector_name = self.rejector_cfg.pop("model")
        if rejector_name == "DecisionTreeClassifier":
            rejector = DecisionTreeClassifier(**self.rejector_cfg)
        elif rejector_name == "LogisticRegression":
            rejector = LogisticRegression(**self.rejector_cfg)
        else:
            raise ValueError(f"I do not know the classifier {rejector_name}. Please use another classifier.")
        
        if self.train_method == "confidence":
            P = int(np.floor(self.p * X.shape[0]))
            if P > 0 and P < X.shape[0] and np.unique(Y).shape[0] > 1: 
                # Check if actually use the small and the big model 
                indices_of_bottom_K = np.argsort(Y)[:P]
                targets = np.zeros_like(Y)
                targets[indices_of_bottom_K] = 1
                
                if verbose:
                    print(f"{pbar_desc} Fitting rejector")
                rejector.fit(X,targets)
            else:
                rejector = None
        else:
            if verbose:
                print(f"{pbar_desc} Fitting rejector")
            rejector.fit(X,Y)
            
        self.rejector = rejector

        return self.fsmall, self.fbig, self.rejector

    def predict_single(self, x, return_cnt = False):
        return self.predict_batch(x, return_cnt)

    def predict_batch(self, T, return_cnt = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = T.to(device)

        if self.rejector is None:
            with torch.no_grad():
                if self.p == 1:
                    preds, cnt = self.fbig.predict_batch(T, return_cnt=False), T.shape[0]
                else:
                    preds, cnt = self.fsmall.predict_batch(T, return_cnt=False), 0

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
                    fbig_preds = self.fbig.predict_batch(T[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred