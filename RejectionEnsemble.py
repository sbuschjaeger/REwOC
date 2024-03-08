import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm
from sklearn.tree import DecisionTreeClassifier

class RejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector_cfg={"model":"DecisionTreeClassifier"}, return_cnt = False):
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector_cfg = rejector_cfg
        self.p = p
        self.return_cnt = return_cnt

    def train_pytorch(self, dataset_loader, pbar_desc=""):
        X = []
        Y = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with tqdm.tqdm(total=len(dataset_loader.dataset), desc = f"{pbar_desc} Preparing training data") as pb:
            for i, (xbatch, ybatch) in enumerate(dataset_loader):
                xbatch = xbatch.to(device)
                ybatch = ybatch.to(device)
                with torch.no_grad():
                    preds_small = self.fsmall.predict_batch(xbatch, False)
                    mask = preds_small.argmax(1) != ybatch
                    y,_ = preds_small.max(1)
                    y[mask] = 0
                    #y = [preds_small[i].max() if preds_small[i].argmax() == ybatch[i] else 0 for i in range(preds_small.shape[0])]
                    Y.extend(y.cpu().numpy())
                    X.append(self.fsmall.features(xbatch).cpu())

                pb.update(ybatch.shape[0])
        X = np.vstack(X)
        Y = np.array(Y)

        P = int(np.floor(self.p * X.shape[0]))
        if P > 0 and P < X.shape[0] and np.unique(Y).shape[0] > 1: 
            # Check if actually use the small and the big model 
            indices_of_bottom_K = np.argsort(Y)[:P]
            targets = np.zeros_like(Y)
            targets[indices_of_bottom_K] = 1
            
            rejector_name = self.rejector_cfg.pop("model")
            if rejector_name == "DecisionTreeClassifier":
                rejector = DecisionTreeClassifier(**self.rejector_cfg)
            elif rejector_name == "LogisticRegression":
                rejector = LogisticRegression(**self.rejector_cfg)
            else:
                raise ValueError(f"I do not know the classifier {rejector_name}. Please use another classifier.")
            
            print(f"{pbar_desc} Fitting rejector")
            rejector.fit(X,targets)
            
            self.rejector = rejector
        else:
            self.rejector = None

        return self.fsmall, self.fbig, self.rejector

    # def __call__(self, T):
    #     return self.predict_batch_optimized(T, self.return_cnt)

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
                tmp = self.fsmall.features(T)
                
                # Only move necessary data to CPU for rejector
                r_pred = self.rejector.predict_proba(tmp.cpu().numpy())

            # Convert to tensor for efficient computation
            r_pred_tensor = torch.tensor(r_pred, device=device)
            
            # Determine indices for Ts and Tb using boolean masks
            Ts_mask = r_pred_tensor.argmax(dim=1) == 0
            Tb_mask = r_pred_tensor.argmax(dim=1) == 1

            # Predict in batches for fbig and fsmall
            with torch.no_grad():
                fsmall_preds = self.fsmall.classifier(tmp[Ts_mask])

                ypred = torch.empty((T.shape[0], fsmall_preds.shape[1]), dtype=fsmall_preds.dtype, device=device)
                ypred[Ts_mask] = fsmall_preds
                if not torch.all(Tb_mask == False):
                    fbig_preds = self.fbig.predict_batch(T[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred