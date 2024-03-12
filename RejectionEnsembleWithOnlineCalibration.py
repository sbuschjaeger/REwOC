import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm
from sklearn.tree import DecisionTreeClassifier

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

class RejectionEnsembleWithOnlineCalibration():
    def __init__(self, fsmall, fbig, p, rejector_cfg={"model":"DecisionTreeClassifier"}, return_cnt = False):
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector_cfg = rejector_cfg
        self.return_cnt = return_cnt
        self.p = p

        self.t = 0.5
        self.step_size = 0.02

    def reset(self):
        self.t = 0.5
        self.step_size = 0.02

    def train_pytorch(self, dataset_loader, pbar_desc="", verbose=False):
        X = []
        Y = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with tqdm.tqdm(total=len(dataset_loader.dataset), desc = f"{pbar_desc} Preparing training data", disable=not verbose) as pb:
            for i, (xbatch, ybatch) in enumerate(dataset_loader):
                xbatch = xbatch.to(device)
                ybatch = ybatch.to(device)
                with torch.no_grad():
                    tmp = self.fsmall.features(xbatch)
                    X.append(tmp.cpu())
                    preds_small = self.fsmall.classifier(tmp).argmax(1) 
                    preds_big = self.fbig.predict_batch(xbatch).argmax(1)

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

        rejector_name = self.rejector_cfg.pop("model")
        if rejector_name == "DecisionTreeClassifier":
            rejector = DecisionTreeClassifier(**self.rejector_cfg)
        elif rejector_name == "LogisticRegression":
            rejector = LogisticRegression(**self.rejector_cfg)
        else:
            raise ValueError(f"I do not know the classifier {rejector_name}. Please use another classifier.")
        
        if verbose:
            print(f"{pbar_desc} Fitting rejector")
        rejector.fit(X,Y)
        
        self.rejector = rejector

        return self.fsmall, self.fbig, rejector

    def predict_single(self, x, return_cnt = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        with torch.no_grad():
            tmp = self.fsmall.features(x)
            r_pred = self.rejector.predict_proba(tmp.cpu().numpy())
            r = r_pred[0][1]

            phat = sigmoid(r - self.t)
            grad = -2*(phat - self.p)*phat*(1-phat)
            self.t = self.t - self.step_size * grad
            
            if r > self.t:
                ypred = self.fbig.predict_single(x)
            else:
                ypred = self.fsmall.classifier(tmp)
            
            if return_cnt:
                return ypred, 1 if r > self.t else 0 
            else:
                return ypred
            
    def predict_batch(self, T, return_cnt = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = T.to(device)
        with torch.no_grad():
            tmp = self.fsmall.features(T)
            
            # Only move necessary data to CPU for rejector
            r_pred = self.rejector.predict_proba(tmp.cpu().numpy())

        M = len(T)
        P = int(np.floor(self.p * M))

        # Convert to tensor for efficient computation
        r_pred_tensor = torch.tensor(r_pred, device=device)
        
        # Determine indices for Ts and Tb using boolean masks
        #Ts_mask = r_pred_tensor.argmax(dim=1) == 0
        Tb_mask = r_pred_tensor.argmax(dim=1) == 1
        #Tb_mask = torch.tensor([True for _ in range(M)], device=device, dtype=torch.bool)

        # Assuming Tb_mask is not empty, and we have already computed Tb_confidence and Tb_indices

        # Sort Tb based on confidence and slice according to p, now properly utilizing Tb_indices
        _, Tb_sorted_indices = torch.sort(r_pred_tensor[Tb_mask, 1], descending=True)
        Tb_indices = Tb_mask.nonzero(as_tuple=True)[0][Tb_sorted_indices]

        # Allocate the top P samples to Tb and the rest to Ts, if Tb_indices has fewer entries than P, use all for Tb
        if len(Tb_indices) <= P:
            Tb_final_indices = Tb_indices
        else:
            Tb_final_indices = Tb_indices[:P]

        # Update masks based on final indices for Tb
        Tb_final_mask = torch.zeros(M, dtype=torch.bool, device=device)
        Tb_final_mask[Tb_final_indices] = True
        Ts_final_mask = ~Tb_final_mask  # This simplifies the logic for assigning the rest to Ts

        # Predict in batches for fbig and fsmall
        with torch.no_grad():
            fsmall_preds = self.fsmall.classifier(tmp[Ts_final_mask])

            ypred = torch.empty((M, fsmall_preds.shape[1]), dtype=fsmall_preds.dtype, device=device)
            ypred[Ts_final_mask] = fsmall_preds
            if not torch.all(Tb_final_mask == False):
                fbig_preds = self.fbig.classifier(T[Tb_final_mask])
                ypred[Tb_final_mask] = fbig_preds
        
        if return_cnt:
            return ypred, len(Tb_final_indices)
        else:
            return ypred