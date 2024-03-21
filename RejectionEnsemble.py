import numpy as np

class RejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector, train_method="confidence", calibration=True):
        """
        Initialize a RejectionEnsemble object.

        Args:
            fsmall: The already trained small model. This object must offer predict_proba method similar to scikit-learn models
            fbig: The already big small model. This object must offer predict_proba method similar to scikit-learn models
            p (float): The probability threshold for rejection.
            rejector: The rejector object used for rejection. This object must offer a fit and predict_proba method similar to scikit-learn models
            train_method (str, optional): The training method to use. Defaults to "confidence".
            calibration (bool, optional): Whether to perform calibration. Defaults to True.

        Raises:
            AssertionError: If the train_method is not one of ["confidence", "virtual-labels"].
        """
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector = rejector
        self.p = p
        self.train_method = train_method
        self.calibration = calibration
        assert self.train_method in ["confidence", "virtual-labels"]

    def fit(self, X, Y):
        """
        Fits the RejectionEnsemble model to the given training data. The return value can be ignored if the object is maintained after calling fit.

        Parameters:
        - X: The input features of shape (n_samples, n_features).
        - Y: The target labels of shape (n_samples,).

        Returns:
        - fsmall: The fitted small model, which has been provided in the constructor.
        - fbig: The fitted big model, which has been provided in the constructor
        - rejector: The fitted rejector model. None if no rejector was fitted.
        """
        preds_small = self.fsmall.predict_proba(X)
        self.n_classes_ = len(set(Y))

        if self.train_method == "confidence":
            mask = preds_small.argmax(1) != Y
            y = preds_small.max(1)
            y[mask] = 0
            targets = y

            P = int(np.floor(self.p * X.shape[0]))
            if P > 0 and P < X.shape[0] and np.unique(Y).shape[0] > 1: 
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

    def predict_proba(self, T, return_cnt=False):
        """
        Predict class probabilities for the input samples.

        Parameters:
            T (array-like): The input samples of shape (n_samples, n_features).
            return_cnt (bool, optional): Whether to return the useage of the big model. Default is False.

        Returns:
            array-like or tuple: If `return_cnt` is False, returns an array-like object containing the predicted class probabilities for each sample.
                                 If `return_cnt` is True, returns a tuple containing the predicted class probabilities and the useage count of big model.

        Raises:
            None

        Notes:
            - If `self.rejector` is None, the method uses the predictions from either `self.fbig` or `self.fsmall` models based on the value of `self.p`.
            - If `self.rejector` is not None, the method uses the predictions from `self.rejector` model and applies rejection calibration if `self.calibration` is True.
        """
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