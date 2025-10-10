import torch

class KNN:
    def __init__(self, task: str = 'classification', k: int = 5,
                 p: int = 2, weights: str = 'uniform',
                 device: str = 'cpu', chunk_size: int = 2048):
        assert task in ("classification", "regression")
        assert weights in ("uniform", "distance")
        assert p in (1, 2)
        self.task = task
        self.k = k
        self.p = p
        self.weights = weights
        self.device = device
        self.chunk_size = chunk_size
        self.X = None
        self.y = None
        self.classes_ = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.to(self.device, dtype=torch.float32, non_blocking=True)
        if self.task == 'classification':
            classes, inverse = torch.unique(y, sorted=True, return_inverse=True)
            self.classes_ = classes.to(self.device)
            self.y = inverse.to(self.device, dtype=torch.long)
        else:
            self.y = y.to(self.device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def predict(self, Xq: torch.Tensor, return_prob: bool = False, batch_size: int | None = None):
        assert self.X is not None, "Call fit first."
        Xq = Xq.to(self.device, dtype=torch.float32, non_blocking=True)
        bs = batch_size or self.chunk_size

        outs = []
        prob_outs = [] if (self.task == 'classification' and return_prob) else None
        for s in range(0, Xq.shape[0], bs):
            yhat, probs = self._predict_block(Xq[s:s+bs], return_prob)
            outs.append(yhat)
            if prob_outs is not None:
                prob_outs.append(probs)

        ypred_idx = torch.cat(outs, dim=0)
        if self.task == 'classification':
            ypred_labels = self.classes_[ypred_idx].detach().cpu()
            if prob_outs is not None:
                probs = torch.cat(prob_outs, dim=0).detach().cpu()
                return ypred_labels, probs
            return ypred_labels
        else:
            return ypred_idx.detach().cpu()

    @torch.no_grad()
    def _predict_block(self, Xq: torch.Tensor, return_prob: bool):
        d = torch.cdist(Xq, self.X, p=self.p)
        dk, idx = torch.topk(d, k=self.k, largest=False)
        yk = self.y[idx]

        if self.weights == 'uniform':
            w = torch.ones_like(dk)
        else:
            w = 1.0 / (dk + 1e-12)

        if self.task == 'classification':
            C = int(self.y.max().item()) + 1
            scores = torch.zeros((Xq.shape[0], C), device=self.device, dtype=torch.float32)
            scores.scatter_add_(1, yk, w)
            yhat_idx = scores.argmax(dim=1)
            if return_prob:
                probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-12)
                return yhat_idx, probs
            return yhat_idx, None
        else:
            num = (w * yk.to(torch.float32)).sum(dim=1)
            den = w.sum(dim=1) + 1e-12
            yhat = num / den
            return yhat, None

    @torch.no_grad()
    def score(self, Xq: torch.Tensor, y_true: torch.Tensor):
        if self.task == 'classification':
            y_true = y_true.cpu()
            y_pred = self.predict(Xq)
            return (y_pred == y_true).float().mean().item()
        else:
            y_true = y_true.float().cpu()
            y_pred = self.predict(Xq).float()
            return -torch.mean((y_pred - y_true)**2).item() 