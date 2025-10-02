import torch

class ClassificationMetrics:
    @staticmethod
    def _confusion_counts(y_true: torch.Tensor, y_pred: torch.Tensor):
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        y_true = y_true.bool()
        y_pred = y_pred.bool()
        tp = (y_true &  y_pred).sum().to(dtype=torch.float32)
        fp = (~y_true & y_pred).sum().to(dtype=torch.float32)
        tn = (~y_true & ~y_pred).sum().to(dtype=torch.float32)
        fn = (y_true & ~y_pred).sum().to(dtype=torch.float32)
        return tp, fp, tn, fn

    @staticmethod
    def accuracy(tp, fp, tn, fn):
        denom = tp + fp + tn + fn
        return torch.zeros_like(denom) if denom == 0 else (tp + tn) / denom

    @staticmethod
    def precision(tp, fp):
        denom = tp + fp
        return torch.zeros_like(denom) if denom == 0 else tp / denom

    @staticmethod
    def recall(tp, fn):
        denom = tp + fn
        return torch.zeros_like(denom) if denom == 0 else tp / denom

    @staticmethod
    def f1(precision, recall):
        denom = precision + recall
        return torch.zeros_like(denom) if denom == 0 else 2 * (precision * recall) / denom
    
    @staticmethod
    def log_loss(y_true: torch.Tensor, y_prob: torch.Tensor):
        if y_true.shape != y_prob.shape:
            raise ValueError("y_true and y_prob must have the same shape")
        eps = 1e-15
        y_prob = torch.clamp(y_prob, eps, 1 - eps)
        n = y_true.numel()
        if n == 0:
            raise ValueError("Empty tensors are not allowed")
        loss = y_true * torch.log(y_prob) + (1 - y_true) * torch.log(1 - y_prob)
        return -loss.sum() / n

    @staticmethod
    def metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor = None):
        tp, fp, tn, fn = ClassificationMetrics._confusion_counts(y_true, y_pred)
        acc = ClassificationMetrics.accuracy(tp, fp, tn, fn)
        prec = ClassificationMetrics.precision(tp, fp)
        rec = ClassificationMetrics.recall(tp, fn)
        f1 = ClassificationMetrics.f1(prec, rec)
        
        # Log loss требует вероятности, если они не предоставлены, используем y_pred как вероятности
        if y_prob is not None:
            log_loss = ClassificationMetrics.log_loss(y_true, y_prob)
        else:
            # Если вероятности не предоставлены, не вычисляем log_loss
            log_loss = torch.tensor(0.0)
        
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "log_loss": log_loss
        }
