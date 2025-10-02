import torch

class RegressionMetrics:
    
    @staticmethod
    def _check(y_true: torch.Tensor, y_pred: torch.Tensor):
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_true.numel() == 0:
            raise ValueError("Empty tensors are not allowed")

    @staticmethod
    def _aggregates(y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Возвращает набор базовых сумм/значений, из которых считаются все метрики.
        """
        RegressionMetrics._check(y_true, y_pred)
        y_true = y_true.to(dtype=torch.float32)
        y_pred = y_pred.to(dtype=torch.float32)

        diff = y_true - y_pred
        n = diff.numel()

        sum_abs = diff.abs().sum()
        sum_sq = (diff * diff).sum()

        if torch.any(y_true == 0):
            mape_sum_abs_rel = None
        else:
            mape_sum_abs_rel = (diff.abs() / y_true.abs()).sum()

        y_mean = y_true.mean()
        ss_res = sum_sq
        ss_tot = ((y_true - y_mean) ** 2).sum()

        return n, sum_abs, sum_sq, mape_sum_abs_rel, ss_res, ss_tot

    @staticmethod
    def mse(sum_sq: torch.Tensor, n: int) -> torch.Tensor:
        return sum_sq / n

    @staticmethod
    def mae(sum_abs: torch.Tensor, n: int) -> torch.Tensor:
        return sum_abs / n

    @staticmethod
    def rmse(mse: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(mse)

    @staticmethod
    def mape(sum_abs_rel: torch.Tensor, n: int) -> torch.Tensor:
        return (100.0 / n) * sum_abs_rel

    @staticmethod
    def r2(ss_res: torch.Tensor, ss_tot: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, dtype=ss_tot.dtype, device=ss_tot.device) if ss_tot == 0 else 1 - ss_res / ss_tot

    @staticmethod
    def metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
        n, sum_abs, sum_sq, mape_sum_abs_rel, ss_res, ss_tot = RegressionMetrics._aggregates(y_true, y_pred)
        mse = RegressionMetrics.mse(sum_sq, n)
        mae = RegressionMetrics.mae(sum_abs, n)
        rmse = RegressionMetrics.rmse(mse)
        r2 = RegressionMetrics.r2(ss_res, ss_tot)
        out = {
            "n": n,
            "sum_abs": sum_abs,
            "sum_sq": sum_sq,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
        if mape_sum_abs_rel is not None:
            out["mape"] = RegressionMetrics.mape(mape_sum_abs_rel, n)
        return out