import torch
from src.models.base_model import BaseModel
from src.metrics.regression_metrics import RegressionMetrics


class LinearRegression(BaseModel):
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, device: str = 'cpu'):
        super().__init__(learning_rate, max_epochs, device)
        self.weights = None
        self.bias = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor = None, y_val: torch.Tensor = None):
        self.weights = torch.randn(X.shape[1], device=self.device, requires_grad=False) * 0.01
        self.bias = torch.zeros(1, device=self.device, requires_grad=False)
        
        self.history = {'train_loss': [], 'train_metrics': [], 'val_metrics': []}
        
        with torch.no_grad():
            for _ in range(self.max_epochs):
                y_pred = self.predict(X)
                
                diff = y - y_pred
                n = diff.numel()
                sum_sq = (diff * diff).sum()
                loss = RegressionMetrics.mse(sum_sq, n)
                
                grad_weights = (2 / len(X)) * X.T @ (y_pred - y)
                grad_bias = (2 / len(X)) * torch.sum(y_pred - y)
                
                self.weights = self.weights - self.learning_rate * grad_weights
                self.bias = self.bias - self.learning_rate * grad_bias
                
                self.history['train_loss'].append(loss.item())
                train_metrics = self.score(X, y)
                self.history['train_metrics'].append(train_metrics)
                
                if X_val is not None and y_val is not None:
                    val_metrics = self.score(X_val, y_val)
                    self.history['val_metrics'].append(val_metrics)
    
    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            return X @ self.weights + self.bias
    
    def score(self, X: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            return RegressionMetrics.metrics(y, self.predict(X))