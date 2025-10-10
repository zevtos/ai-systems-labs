import torch
from src.models.base_model import BaseModel
from src.metrics.classification_metrics import ClassificationMetrics


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, device: str = 'cpu', reg_type: str = 'none', l1_lambda: float = 0.0, l2_lambda: float = 0.0, alpha: float = 0.0):
        super().__init__(learning_rate, max_epochs, device)
        self.weights = None
        self.bias = None
        self.reg_type = reg_type
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.alpha = alpha

    def fit(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor = None, y_val: torch.Tensor = None):
        self.weights = torch.randn(X.shape[1], device=self.device, requires_grad=False) * 0.01
        self.bias = torch.zeros(1, device=self.device, requires_grad=False)
        
        self.history = {'train_loss': [], 'train_metrics': [], 'val_metrics': []}
        
        with torch.no_grad():
            for _ in range(self.max_epochs):
                y_pred = self.predict_proba(X)
                
                data_loss = self._bce_loss(y, y_pred)
                
                if torch.isnan(data_loss) or torch.isinf(data_loss):
                    print(f"Обучение остановлено на эпохе {_}: обнаружены NaN/inf значения")
                    break
                
                grad_weights, grad_bias = self._compute_gradients(X, y, y_pred)
                
                if self.reg_type == 'l1':
                    grad_weights = grad_weights + self.l1_lambda * torch.sign(self.weights)
                elif self.reg_type == 'l2':
                    grad_weights = grad_weights + 2.0 * self.l2_lambda * self.weights
                elif self.reg_type == 'elasticnet':
                    grad_weights = grad_weights \
                        + (self.alpha * self.l1_lambda) * torch.sign(self.weights) \
                        + 2.0 * ((1 - self.alpha) * self.l2_lambda) * self.weights

                self.weights = self.weights - self.learning_rate * grad_weights
                self.bias = self.bias - self.learning_rate * grad_bias
                
                self.history['train_loss'].append(loss.item())
                train_metrics = self.score(X, y)
                self.history['train_metrics'].append(train_metrics)
                
                if X_val is not None and y_val is not None:
                    val_metrics = self.score(X_val, y_val)
                    self.history['val_metrics'].append(val_metrics)
    
    def _bce_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return torch.mean(loss)
    
    def _compute_gradients(self, X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor):
        grad_weights = (1 / len(X)) * X.T @ (y_pred - y_true)
        grad_bias = (1 / len(X)) * torch.sum(y_pred - y_true)
        
        return grad_weights, grad_bias
    
    def predict_proba(self, X: torch.Tensor):
        with torch.no_grad():
            return torch.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X: torch.Tensor, threshold: float = 0.5):
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).float()
    
    def score(self, X: torch.Tensor, y: torch.Tensor, threshold: float = 0.5):
        y_pred = self.predict(X, threshold)
        y_prob = self.predict_proba(X)
        return ClassificationMetrics.metrics(y, y_pred, y_prob)
