import torch
from src.models.base_model import BaseModel
from src.metrics.classification_metrics import ClassificationMetrics


class SVM(BaseModel):
    def __init__(self, learning_rate=0.001, max_epochs=100, batch_size=64, C=1.0, device='cpu'):
        super().__init__(learning_rate, max_epochs, device)
        self.batch_size = batch_size
        self.C = C
        self.weights = None
        self.bias = None
        self.history = {'train_loss': [], 'train_metrics': [], 'val_metrics': []}
        self.threshold = 0.0

    def _hinge_loss(self, X, y):
        margins = 1 - y * (X @ self.weights + self.bias)
        losses = torch.clamp(margins, min=0)
        return 0.5 * torch.sum(self.weights ** 2) + self.C * torch.mean(losses)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        if self.weights is None or self.bias is None:
            self.weights = torch.zeros(n_features, device=self.device)
            self.bias = torch.zeros(1, device=self.device)

        X, y = X.to(self.device), y.to(self.device)
        y = torch.where(y == 0, -1, y).float()

        for epoch in range(self.max_epochs):
            permutation = torch.randperm(n_samples)
            total_loss = 0.0

            for i in range(0, n_samples, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                X_batch, y_batch = X[indices], y[indices]

                preds = X_batch @ self.weights + self.bias
                margins = 1 - y_batch * preds
                mask = margins > 0

                grad_w = self.weights - self.C * (X_batch[mask].T @ y_batch[mask]) / len(X_batch)
                grad_b = -self.C * torch.sum(y_batch[mask]) / len(X_batch)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

                batch_loss = self._hinge_loss(X_batch, y_batch)
                total_loss += batch_loss.item()

            avg_loss = total_loss / (n_samples / self.batch_size)
            self.history['train_loss'].append(avg_loss)

            train_metrics = self.score(X, y)
            self.history['train_metrics'].append(train_metrics)

            if X_val is not None and y_val is not None:
                val_metrics = self.score(X_val, y_val)
                self.history['val_metrics'].append(val_metrics)

            if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                print(f"Epoch {epoch+1}/{self.max_epochs}, loss={avg_loss:.4f}")


    def decision_function(self, X):
        X = X.to(self.device)
        return X @ self.weights + self.bias  # raw margins

    def predict(self, X):
        scores = self.decision_function(X)
        return torch.where(scores > self.threshold, 1.0, 0.0)

    def score(self, X, y):
        X = X.to(self.device)
        y01 = torch.where(y > 0, 1.0, 0.0).to(self.device)
        y_pred = self.predict(X)
        logits = X @ self.weights + self.bias
        y_prob = torch.sigmoid(logits)
        return ClassificationMetrics.metrics(y01, y_pred, y_prob)
