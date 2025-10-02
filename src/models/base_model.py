from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    def __init__(self, learning_rate: float, max_epochs: int, device: str):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.device = device
        self.history = {'train_loss': [], 'train_metrics': [], 'val_metrics': []}
        
    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor):
        pass
    
    @abstractmethod
    def score(self, X: torch.Tensor, y: torch.Tensor):
        pass