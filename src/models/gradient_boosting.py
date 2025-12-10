import torch
from typing import Optional, List
from src.models.decision_tree import DecisionTree


class GradientBoosting:
    
    def __init__(
        self,
        classification: bool = True,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        device: str = 'cpu'
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.device = device
        
        self.trees: List[DecisionTree] = []
        self.classes_: Optional[torch.Tensor] = None
        self.initial_prediction: Optional[torch.Tensor] = None
        self.n_classes: int = 2
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        
        n_samples, n_features = X.shape
        
        if self.classification:
            classes, y_encoded = torch.unique(y, sorted=True, return_inverse=True)
            self.classes_ = classes
            n_classes = len(classes)
            self.n_classes = n_classes
            
            if n_classes == 2:
                y = y_encoded.float()
                p = y.mean()
                p = torch.clamp(p, 1e-15, 1 - 1e-15)
                self.initial_prediction[c] = torch.log(p)
                current_prediction = self.initial_prediction.clone()
            else:
                y = y_encoded.long()
                self.initial_prediction = torch.zeros(n_classes, device=self.device)
                for c in range(n_classes):
                    p = (y == c).float().mean()
                    p = torch.clamp(p, 1e-15, 1 - 1e-15)
                    self.initial_prediction[c] = torch.log(p)
                current_prediction = self.initial_prediction.unsqueeze(0).repeat(n_samples, 1)
        else:
            y = y.float()
            self.initial_prediction = y.mean()
            current_prediction = torch.full((n_samples,), self.initial_prediction, device=self.device)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            if self.classification:
                if n_classes == 2:
                    probs = torch.sigmoid(current_prediction)
                    negative_gradient = y - probs
                else:
                    probs = torch.softmax(current_prediction, dim=1)
                    negative_gradient = torch.zeros_like(current_prediction)
                    for c in range(n_classes):
                        y_one_hot = (y == c).float()
                        negative_gradient[:, c] = y_one_hot - probs[:, c]
            else:
                negative_gradient = y - current_prediction
            
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                if self.random_state is not None:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(self.random_state + i)
                    indices = torch.randperm(n_samples, generator=generator, device=self.device)[:n_subsample]
                else:
                    indices = torch.randperm(n_samples, device=self.device)[:n_subsample]
                X_train = X[indices]
                if self.classification and n_classes > 2:
                    y_train = negative_gradient[indices]
                else:
                    y_train = negative_gradient[indices]
            else:
                X_train = X
                y_train = negative_gradient
            
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            
            if self.classification and n_classes > 2:
                tree_predictions = torch.zeros((n_samples, n_classes), device=self.device)
                for c in range(n_classes):
                    tree = DecisionTree(
                        classification=False,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        criterion='mse',
                        max_features=None,
                        random_state=tree_random_state,
                        device=self.device
                    )
                    tree.fit(X_train, y_train[:, c])
                    self.trees.append(tree)
                    
                    tree_pred = tree.predict(X)
                    tree_predictions[:, c] = tree_pred
                
                current_prediction = current_prediction + self.learning_rate * tree_predictions
            else:
                tree = DecisionTree(
                    classification=False,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion='mse',
                    max_features=None,
                    random_state=tree_random_state,
                    device=self.device
                )
                
                tree.fit(X_train, y_train)
                self.trees.append(tree)
                
                tree_pred = tree.predict(X)
                current_prediction = current_prediction + self.learning_rate * tree_pred
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if len(self.trees) == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = X.to(self.device)
        n_samples = len(X)
        
        if self.classification:
            if self.n_classes == 2:
                prediction = self.initial_prediction.clone()
                
                for tree in self.trees:
                    tree_pred = tree.predict(X)
                    prediction = prediction + self.learning_rate * tree_pred
                
                probs = torch.sigmoid(prediction)
                result = (probs > 0.5).long()
                result = self.classes_[result]
            else:
                prediction = self.initial_prediction.unsqueeze(0).repeat(n_samples, 1)
                
                tree_idx = 0
                for _ in range(self.n_estimators):
                    for c in range(self.n_classes):
                        tree = self.trees[tree_idx]
                        tree_pred = tree.predict(X)
                        prediction[:, c] = prediction[:, c] + self.learning_rate * tree_pred
                        tree_idx += 1
                
                probs = torch.softmax(prediction, dim=1)
                result = probs.argmax(dim=1)
                result = self.classes_[result]
        else:
            prediction = torch.full((n_samples,), self.initial_prediction, device=self.device)
            
            for tree in self.trees:
                tree_pred = tree.predict(X)
                prediction = prediction + self.learning_rate * tree_pred
            
            result = prediction
        
        return result.cpu()
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.predict(X)
        y = y.cpu()
        
        if self.classification:
            return (y_pred == y).float().mean().item()
        else:
            return -torch.mean((y_pred - y.float()) ** 2).item()

