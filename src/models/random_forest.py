import torch
from typing import Optional, List
from src.models.decision_tree import DecisionTree


class RandomForest:
    
    def __init__(
        self,
        classification: bool = True,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        max_features: Optional[int] = None,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        device: str = 'cpu'
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.device = device
        
        self.trees: List[DecisionTree] = []
        self.classes_: Optional[torch.Tensor] = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        
        if self.classification:
            classes, _ = torch.unique(y, sorted=True, return_inverse=True)
            self.classes_ = classes
        
        n_samples, n_features = X.shape
        
        if self.max_features is None:
            if self.classification:
                self.max_features = max(1, int(n_features / 3))
        
            self.trees = []
        
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            
            tree = DecisionTree(
                classification=self.classification,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_random_state,
                device=self.device
            )
            
            if self.bootstrap:
                if tree_random_state is not None:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(tree_random_state)
                    indices = torch.randint(0, n_samples, (n_samples,), generator=generator, device=self.device)
                else:
                    indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if len(self.trees) == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = X.to(self.device)
        n_samples = len(X)
        
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_predictions.append(predictions)
        
        predictions_tensor = torch.stack(all_predictions)
        
        if self.classification:
            predictions_long = predictions_tensor.long()
            result = torch.zeros(n_samples, dtype=torch.long, device=self.device)
            for i in range(n_samples):
                values, counts = torch.unique(predictions_long[:, i], return_counts=True)
                result[i] = values[counts.argmax()]
            
            result = self.classes_[result]
        else:
            result = predictions_tensor.mean(dim=0)
        
        return result.cpu()
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y_pred = self.predict(X)
        y = y.cpu()
        
        if self.classification:
            return (y_pred == y).float().mean().item()
        else:
            return -torch.mean((y_pred - y.float()) ** 2).item()