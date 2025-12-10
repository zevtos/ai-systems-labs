import torch
from typing import Optional, Tuple


class Node:
    def __init__(self):
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.value: Optional[float] = None


class DecisionTree:
    
    def __init__(
        self,
        classification: bool = True,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        device: str = 'cpu',
        max_features: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.classification = classification
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.device = device
        self.max_features = max_features
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
        self.root: Optional[Node] = None
        self.classes_: Optional[torch.Tensor] = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        
        if self.classification:
            classes, y_encoded = torch.unique(y, sorted=True, return_inverse=True)
            self.classes_ = classes
            y = y_encoded
        else:
            y = y.float()
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> Node:
        node = Node()
        
        if self._should_stop(X, y, depth):
            node.value = self._answer(y)
            return node
        
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain <= 0:
            node.value = self._answer(y)
            return node
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.value = self._answer(y)
            return node
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _should_stop(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> bool:
        if self.classification:
            if len(torch.unique(y)) == 1:
                return True
        else:
            if torch.allclose(y, y[0]):
                return True
        
        if len(X) < self.min_samples_split:
            return True
        
        if depth >= self.max_depth:
            return True
        
        return False
    
    def _answer(self, y: torch.Tensor) -> float:
        if self.classification:
            values, counts = torch.unique(y, return_counts=True)
            return values[counts.argmax()].item()
        else:
            return y.mean().item()
    
    def _find_best_split(self, X: torch.Tensor, y: torch.Tensor, max_thresholds: int = 32) -> Tuple[int, float, float]:
        device = X.device
        best_gain = -float('inf')
        best_feature = 0
        best_threshold = 0.0

        current_impurity = self._impurity(y)
        n_features = X.shape[1]
        n_samples = len(X)

        if self.max_features is not None and self.max_features < n_features:
            features_to_consider = torch.randperm(n_features, device=device)[:self.max_features]
        else:
            features_to_consider = torch.arange(n_features, device=device)

        for feature_idx in features_to_consider:
            feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
            feature_values = X[:, feature_idx]

            sorted_vals, _ = torch.sort(feature_values)

            unique_values = torch.unique(sorted_vals)

            if unique_values.numel() < 2:
                continue

            midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0

            if midpoints.numel() > max_thresholds:
                qs = torch.linspace(0, 1, steps=max_thresholds + 2, device=device)[1:-1]
                idxs = (qs * (midpoints.numel() - 1)).round().long()
                thresholds = midpoints[idxs]
            else:
                thresholds = midpoints

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                n_left = left_mask.sum().item()
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                left_impurity = self._impurity(y_left)
                right_impurity = self._impurity(y_right)

                gain = current_impurity - (
                    (n_left / n_samples) * left_impurity +
                    (n_right / n_samples) * right_impurity
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold.item())

        return best_feature, best_threshold, best_gain

    
    def _impurity(self, y: torch.Tensor) -> float:
        if len(y) == 0:
            return 0.0
        
        if self.classification:
            if self.criterion == 'gini':
                return self._gini(y)
            elif self.criterion == 'entropy':
                return self._entropy(y)
            else:
                raise ValueError(f"Unknown criterion for classification: {self.criterion}")
        else:
            if self.criterion == 'mse':
                return self._mse(y)
            elif self.criterion == 'mae':
                return self._mae(y)
            else:
                raise ValueError(f"Unknown criterion for regression: {self.criterion}")
    
    def _gini(self, y: torch.Tensor) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = torch.unique(y, return_counts=True)
        probs = counts.float() / len(y)
        return 1.0 - (probs ** 2).sum().item()
    
    def _entropy(self, y: torch.Tensor) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = torch.unique(y, return_counts=True)
        probs = counts.float() / len(y)
        probs = probs[probs > 0]
        return -(probs * torch.log2(probs)).sum().item()
    
    def _mse(self, y: torch.Tensor) -> float:
        if len(y) == 0:
            return 0.0
        mean = y.mean()
        return ((y - mean) ** 2).mean().item()
    
    def _mae(self, y: torch.Tensor) -> float:
        if len(y) == 0:
            return 0.0
        median = y.median()
        return (y - median).abs().mean().item()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.root is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = X.to(self.device)
        predictions = []
        
        for i in range(len(X)):
            node = self.root
            while node.value is None:
                if X[i, node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        
        result = torch.tensor(predictions, device=self.device)
        
        if self.classification:
            result = self.classes_[result.long()]
        
        return result.cpu()
    
    def score(self, X: torch.Tensor, y: torch.Tensor):
        y_pred = self.predict(X)
        y = y.cpu()
        
        if self.classification:
            return (y_pred == y).float().mean().item()
        else:
            return -torch.mean((y_pred - y.float()) ** 2).item()

