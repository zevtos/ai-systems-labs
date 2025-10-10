import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history, title="Training Curves", model_type="regression", max_epochs_display=None):
    """Unified API used by notebooks to plot training curves.
    Delegates to regression/classification-specific plotters and supports optional epoch truncation.
    """
    if max_epochs_display is None:
        sliced = history
    else:
        # Create a shallow-sliced copy to avoid mutating original history
        sliced = {
            'train_loss': history.get('train_loss', [])[:max_epochs_display],
            'train_metrics': history.get('train_metrics', [])[:max_epochs_display],
            'val_metrics': history.get('val_metrics', [])[:max_epochs_display],
        }
    if model_type == "classification":
        return plot_classification_history(sliced, title=title)
    return plot_training_history(sliced, title=title)

def plot_predictions(y_true, y_pred, title="Predictions vs True Values"):
    """Scatter plot comparing predictions with true values for regression tasks."""
    y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_pred_np = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_np, y_pred_np, alpha=0.5, s=12, label='Predictions')
    min_v = min(np.min(y_true_np), np.min(y_pred_np))
    max_v = max(np.max(y_true_np), np.max(y_pred_np))
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(weights, feature_names, title="Feature Importance (Absolute Weights)"):
    """Bar chart of absolute weights with features sorted by magnitude."""
    w = weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else np.asarray(weights)
    abs_w = np.abs(w)
    order = np.argsort(abs_w)[::-1]
    w_sorted = w[order]
    abs_sorted = abs_w[order]
    names_sorted = [feature_names[i] for i in order]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(abs_sorted)), abs_sorted, color='steelblue', edgecolor='black')
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=60, ha='right')
    plt.ylabel('Absolute Weight')
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Simple 2x2 confusion matrix heatmap for binary classification."""
    yt = y_true.detach().cpu().to(dtype=torch.bool) if isinstance(y_true, torch.Tensor) else torch.tensor(y_true).bool()
    yp = y_pred.detach().cpu().to(dtype=torch.bool) if isinstance(y_pred, torch.Tensor) else torch.tensor(y_pred).bool()
    tp = (yt & yp).sum().item()
    fp = ((~yt) & yp).sum().item()
    tn = ((~yt) & (~yp)).sum().item()
    fn = (yt & (~yp)).sum().item()
    mat = np.array([[tn, fp], [fn, tp]], dtype=int)
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['Pred 0', 'Pred 1'])
    plt.yticks([0, 1], ['True 0', 'True 1'])
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha='center', va='center', color='black', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics over epochs"""
    def to_numpy(data):
        return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if history['train_loss']:
        train_loss = to_numpy(history['train_loss'])
        ax1.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_loss = [m['mse'].item() for m in history['val_metrics']]
        ax1.plot(val_loss, label='Validation Loss', color='red', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² Score
    ax2 = axes[0, 1]
    if history['train_metrics']:
        train_r2 = [m['r2'].item() for m in history['train_metrics']]
        ax2.plot(train_r2, label='Training R²', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_r2 = [m['r2'].item() for m in history['val_metrics']]
        ax2.plot(val_r2, label='Validation R²', color='red', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MAE
    ax3 = axes[1, 0]
    if history['train_metrics']:
        train_mae = [m['mae'].item() for m in history['train_metrics']]
        ax3.plot(train_mae, label='Training MAE', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_mae = [m['mae'].item() for m in history['val_metrics']]
        ax3.plot(val_mae, label='Validation MAE', color='red', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.set_title('Mean Absolute Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RMSE
    ax4 = axes[1, 1]
    if history['train_metrics']:
        train_rmse = [m['rmse'].item() for m in history['train_metrics']]
        ax4.plot(train_rmse, label='Training RMSE', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_rmse = [m['rmse'].item() for m in history['val_metrics']]
        ax4.plot(val_rmse, label='Validation RMSE', color='red', linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Root Mean Square Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_classification_history(history, title="Classification Training History"):
    """Plot training and validation metrics for classification"""
    def to_numpy(data):
        return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if history['train_loss']:
        train_loss = to_numpy(history['train_loss'])
        ax1.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_loss = [m['log_loss'].item() for m in history['val_metrics']]
        ax1.plot(val_loss, label='Validation Loss', color='red', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    if history['train_metrics']:
        train_acc = [m['accuracy'].item() for m in history['train_metrics']]
        ax2.plot(train_acc, label='Training Accuracy', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_acc = [m['accuracy'].item() for m in history['val_metrics']]
        ax2.plot(val_acc, label='Validation Accuracy', color='red', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    ax3 = axes[1, 0]
    if history['train_metrics']:
        train_f1 = [m['f1_score'].item() for m in history['train_metrics']]
        ax3.plot(train_f1, label='Training F1', color='blue', linewidth=2)
    
    if history['val_metrics']:
        val_f1 = [m['f1_score'].item() for m in history['val_metrics']]
        ax3.plot(val_f1, label='Validation F1', color='red', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision & Recall
    ax4 = axes[1, 1]
    if history['train_metrics']:
        train_prec = [m['precision'].item() for m in history['train_metrics']]
        train_rec = [m['recall'].item() for m in history['train_metrics']]
        ax4.plot(train_prec, label='Training Precision', color='blue', linewidth=2)
        ax4.plot(train_rec, label='Training Recall', color='green', linewidth=2)
    
    if history['val_metrics']:
        val_prec = [m['precision'].item() for m in history['val_metrics']]
        val_rec = [m['recall'].item() for m in history['val_metrics']]
        ax4.plot(val_prec, label='Validation Precision', color='red', linewidth=2)
        ax4.plot(val_rec, label='Validation Recall', color='orange', linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision & Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _plot_compare(histories: dict, key: str, title: str, ylabel: str):
    """Simple plotting function for comparing metrics across regularization types"""
    plt.figure(figsize=(8, 6))
    for name, hist in histories.items():
        data = None
        # Direct key exists (e.g., 'train_loss')
        if key in hist and isinstance(hist[key], list) and len(hist[key]) > 0:
            vals = hist[key]
            if hasattr(vals[0], 'item'):
                data = [v.item() if hasattr(v, 'item') else v for v in vals]
            elif isinstance(vals[0], (float, int)):
                data = vals
        # Virtual mapping: 'val_*' -> extract from 'val_metrics'
        if data is None and key.startswith('val_') and 'val_metrics' in hist and len(hist['val_metrics']) > 0:
            base_key = key[len('val_'):]
            vals = hist['val_metrics']
            if isinstance(vals[0], dict):
                # Map aliases commonly used in notebooks/friend code
                candidates = [base_key]
                if base_key == 'loss':
                    candidates = ['log_loss', 'mse', 'loss']
                if base_key == 'acc':
                    candidates = ['accuracy']
                if base_key == 'f1':
                    candidates = ['f1_score']
                for cand in candidates:
                    if cand in vals[0]:
                        data = [m[cand].item() if hasattr(m[cand], 'item') else m[cand] for m in vals]
                        break
        # Virtual mapping: 'train_*' -> extract from 'train_metrics'
        if data is None and key.startswith('train_') and 'train_metrics' in hist and len(hist['train_metrics']) > 0:
            base_key = key[len('train_'):]
            vals = hist['train_metrics']
            if isinstance(vals[0], dict):
                candidates = [base_key]
                if base_key == 'loss':
                    candidates = ['log_loss', 'mse', 'loss']
                if base_key == 'acc':
                    candidates = ['accuracy']
                if base_key == 'f1':
                    candidates = ['f1_score']
                for cand in candidates:
                    if cand in vals[0]:
                        data = [m[cand].item() if hasattr(m[cand], 'item') else m[cand] for m in vals]
                        break
        if data is not None:
            style = {'label': name, 'linewidth': 2}
            if isinstance(data, (list, tuple)) and len(data) <= 2:
                style['marker'] = 'o'
            plt.plot(data, **style)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_regularization_comparison(histories, title="Regularization Comparison", model_type="regression"):
    """
    Plot comparison of different regularization types (None, L1, L2, ElasticNet)
    Uses the simpler plotting approach
    """
    print(f"Plotting {model_type} regularization comparison...")
    
    # Debug: Print what we have in histories
    print("Debug - History keys:", list(histories.keys()))
    for reg_type in histories:
        print(f"{reg_type}: {list(histories[reg_type].keys())}")
        if 'train_loss' in histories[reg_type]:
            print(f"  train_loss length: {len(histories[reg_type]['train_loss'])}")
        if 'val_metrics' in histories[reg_type] and len(histories[reg_type]['val_metrics']) > 0:
            print(f"  val_metrics keys: {list(histories[reg_type]['val_metrics'][0].keys())}")
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Training Loss (top-left)
    ax1 = axes[0, 0]
    for name, hist in histories.items():
        if 'train_loss' in hist and len(hist['train_loss']) > 0:
            data = hist['train_loss']
            if isinstance(data, list) and len(data) > 0:
                if hasattr(data[0], 'item'):
                    data = [d.item() if hasattr(d, 'item') else d for d in data]
            style = {'label': name, 'linewidth': 2}
            if isinstance(data, (list, tuple)) and len(data) <= 2:
                style['marker'] = 'o'
            ax1.plot(data, **style)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if model_type == "regression":
        # Plot 2: Validation MSE (top-right)
        ax2 = axes[0, 1]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['mse'].item() if hasattr(m['mse'], 'item') else m['mse'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax2.plot(data, **style)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation MSE')
        ax2.set_title('Validation MSE Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation R² Score (bottom-left)
        ax3 = axes[1, 0]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['r2'].item() if hasattr(m['r2'], 'item') else m['r2'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax3.plot(data, **style)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation R² Score')
        ax3.set_title('Validation R² Score Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation MAE (bottom-right)
        ax4 = axes[1, 1]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['mae'].item() if hasattr(m['mae'], 'item') else m['mae'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax4.plot(data, **style)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation MAE')
        ax4.set_title('Validation MAE Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    else:  # classification
        # Plot 2: Validation Accuracy (top-right)
        ax2 = axes[0, 1]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['accuracy'].item() if hasattr(m['accuracy'], 'item') else m['accuracy'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax2.plot(data, **style)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation F1 Score (bottom-left)
        ax3 = axes[1, 0]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['f1_score'].item() if hasattr(m['f1_score'], 'item') else m['f1_score'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax3.plot(data, **style)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation F1 Score')
        ax3.set_title('Validation F1 Score Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation Precision (bottom-right)
        ax4 = axes[1, 1]
        for name, hist in histories.items():
            if 'val_metrics' in hist and len(hist['val_metrics']) > 0:
                data = [m['precision'].item() if hasattr(m['precision'], 'item') else m['precision'] for m in hist['val_metrics']]
                style = {'label': name, 'linewidth': 2}
                if isinstance(data, (list, tuple)) and len(data) <= 2:
                    style['marker'] = 'o'
                ax4.plot(data, **style)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Precision')
        ax4.set_title('Validation Precision Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_knn_performance(k_values, metrics, title="KNN Performance", metric_name="Metric"):
    """Plot KNN performance vs k value"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, metrics, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Highlight best k
    best_idx = np.argmax(metrics) if metric_name in ['Accuracy', 'R² Score'] else np.argmin(metrics)
    best_k = k_values[best_idx]
    best_metric = metrics[best_idx]
    plt.plot(best_k, best_metric, 'ro', markersize=12, label=f'Best k={best_k}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()