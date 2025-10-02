import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_training_curves(history, title="Training Curves", model_type="classification", max_epochs_display=None):
    def to_numpy(data):
        return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    
    def plot_metric(ax, train_data, val_data, title, ylabel):
        ax.plot(train_data, label='Train', color='blue', linewidth=2)
        if val_data:
            ax.plot(val_data, label='Val', color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    train_loss = to_numpy(history['train_loss'])
    train_metrics = history.get('train_metrics', [])
    val_metrics = history.get('val_metrics', [])
    
    if max_epochs_display is not None and len(train_loss) > max_epochs_display:
        total_epochs = len(train_loss)
        step = total_epochs // max_epochs_display
        indices = list(range(0, total_epochs, step))[:max_epochs_display]
        
        train_loss = np.array(train_loss)[indices]
        train_metrics = [train_metrics[i] for i in indices]
        val_metrics = [val_metrics[i] for i in indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    val_loss = [m['mse' if model_type == "regression" else 'log_loss'].item() for m in val_metrics] if val_metrics else []
    plot_metric(axes[0, 0], train_loss, val_loss, 'Падение ошибок', 'Loss')
    
    if model_type == "classification":
        train_acc = [m['accuracy'].item() for m in train_metrics] if train_metrics else []
        val_acc = [m['accuracy'].item() for m in val_metrics] if val_metrics else []
        plot_metric(axes[0, 1], train_acc, val_acc, 'Рост точности', 'Accuracy')
        
        train_f1 = [m['f1_score'].item() for m in train_metrics] if train_metrics else []
        val_f1 = [m['f1_score'].item() for m in val_metrics] if val_metrics else []
        plot_metric(axes[1, 0], train_f1, val_f1, 'Рост F1-Score', 'F1-Score')
        
        train_prec = [m['precision'].item() for m in train_metrics] if train_metrics else []
        train_rec = [m['recall'].item() for m in train_metrics] if train_metrics else []
        val_prec = [m['precision'].item() for m in val_metrics] if val_metrics else []
        val_rec = [m['recall'].item() for m in val_metrics] if val_metrics else []
        
        axes[1, 1].plot(train_prec, label='Train Precision', color='cyan', linewidth=2)
        axes[1, 1].plot(train_rec, label='Train Recall', color='magenta', linewidth=2)
        if val_prec:
            axes[1, 1].plot(val_prec, label='Val Precision', color='darkblue', linewidth=2, linestyle='--')
            axes[1, 1].plot(val_rec, label='Val Recall', color='darkred', linewidth=2, linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision и Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        train_r2 = [m['r2'].item() for m in train_metrics] if train_metrics else []
        val_r2 = [m['r2'].item() for m in val_metrics] if val_metrics else []
        plot_metric(axes[0, 1], train_r2, val_r2, 'Рост R² Score', 'R² Score')
        
        train_mae = [m['mae'].item() for m in train_metrics] if train_metrics else []
        val_mae = [m['mae'].item() for m in val_metrics] if val_metrics else []
        plot_metric(axes[1, 0], train_mae, val_mae, 'Падение MAE', 'MAE')
        
        train_rmse = [m['rmse'].item() for m in train_metrics] if train_metrics else []
        val_rmse = [m['rmse'].item() for m in val_metrics] if val_metrics else []
        plot_metric(axes[1, 1], train_rmse, val_rmse, 'Падение RMSE', 'RMSE')
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, title="Predictions vs True Values"):
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_feature_importance(weights, feature_names=None, title="Feature Importance"):
    weights = weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else weights
    weights = weights.flatten() if weights.ndim > 1 else weights
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(weights))]
    
    if len(weights) != len(feature_names):
        raise ValueError(f"Количество весов ({len(weights)}) не соответствует количеству названий признаков ({len(feature_names)})")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'weight': weights,
        'abs_weight': np.abs(weights)
    }).sort_values('abs_weight', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['abs_weight'], 
             color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (|Weight|)')
    plt.ylabel('Features')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['abs_weight'] + 0.01, i, f'{row["abs_weight"]:.3f}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print("\nСоответствие признаков и весов:")
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        print(f"  {row['feature']}: {row['weight']:.4f}")