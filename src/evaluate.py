"""
Model Evaluation and Metric Utilities.

Provides standardized evaluation for classification and time-series forecasting models,
generating confusion matrix plots and ROC curves.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def evaluate_classifier(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Computes standard classification performance metrics.
    
    Args:
        y_true: True binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_prob: Predicted positive class probabilities.
        model_name: Identifier name for the model.
        
    Returns:
        Dict of computed evaluation metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path="reports/figures/confusion_matrix.png"):
    """Plots and saves normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
        cbar=True
    )
    plt.title(f"Normalized Confusion Matrix - {model_name}", fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", labelpad=8)
    plt.ylabel("True Label", labelpad=8)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(model_curves: dict, save_path="reports/figures/roc_curves.png"):
    """
    Plots multiple ROC curves on a single figure.
    
    Args:
        model_curves: Dict of {model_name: (y_true, y_prob, auc_score)}
    """
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob, score) in model_curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
    plt.ylabel("True Positive Rate (Recall)", labelpad=10)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_forecast(y_true, y_pred, model_name="ForecastModel"):
    """
    Computes time series forecasting accuracy metrics.
    
    Args:
        y_true: Ground truth series / array.
        y_pred: Predicted values series / array.
        model_name: Model identifier string.
        
    Returns:
        Dict with RMSE, MAE, and MAPE.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    # Avoid division by zero in MAPE
    mask = y_t > 0
    mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0.0
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    
    return {
        "model": model_name,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2)
    }
