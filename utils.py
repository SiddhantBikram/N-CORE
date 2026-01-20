import os
import random
import numpy as np
import torch
from typing import List, Optional, Union

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(
    y_true: List[int],
    y_score_probs: np.ndarray,
    num_classes: int
) -> float:
    """
    Calculate AUC score for multi-class classification.
    
    Args:
        y_true: True labels
        y_score_probs: Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        AUC score
    """
    if not y_true or not isinstance(y_score_probs, np.ndarray) or y_score_probs.size == 0:
        return 0.0
    
    try:
        y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
        
        if y_true_binarized.shape[1] == 0 and num_classes > 0:
            return 0.0
        
        if num_classes == 2:
            score_for_positive = (
                y_score_probs[:, 1]
                if y_score_probs.ndim > 1 and y_score_probs.shape[1] == 2
                else y_score_probs
            )
            return roc_auc_score(y_true, score_for_positive)
        elif num_classes > 2:
            return roc_auc_score(y_true_binarized, y_score_probs, average='macro', multi_class='ovr')
        else:
            return 0.0
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[np.ndarray]] = None,
    num_classes: Optional[int] = None
) -> dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted probability scores (optional)
        num_classes: Number of classes (required if y_scores provided)
        
    Returns:
        Dictionary with accuracy, f1, uar, and optionally auc
    """
    if not y_true or not y_pred:
        return {'accuracy': 0.0, 'f1': 0.0, 'uar': 0.0, 'auc': 0.0}
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'uar': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    if y_scores is not None and num_classes is not None:
        y_scores_array = np.array(y_scores)
        metrics['auc'] = calculate_auc(y_true, y_scores_array, num_classes)
    else:
        metrics['auc'] = 0.0
    
    return metrics


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking training statistics.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        mode: 'max' for metrics where higher is better, 'min' for lower is better
        delta: Minimum change to qualify as improvement
    """
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device: Specific device string, or None for auto-detection
        
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
