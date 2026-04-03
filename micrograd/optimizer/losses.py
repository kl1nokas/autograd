# losses.py
import numpy as np
from engine import Tensor

def mse_loss(pred, target):
    """
    Mean Squared Error loss
    
    Args:
        pred: Tensor, predictions from model
        target: Tensor, ground truth values
    
    Returns:
        Tensor, scalar loss value
    """
    diff = pred - target
    return (diff * diff).mean()


def binary_cross_entropy(pred, target, eps=1e-8):
    """
    Binary Cross Entropy loss for binary classification
    
    Args:
        pred: Tensor, predicted probabilities (0 to 1)
        target: Tensor, ground truth (0 or 1)
        eps: float, small value to avoid log(0)
    
    Returns:
        Tensor, scalar loss value
    """
    pred_data = pred.data
    target_data = target.data
    
    pred_data = np.clip(pred_data, eps, 1 - eps)
    
    loss_data = -(target_data * np.log(pred_data) + (1 - target_data) * np.log(1 - pred_data))
    loss = Tensor(np.mean(loss_data), (pred, target), 'binary_cross_entropy')
    
    def _backward():
        grad = (pred_data - target_data) / (pred_data * (1 - pred_data) + eps) / pred_data.size
        pred.grad += grad * loss.grad
    
    loss._backward = _backward
    return loss


def cross_entropy(pred, target):
    """
    Categorical Cross Entropy for multi-class classification
    
    Args:
        pred: Tensor, logits (before softmax)
        target: Tensor, class indices or one-hot encoded
    
    Returns:
        Tensor, scalar loss value
    """
    pred_data = pred.data
    
    exp_pred = np.exp(pred_data - np.max(pred_data, axis=-1, keepdims=True))
    softmax = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
    
    if len(target.data.shape) == 1:
        n_samples = pred_data.shape[0]
        loss_data = -np.log(softmax[np.arange(n_samples), target.data.astype(int)] + 1e-8)
    else:
        loss_data = -np.sum(target.data * np.log(softmax + 1e-8), axis=-1)
    
    loss = Tensor(np.mean(loss_data), (pred, target), 'cross_entropy')
    
    def _backward():
        grad = (softmax - target.data) / pred_data.shape[0]
        pred.grad += grad * loss.grad
    
    loss._backward = _backward
    return loss


def mae_loss(pred, target):
    """
    Mean Absolute Error loss
    
    Args:
        pred: Tensor, predictions
        target: Tensor, ground truth
    
    Returns:
        Tensor, scalar loss value
    """
    diff = pred - target
    return (diff.abs()).mean()


def huber_loss(pred, target, delta=1.0):
    """
    Huber loss (combines MSE and MAE)
    
    Args:
        pred: Tensor, predictions
        target: Tensor, ground truth
        delta: float, threshold for switching between MSE and MAE
    
    Returns:
        Tensor, scalar loss value
    """
    diff = pred - target
    abs_diff = diff.abs()
    
    quadratic = (diff * diff) * 0.5
    linear = delta * (abs_diff - 0.5 * delta)
    
    loss_data = np.where(abs_diff.data <= delta, quadratic.data, linear.data)
    loss = Tensor(np.mean(loss_data), (pred, target), 'huber')
    
    def _backward():
        grad = np.where(abs_diff.data <= delta, diff.data, delta * np.sign(diff.data))
        pred.grad += grad / diff.data.size * loss.grad
        target.grad -= grad / diff.data.size * loss.grad
    
    loss._backward = _backward
    return loss