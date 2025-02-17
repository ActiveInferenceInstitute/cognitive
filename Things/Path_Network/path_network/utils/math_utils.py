"""
Mathematical utilities for the Path Network simulation.
"""

import numpy as np
from typing import List, Tuple, Union
import torch

def compute_free_energy(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sensory_input: torch.Tensor
) -> torch.Tensor:
    """
    Compute the variational free energy.
    
    Args:
        mu: Expected state
        sigma: Variance of state estimation
        sensory_input: Observed sensory input
        
    Returns:
        Free energy value
    """
    precision = 1.0 / sigma
    prediction_error = sensory_input - mu
    return 0.5 * (prediction_error**2 * precision + torch.log(sigma))

def generalized_coordinates_update(
    coords: torch.Tensor,
    dt: float,
    order: int
) -> torch.Tensor:
    """
    Update generalized coordinates using Taylor series expansion.
    
    Args:
        coords: Current generalized coordinates
        dt: Time step
        order: Order of Taylor expansion
        
    Returns:
        Updated generalized coordinates
    """
    new_coords = coords.clone()
    
    for i in range(order - 1):
        new_coords[i] += coords[i + 1] * dt
    
    return new_coords

def compute_correlation_matrix(
    time_series: List[np.ndarray]
) -> np.ndarray:
    """
    Compute correlation matrix between multiple time series.
    
    Args:
        time_series: List of time series data
        
    Returns:
        Correlation matrix
    """
    n = len(time_series)
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(time_series[i], time_series[j])[0, 1]
    
    return corr_matrix

def compute_prediction_error_metrics(
    predictions: np.ndarray,
    observations: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute various prediction error metrics.
    
    Args:
        predictions: Predicted values
        observations: Observed values
        
    Returns:
        Tuple of (MSE, MAE, RMSE)
    """
    mse = np.mean((predictions - observations) ** 2)
    mae = np.mean(np.abs(predictions - observations))
    rmse = np.sqrt(mse)
    return mse, mae, rmse

def sigmoid_scale(
    x: Union[float, np.ndarray],
    scale: float = 1.0,
    offset: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Apply sigmoid scaling to a value or array.
    
    Args:
        x: Input value(s)
        scale: Scaling factor
        offset: Offset value
        
    Returns:
        Scaled value(s)
    """
    return 1 / (1 + np.exp(-(x - offset) / scale))

def compute_entropy(
    probabilities: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the entropy of a probability distribution.
    
    Args:
        probabilities: Probability distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        Entropy value
    """
    # Ensure probabilities sum to 1 and are positive
    p = np.clip(probabilities, epsilon, 1.0)
    p = p / np.sum(p)
    
    return -np.sum(p * np.log(p))

def exponential_moving_average(
    data: np.ndarray,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Compute exponential moving average of a time series.
    
    Args:
        data: Input time series
        alpha: Smoothing factor (0 < alpha < 1)
        
    Returns:
        Smoothed time series
    """
    result = np.zeros_like(data)
    result[0] = data[0]
    
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t-1]
    
    return result 