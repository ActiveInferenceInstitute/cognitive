"""
Utility functions for matrix operations and validation in Active Inference models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def ensure_matrix_properties(matrix: np.ndarray, 
                           constraints: Union[List[str], str, None] = None) -> np.ndarray:
    """Ensure matrix satisfies required properties.
    
    Args:
        matrix: Input matrix to validate/normalize
        constraints: List of constraints or single constraint string
            Supported constraints:
            - 'column_stochastic': Each column sums to 1
            - 'row_stochastic': Each row sums to 1
            - 'non_negative': All values are non-negative
            
    Returns:
        Normalized matrix satisfying specified properties
    """
    if constraints is None:
        constraints = []
    elif isinstance(constraints, str):
        constraints = [constraints]
        
    # Handle non-negative constraint
    if 'non_negative' in constraints:
        matrix = np.maximum(matrix, 0)
    
    # Handle stochastic constraints
    if 'column_stochastic' in constraints:
        # Add small epsilon to avoid division by zero
        sums = matrix.sum(axis=0, keepdims=True)
        sums = np.maximum(sums, 1e-12)
        matrix = matrix / sums
    elif 'row_stochastic' in constraints:
        sums = matrix.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-12)
        matrix = matrix / sums
    
    return matrix

def compute_entropy(distribution: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute entropy of probability distribution.
    
    Args:
        distribution: Probability distribution
        axis: Axis along which to compute entropy
        
    Returns:
        Entropy value(s)
    """
    # Add small epsilon to avoid log(0)
    distribution = np.maximum(distribution, 1e-12)
    return -np.sum(distribution * np.log(distribution), axis=axis)

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax function with temperature.
    
    Args:
        x: Input array
        temperature: Temperature parameter (higher = more uniform)
        
    Returns:
        Softmax probabilities
    """
    x = x / temperature
    exp_x = np.exp(x - x.max())  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid division by zero and log(0)
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    return np.sum(p * np.log(p / q))

def expected_free_energy(A: np.ndarray, 
                        B: np.ndarray,
                        C: np.ndarray,
                        beliefs: np.ndarray,
                        action: int) -> float:
    """Compute expected free energy for an action.
    
    Args:
        A: Observation likelihood matrix
        B: State transition matrix
        C: Preference matrix
        beliefs: Current belief distribution
        action: Action index
        
    Returns:
        Expected free energy value
    """
    # Predicted next state distribution
    pred_states = B[:, :, action] @ beliefs
    
    # Expected observations
    expected_obs = A @ pred_states
    
    # Ambiguity term (entropy over observations)
    ambiguity = compute_entropy(expected_obs)
    
    # Risk term (KL from preferences)
    risk = kl_divergence(expected_obs, softmax(C))
    
    return ambiguity + risk 