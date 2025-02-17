"""
Core matrix operations for Active Inference computations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import yaml

class MatrixOps:
    """Core matrix operations for Active Inference."""
    
    @staticmethod
    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix columns to sum to 1."""
        return matrix / (matrix.sum(axis=0) + 1e-12)
    
    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix rows to sum to 1."""
        return matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)
    
    @staticmethod
    def ensure_probability_distribution(matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix represents valid probability distribution."""
        matrix = np.maximum(matrix, 0)  # Non-negative
        return MatrixOps.normalize_columns(matrix)
    
    @staticmethod
    def compute_entropy(distribution: np.ndarray) -> float:
        """Compute entropy of probability distribution."""
        # Handle zero probabilities
        nonzero_probs = distribution[distribution > 0]
        if len(nonzero_probs) == 0:
            return 0.0
        return -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    @staticmethod
    def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """Compute KL divergence between distributions."""
        return np.sum(P * (np.log(P + 1e-12) - np.log(Q + 1e-12)))
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MatrixLoader:
    """Utility for loading and validating matrices."""
    
    @staticmethod
    def load_spec(spec_path: Path) -> Dict:
        """Load matrix specification from markdown file."""
        with open(spec_path, 'r') as f:
            content = f.read()
            
        # Extract YAML frontmatter
        if content.startswith('---'):
            _, frontmatter, _ = content.split('---', 2)
            return yaml.safe_load(frontmatter)
        return {}
    
    @staticmethod
    def load_matrix(data_path: Path) -> np.ndarray:
        """Load matrix data from storage."""
        return np.load(data_path)
    
    @staticmethod
    def validate_matrix(matrix: np.ndarray, spec: Dict) -> bool:
        """Validate matrix against its specification."""
        # Check dimensions
        if 'dimensions' in spec:
            expected_shape = [spec['dimensions'][d] for d in ['rows', 'cols']]
            if matrix.shape != tuple(expected_shape):
                return False
        
        # Check constraints
        if 'shape_constraints' in spec:
            constraints = spec['shape_constraints']
            if 'sum(cols) == 1.0' in constraints:
                if not np.allclose(matrix.sum(axis=0), 1.0):
                    return False
            if 'all_values >= 0' in constraints:
                if not np.all(matrix >= 0):
                    return False
        
        return True

class MatrixInitializer:
    """Initialize matrices with specific properties."""
    
    @staticmethod
    def random_stochastic(shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize random stochastic matrix."""
        matrix = np.random.rand(*shape)
        return MatrixOps.normalize_columns(matrix)
    
    @staticmethod
    def identity_based(shape: Tuple[int, ...], strength: float = 0.9) -> np.ndarray:
        """Initialize near-identity transition matrix."""
        n = shape[0]
        # Ensure off-diagonal elements are small enough to preserve strength after normalization
        off_diag_strength = (1 - strength) / (n - 1)
        matrix = np.full(shape, off_diag_strength)
        np.fill_diagonal(matrix, strength)
        return matrix  # Already normalized by construction
    
    @staticmethod
    def uniform(shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize uniform distribution matrix."""
        return np.ones(shape) / np.prod(shape)

class MatrixVisualizer:
    """Visualization utilities for matrices."""
    
    @staticmethod
    def prepare_heatmap_data(matrix: np.ndarray) -> Dict:
        """Prepare matrix data for heatmap visualization."""
        return {
            'data': matrix,
            'x_ticks': range(matrix.shape[1]),
            'y_ticks': range(matrix.shape[0])
        }
    
    @staticmethod
    def prepare_bar_data(vector: np.ndarray) -> Dict:
        """Prepare vector data for bar visualization."""
        return {
            'data': vector,
            'x_ticks': range(len(vector))
        }
    
    @staticmethod
    def prepare_multi_heatmap_data(tensor: np.ndarray) -> Dict:
        """Prepare 3D tensor data for multiple heatmap visualization."""
        return {
            'slices': [tensor[i] for i in range(tensor.shape[0])],
            'x_ticks': range(tensor.shape[2]),
            'y_ticks': range(tensor.shape[1])
        } 