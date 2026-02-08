"""
Core matrix operations for Active Inference computations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class MatrixOps:
    """Core matrix operations for Active Inference."""
    
    @staticmethod
    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix columns to sum to 1.
        
        Args:
            matrix: Input matrix to normalize
            
        Returns:
            Normalized matrix with columns summing to 1
            
        Example:
            >>> matrix = np.array([[1, 2], [3, 4]])
            >>> normalized = MatrixOps.normalize_columns(matrix)
            >>> np.allclose(normalized.sum(axis=0), 1.0)
            True
        """
        logger.debug(f"Normalizing columns of matrix with shape {matrix.shape}")
        result = matrix / (matrix.sum(axis=0) + 1e-12)
        logger.debug(f"Column normalization complete. Column sums: {result.sum(axis=0)}")
        return result
    
    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix rows to sum to 1.
        
        Args:
            matrix: Input matrix to normalize
            
        Returns:
            Normalized matrix with rows summing to 1
            
        Example:
            >>> matrix = np.array([[1, 2], [3, 4]])
            >>> normalized = MatrixOps.normalize_rows(matrix)
            >>> np.allclose(normalized.sum(axis=1), 1.0)
            True
        """
        logger.debug(f"Normalizing rows of matrix with shape {matrix.shape}")
        result = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)
        logger.debug(f"Row normalization complete. Row sums: {result.sum(axis=1)}")
        return result
    
    @staticmethod
    def ensure_probability_distribution(matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix represents valid probability distribution.
        
        Ensures non-negative values and normalizes columns to sum to 1.
        
        Args:
            matrix: Input matrix to validate and normalize
            
        Returns:
            Valid probability distribution matrix
            
        Example:
            >>> matrix = np.array([[-1, 2], [3, 4]])
            >>> prob_dist = MatrixOps.ensure_probability_distribution(matrix)
            >>> np.all(prob_dist >= 0) and np.allclose(prob_dist.sum(axis=0), 1.0)
            True
        """
        logger.debug(f"Ensuring probability distribution for matrix with shape {matrix.shape}")
        if np.any(matrix < 0):
            logger.warning(f"Found {np.sum(matrix < 0)} negative values, setting to zero")
        matrix = np.maximum(matrix, 0)  # Non-negative
        result = MatrixOps.normalize_columns(matrix)
        logger.debug(f"Probability distribution validation complete. Column sums: {result.sum(axis=0)}")
        return result
    
    @staticmethod
    def compute_entropy(distribution: np.ndarray) -> float:
        """Compute entropy of probability distribution.
        
        Computes Shannon entropy: H(X) = -Σ p(x) * log(p(x))
        
        Args:
            distribution: Probability distribution array
            
        Returns:
            Entropy value in bits (using natural log)
            
        Example:
            >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
            >>> entropy = MatrixOps.compute_entropy(uniform)
            >>> np.isclose(entropy, -4 * 0.25 * np.log(0.25))
            True
        """
        logger.debug(f"Computing entropy for distribution with shape {distribution.shape}")
        # Handle zero probabilities
        nonzero_probs = distribution[distribution > 0]
        if len(nonzero_probs) == 0:
            logger.debug("Distribution has no non-zero probabilities, entropy is 0")
            return 0.0
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
        logger.debug(f"Computed entropy: {entropy:.4f}")
        return entropy
    
    @staticmethod
    def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """Compute KL divergence between distributions.
        
        Computes D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        
        Args:
            P: First probability distribution
            Q: Second probability distribution (must have same shape as P)
            
        Returns:
            KL divergence value (non-negative)
            
        Raises:
            ValueError: If P and Q have different shapes
            
        Example:
            >>> P = np.array([0.5, 0.5])
            >>> Q = np.array([0.25, 0.75])
            >>> kl = MatrixOps.compute_kl_divergence(P, Q)
            >>> kl > 0
            True
        """
        if P.shape != Q.shape:
            logger.error(f"Shape mismatch: P has shape {P.shape}, Q has shape {Q.shape}")
            raise ValueError(f"Distributions must have same shape. Got P: {P.shape}, Q: {Q.shape}")
        logger.debug(f"Computing KL divergence between distributions with shape {P.shape}")
        kl = np.sum(P * (np.log(P + 1e-12) - np.log(Q + 1e-12)))
        logger.debug(f"Computed KL divergence: {kl:.4f}")
        return kl
    
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
        """Validate matrix against its specification.
        
        Checks matrix dimensions and constraints against specification.
        
        Args:
            matrix: Matrix to validate
            spec: Specification dictionary with 'dimensions' and 'shape_constraints'
            
        Returns:
            True if matrix matches specification, False otherwise
            
        Example:
            >>> matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
            >>> spec = {'dimensions': {'rows': 2, 'cols': 2}, 
            ...         'shape_constraints': ['sum(cols) == 1.0', 'all_values >= 0']}
            >>> MatrixLoader.validate_matrix(matrix, spec)
            True
        """
        logger.debug(f"Validating matrix with shape {matrix.shape} against specification")
        
        # Check dimensions
        if 'dimensions' in spec:
            expected_shape = [spec['dimensions'][d] for d in ['rows', 'cols']]
            if matrix.shape != tuple(expected_shape):
                logger.warning(f"Dimension mismatch: expected {expected_shape}, got {matrix.shape}")
                return False
            logger.debug(f"Dimension check passed: {matrix.shape}")
        
        # Check constraints
        if 'shape_constraints' in spec:
            constraints = spec['shape_constraints']
            if 'sum(cols) == 1.0' in constraints:
                col_sums = matrix.sum(axis=0)
                if not np.allclose(col_sums, 1.0):
                    logger.warning(f"Column sum constraint violated. Column sums: {col_sums}")
                    return False
                logger.debug("Column sum constraint satisfied")
            if 'all_values >= 0' in constraints:
                if not np.all(matrix >= 0):
                    negative_count = np.sum(matrix < 0)
                    logger.warning(f"Non-negativity constraint violated. Found {negative_count} negative values")
                    return False
                logger.debug("Non-negativity constraint satisfied")
        
        logger.debug("Matrix validation passed")
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