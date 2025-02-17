"""
Tests for matrix operations.
"""

import pytest
import numpy as np
from src.models.matrices.matrix_ops import MatrixOps, MatrixLoader, MatrixInitializer

class TestMatrixOps:
    """Test matrix operation utilities."""
    
    def test_normalize_columns(self, sample_matrix_2d):
        """Test column normalization."""
        normalized = MatrixOps.normalize_columns(sample_matrix_2d)
        # Check each column sums to 1
        assert np.allclose(normalized.sum(axis=0), 1.0)
        # Check non-negativity preserved
        assert np.all(normalized >= 0)
    
    def test_normalize_rows(self, sample_matrix_2d):
        """Test row normalization."""
        normalized = MatrixOps.normalize_rows(sample_matrix_2d)
        # Check each row sums to 1
        assert np.allclose(normalized.sum(axis=1), 1.0)
        # Check non-negativity preserved
        assert np.all(normalized >= 0)
    
    def test_ensure_probability_distribution(self):
        """Test probability distribution enforcement."""
        # Test with negative values
        matrix = np.array([[-1, 0.5], [2, 0.5]])
        prob_dist = MatrixOps.ensure_probability_distribution(matrix)
        assert np.all(prob_dist >= 0)
        assert np.allclose(prob_dist.sum(axis=0), 1.0)
    
    def test_compute_entropy(self, sample_belief_vector):
        """Test entropy computation."""
        entropy = MatrixOps.compute_entropy(sample_belief_vector)
        # Entropy should be non-negative
        assert entropy >= 0
        # Test with deterministic distribution
        deterministic = np.array([1.0, 0.0, 0.0])
        assert MatrixOps.compute_entropy(deterministic) == 0
    
    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        P = np.array([0.5, 0.5])
        Q = np.array([0.9, 0.1])
        kl = MatrixOps.compute_kl_divergence(P, Q)
        # KL divergence should be non-negative
        assert kl >= 0
        # KL divergence should be zero for identical distributions
        assert MatrixOps.compute_kl_divergence(P, P) == 0
    
    def test_softmax(self):
        """Test softmax computation."""
        x = np.array([1.0, 2.0, 3.0])
        probs = MatrixOps.softmax(x)
        # Check output is probability distribution
        assert np.allclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        # Check ordering preserved
        assert np.all(np.diff(probs) > 0)

class TestMatrixLoader:
    """Test matrix loading utilities."""
    
    def test_load_spec(self, sample_markdown_spec):
        """Test loading matrix specification from markdown."""
        spec = MatrixLoader.load_spec(sample_markdown_spec)
        assert spec['type'] == 'matrix_spec'
        assert spec['dimensions']['rows'] == 3
        assert 'sum(cols) == 1.0' in spec['shape_constraints']
    
    def test_load_matrix(self, sample_matrix_data):
        """Test loading matrix data from file."""
        matrix = MatrixLoader.load_matrix(sample_matrix_data)
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix.sum(axis=0), 1.0)
    
    def test_validate_matrix(self, sample_matrix_2d, sample_matrix_spec):
        """Test matrix validation against specification."""
        assert MatrixLoader.validate_matrix(sample_matrix_2d, sample_matrix_spec)
        
        # Test invalid matrix
        invalid_matrix = np.array([[1.1, -0.1], [0.2, 1.2]])
        assert not MatrixLoader.validate_matrix(invalid_matrix, sample_matrix_spec)

class TestMatrixInitializer:
    """Test matrix initialization utilities."""
    
    def test_random_stochastic(self):
        """Test random stochastic matrix initialization."""
        shape = (3, 3)
        matrix = MatrixInitializer.random_stochastic(shape)
        # Check dimensions
        assert matrix.shape == shape
        # Check stochastic properties
        assert np.allclose(matrix.sum(axis=0), 1.0)
        assert np.all(matrix >= 0)
    
    def test_identity_based(self):
        """Test identity-based matrix initialization."""
        shape = (3, 3)
        strength = 0.9
        matrix = MatrixInitializer.identity_based(shape, strength)
        # Check dimensions
        assert matrix.shape == shape
        # Check diagonal dominance
        assert np.all(np.diag(matrix) > 0.5)
        # Check stochastic properties
        assert np.allclose(matrix.sum(axis=0), 1.0)
        assert np.all(matrix >= 0)
    
    def test_uniform(self):
        """Test uniform matrix initialization."""
        shape = (3, 3)
        matrix = MatrixInitializer.uniform(shape)
        # Check dimensions
        assert matrix.shape == shape
        # Check uniformity
        assert np.allclose(matrix, 1.0/9.0)
        # Check stochastic properties
        assert np.allclose(matrix.sum(), 1.0)
        assert np.all(matrix >= 0)

@pytest.mark.parametrize("shape,strength", [
    ((2, 2), 0.9),
    ((3, 3), 0.8),
    ((4, 4), 0.7)
])
def test_identity_based_parametrized(shape, strength):
    """Parametrized tests for identity-based initialization."""
    matrix = MatrixInitializer.identity_based(shape, strength)
    assert matrix.shape == shape
    assert np.all(np.diag(matrix) >= strength * 0.9)  # Allow for normalization effects
    assert np.allclose(matrix.sum(axis=0), 1.0)

@pytest.mark.parametrize("distribution,expected_entropy", [
    (np.array([1.0, 0.0]), 0.0),  # Deterministic
    (np.array([0.5, 0.5]), np.log(2)),  # Maximum entropy for 2 states
])
def test_entropy_special_cases(distribution, expected_entropy):
    """Test entropy computation for special cases."""
    computed_entropy = MatrixOps.compute_entropy(distribution)
    assert np.allclose(computed_entropy, expected_entropy) 