"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml
import shutil
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

@pytest.fixture(scope="session")
def output_dir():
    """Provide dedicated output directory for test artifacts."""
    output_path = Path("Output").absolute()
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nTest outputs will be saved to: {output_path}")
    
    # Clean previous test outputs
    for file in output_path.glob("*"):
        if file.is_file():
            print(f"Cleaning up previous test file: {file}")
            file.unlink()
    
    yield output_path
    
    # Report saved files after tests complete
    saved_files = list(output_path.glob("*"))
    if saved_files:
        print("\nFiles generated during testing:")
        for file in saved_files:
            print(f"  - {file.absolute()}")
            print(f"    Size: {file.stat().st_size} bytes")
            print(f"    Type: {file.suffix[1:] if file.suffix else 'unknown'}")
    else:
        print("\nNo files were generated during testing.")

@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def sample_matrix_2d():
    """Sample 2D matrix for testing."""
    return np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8]
    ])

@pytest.fixture
def sample_matrix_3d():
    """Sample 3D matrix for testing."""
    return np.array([
        [[0.9, 0.1, 0.0],
         [0.2, 0.8, 0.0],
         [0.1, 0.1, 0.8]],
        [[0.6, 0.4, 0.0],
         [0.4, 0.6, 0.0],
         [0.0, 0.0, 1.0]]
    ])

@pytest.fixture
def sample_belief_vector():
    """Sample belief vector for testing."""
    return np.array([0.3, 0.4, 0.3])

@pytest.fixture
def sample_matrix_spec():
    """Sample matrix specification for testing."""
    return {
        'type': 'matrix_spec',
        'dimensions': {'rows': 3, 'cols': 3},
        'shape_constraints': ['sum(cols) == 1.0', 'all_values >= 0']
    }

@pytest.fixture
def sample_markdown_spec(tmp_path):
    """Create a sample markdown specification file."""
    spec_file = tmp_path / "test_spec.md"
    content = """---
type: matrix_spec
dimensions:
  rows: 3
  cols: 3
shape_constraints:
  - sum(cols) == 1.0
  - all_values >= 0
---

# Matrix Specification
"""
    spec_file.write_text(content)
    return spec_file

@pytest.fixture
def sample_matrix_data(tmp_path):
    """Create a sample matrix data file."""
    data_file = tmp_path / "test_matrix.npy"
    matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8]
    ])
    np.save(data_file, matrix)
    return data_file 