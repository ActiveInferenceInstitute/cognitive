"""Test suite for Generic Thing implementation."""

import logging
import os

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'test_suite.log')),
        logging.StreamHandler()
    ]
)

# Create visualization output directory
viz_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

"""Tests for Generic Thing package.""" 