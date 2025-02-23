"""
Test runner for Generic Thing test suite with coverage and visualization.
"""

import os
import sys
import unittest
import pytest
import coverage
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for test outputs."""
    base_dir = Path(__file__).parent.parent
    dirs = {
        'reports': base_dir / 'reports',
        'coverage': base_dir / 'reports' / 'coverage',
        'test_results': base_dir / 'reports' / 'test_results',
        'visualizations': base_dir / 'visualizations',
        'logs': base_dir / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def run_tests_with_coverage():
    """Run tests with coverage tracking."""
    cov = coverage.Coverage(
        branch=True,
        source=['../'],
        omit=['*/tests/*', '*/run_tests.py'],
        config_file=False
    )
    
    # Start coverage tracking
    cov.start()
    
    # Run tests with pytest
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pytest.main([
        '--verbose',
        '--html=reports/test_results/report_{}.html'.format(timestamp),
        '--self-contained-html',
        '--capture=tee-sys'
    ])
    
    # Stop coverage tracking
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    cov.html_report(directory='reports/coverage')
    
    return cov.report()

def main():
    """Main test runner function."""
    try:
        logger.info("Setting up test environment...")
        dirs = setup_directories()
        
        logger.info("Running tests with coverage...")
        coverage_result = run_tests_with_coverage()
        
        logger.info(f"Test suite completed. Coverage: {coverage_result}%")
        logger.info("Reports generated in reports/ directory")
        
        return 0
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 