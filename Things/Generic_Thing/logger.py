"""Logging configuration for Generic Thing."""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'generic_thing_{timestamp}.log'
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s:%(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    logger.info(f"Log file: {log_file}")

# Create module logger
logger = logging.getLogger(__name__)

# Set up logging if this is the main module
if __name__ == "__main__":
    setup_logging() 