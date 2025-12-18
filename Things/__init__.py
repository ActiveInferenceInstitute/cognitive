"""
Things: Active Inference Agent Implementations

A comprehensive collection of cognitive agent implementations demonstrating Active Inference
principles across multiple domains and complexity levels. This package provides modular,
well-documented, and professionally engineered implementations of autonomous agents.

Package Structure:
- Generic_Thing: Foundational message-passing cognitive framework
- Simple_POMDP: Educational Partially Observable Markov Decision Process agent
- Generic_POMDP: Advanced hierarchical POMDP framework with meta-cognition
- Continuous_Generic: Continuous state space agents with differential equations
- Ant_Colony: Swarm intelligence with stigmergic coordination
- BioFirm: Biological firm theory with ecological cognition
- KG_Multi_Agent: Knowledge graph-based multi-agent coordination
- Path_Network: Network optimization and distributed path finding
- Baseball_Game: Game theory and strategic multi-agent systems
- ActiveInferenceInstitute: Educational and research-oriented implementations

Key Features:
- Test-driven development with comprehensive test suites
- Modular architecture supporting easy extension and customization
- Professional logging and error handling throughout
- Complete documentation with technical specifications
- Type hints and comprehensive docstrings for all public methods
- Real data analysis without mock methods
- Informative logging for debugging and monitoring

Author: Cognitive Modeling Framework Team
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import importlib
import sys

# Package metadata
__version__ = '0.1.0'
__author__ = 'Cognitive Modeling Framework Team'
__license__ = 'MIT'
__description__ = 'Active Inference agent implementations and cognitive frameworks'

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Package configuration
PACKAGE_CONFIG = {
    'implementations': [
        'Generic_Thing',
        'Simple_POMDP',
        'Generic_POMDP',
        'Continuous_Generic',
        'Ant_Colony',
        'BioFirm',
        'KG_Multi_Agent',
        'Path_Network',
        'Baseball_Game',
        'ActiveInferenceInstitute'
    ],
    'core_modules': [
        'active_inference',
        'cognitive_frameworks',
        'multi_agent_systems'
    ],
    'testing_enabled': True,
    'logging_level': 'INFO'
}

class ThingsPackageManager:
    """
    Manages the Things package initialization and provides utilities for working
    with multiple agent implementations.

    This class handles:
    - Dynamic import management for agent implementations
    - Package configuration and validation
    - Logging setup and management
    - Version and dependency checking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the package manager with configuration.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = {**PACKAGE_CONFIG}
        if config:
            self.config.update(config)

        self._implementations = {}
        self._initialized = False

        logger.info(f"Initializing Things package v{__version__}")
        self._setup_package()

    def _setup_package(self) -> None:
        """
        Set up the package environment and validate configuration.
        """
        try:
            # Validate package structure
            self._validate_package_structure()

            # Set up logging based on configuration
            self._configure_logging()

            # Initialize implementation registry
            self._initialize_implementations()

            self._initialized = True
            logger.info("Things package initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Things package: {e}")
            raise

    def _validate_package_structure(self) -> None:
        """
        Validate that all expected implementations are present.
        """
        package_dir = Path(__file__).parent

        missing_implementations = []
        for impl in self.config['implementations']:
            impl_path = package_dir / impl
            if not impl_path.exists():
                missing_implementations.append(impl)

        if missing_implementations:
            logger.warning(f"Missing implementations: {missing_implementations}")
        else:
            logger.debug("All expected implementations found")

    def _configure_logging(self) -> None:
        """
        Configure logging based on package configuration.
        """
        log_level = getattr(logging, self.config.get('logging_level', 'INFO').upper())
        logger.setLevel(log_level)

        # Update all handlers
        for handler in logger.handlers:
            handler.setLevel(log_level)

        logger.debug(f"Logging configured to level: {log_level}")

    def _initialize_implementations(self) -> None:
        """
        Initialize the registry of available implementations.
        """
        for impl_name in self.config['implementations']:
            try:
                # Attempt to import the implementation module
                module_path = f"{__name__}.{impl_name}"
                module = importlib.import_module(module_path)
                self._implementations[impl_name] = module

                logger.debug(f"Registered implementation: {impl_name}")

            except ImportError as e:
                logger.warning(f"Could not import {impl_name}: {e}")
            except Exception as e:
                logger.error(f"Error initializing {impl_name}: {e}")

    def get_implementation(self, name: str) -> Any:
        """
        Get a specific implementation module by name.

        Args:
            name: Name of the implementation to retrieve

        Returns:
            The implementation module

        Raises:
            KeyError: If implementation is not found
        """
        if name not in self._implementations:
            available = list(self._implementations.keys())
            raise KeyError(f"Implementation '{name}' not found. Available: {available}")

        return self._implementations[name]

    def list_implementations(self) -> List[str]:
        """
        Get a list of all available implementations.

        Returns:
            List of implementation names
        """
        return list(self._implementations.keys())

    def get_implementation_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific implementation.

        Args:
            name: Name of the implementation

        Returns:
            Dictionary containing implementation information
        """
        if name not in self._implementations:
            raise KeyError(f"Implementation '{name}' not found")

        module = self._implementations[name]
        info = {
            'name': name,
            'module': module,
            'version': getattr(module, '__version__', 'unknown'),
            'description': getattr(module, '__doc__', '').strip().split('\n')[0] if module.__doc__ else '',
        }

        return info

    def validate_implementations(self) -> Dict[str, bool]:
        """
        Validate that all implementations are properly configured and functional.

        Returns:
            Dictionary mapping implementation names to validation status
        """
        validation_results = {}

        for name, module in self._implementations.items():
            try:
                # Basic validation - check for required attributes
                required_attrs = ['__version__', '__doc__']
                has_required = all(hasattr(module, attr) for attr in required_attrs)

                # Check for test modules if testing is enabled
                if self.config.get('testing_enabled', False):
                    test_module_name = f"{__name__}.{name}.tests"
                    try:
                        importlib.import_module(test_module_name)
                        has_tests = True
                    except ImportError:
                        has_tests = False
                else:
                    has_tests = True  # Skip test check if testing disabled

                validation_results[name] = has_required and has_tests

            except Exception as e:
                logger.error(f"Validation failed for {name}: {e}")
                validation_results[name] = False

        return validation_results

# Global package manager instance
_package_manager = ThingsPackageManager()

def get_implementation(name: str) -> Any:
    """
    Convenience function to get an implementation module.

    Args:
        name: Name of the implementation

    Returns:
        The implementation module
    """
    return _package_manager.get_implementation(name)

def list_implementations() -> List[str]:
    """
    Convenience function to list all available implementations.

    Returns:
        List of implementation names
    """
    return _package_manager.list_implementations()

def validate_package() -> Dict[str, bool]:
    """
    Convenience function to validate all implementations.

    Returns:
        Dictionary of validation results
    """
    return _package_manager.validate_implementations()

# Export key functions and classes
__all__ = [
    # Package metadata
    '__version__',
    '__author__',
    '__license__',
    '__description__',

    # Main functionality
    'get_implementation',
    'list_implementations',
    'validate_package',
    'ThingsPackageManager',

    # Configuration
    'PACKAGE_CONFIG',
]

# Log successful package initialization
logger.info(f"Things package v{__version__} loaded successfully with {len(_package_manager.list_implementations())} implementations") 