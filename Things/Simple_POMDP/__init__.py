"""
Simple_POMDP package for basic POMDP implementation with active inference.
"""

from .simple_pomdp import SimplePOMDP, compute_expected_free_energy

__all__ = ['SimplePOMDP', 'compute_expected_free_energy'] 