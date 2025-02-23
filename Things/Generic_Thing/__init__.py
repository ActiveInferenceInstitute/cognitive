"""
Generic Thing - A universal building block for Active Inference based systems.
"""

__version__ = '0.1.0'

from generic_thing.core import GenericThing
from generic_thing.markov_blanket import MarkovBlanket
from generic_thing.free_energy import FreeEnergy
from generic_thing.message_passing import MessagePassing
from generic_thing.inference import FederatedInference

__all__ = [
    'GenericThing',
    'MarkovBlanket',
    'FreeEnergy',
    'MessagePassing',
    'FederatedInference'
]

"""Generic Thing package.""" 