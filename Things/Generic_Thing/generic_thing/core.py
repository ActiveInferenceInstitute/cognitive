"""
Core implementation of the Generic Thing following Active Inference principles.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# Import from package
from generic_thing.markov_blanket import MarkovBlanket
from generic_thing.free_energy import FreeEnergy
from generic_thing.message_passing import MessagePassing

logger = logging.getLogger(__name__)

@dataclass
class GenericThing:
    """
    A universal building block that implements Active Inference principles.
    
    Attributes:
        id: Unique identifier for this thing
        markov_blanket: Encapsulates the thing's boundaries and interfaces
        free_energy: Handles free energy computation and minimization
        message_passing: Manages message passing between things
        internal_state: Current internal state representation
        external_state: Perceived external state
        children: Nested sub-things forming a holarchic structure
    """
    
    id: str
    name: str = field(default="")
    description: Optional[str] = None
    
    # Core Active Inference components
    markov_blanket: MarkovBlanket = field(default_factory=MarkovBlanket)
    free_energy: FreeEnergy = field(default_factory=FreeEnergy)
    message_passing: MessagePassing = field(default_factory=lambda: MessagePassing(id=""))
    
    # State representations
    internal_state: Dict[str, Any] = field(default_factory=dict)
    external_state: Dict[str, Any] = field(default_factory=dict)
    
    # Holarchic structure
    children: List['GenericThing'] = field(default_factory=list)
    parent: Optional['GenericThing'] = None
    _state_history: List[Dict[str, Any]] = field(default_factory=list)
    _last_update: Optional[np.datetime64] = field(default=None)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self.message_passing.id = self.id
        if not self._state_history:
            self._state_history = []
        if not self._last_update:
            self._last_update = np.datetime64('now')
        logger.info(f"Created {self.name} instance")
        
        # Initialize message passing with correct ID
        if self.message_passing is None:
            self.message_passing = MessagePassing(id=self.id)

    def update(self, observation: Dict[str, Any]) -> None:
        """
        Update the thing's state based on new observations.
        
        Args:
            observation: New sensory data to process
            
        Raises:
            TypeError: If observation is None or not a dictionary
            ValueError: If observation contains invalid values
        """
        if observation is None:
            raise TypeError("Observation cannot be None")
        if not isinstance(observation, dict):
            raise TypeError("Observation must be a dictionary")
        
        # Update Markov blanket with new observation
        self.markov_blanket.update(observation)
        
        # Initialize internal state if needed
        for key in observation:
            if key not in self.internal_state:
                self.internal_state[key] = observation[key]
        
        # Minimize free energy
        self.free_energy.minimize(
            internal_state=self.internal_state,
            external_state=self.external_state,
            observation=observation
        )
        
        # Only propagate messages if connected to other things
        if self.message_passing.has_connections():
            self.message_passing.propagate(
                source=self,
                message=observation
            )
        
    def act(self) -> Dict[str, Any]:
        """
        Generate actions based on current state and predictions.
        
        Returns:
            Dict containing the selected actions
        """
        return self.free_energy.select_action(
            internal_state=self.internal_state,
            external_state=self.external_state
        )
    
    def add_child(self, child: 'GenericThing') -> None:
        """Add a child thing to create nested structure."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            logger.info(f"Added {child.name} as child of {self.name}")
    
    def remove_child(self, child: 'GenericThing') -> None:
        """Remove a child thing."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            logger.info(f"Removed {child.name} from {self.name}")
            
    def get_state(self) -> Dict[str, Any]:
        """Get the complete state of this thing."""
        return {
            'id': self.id,
            'name': self.name,
            'internal_state': self.internal_state,
            'external_state': self.external_state,
            'markov_blanket': self.markov_blanket.get_state(),
            'free_energy': self.free_energy.get_state(),
            'children': [child.get_state() for child in self.children]
        }

    def __repr__(self) -> str:
        """String representation of this thing."""
        return f"GenericThing(id='{self.id}', name='{self.name}')"

    def update_state(self, states: Dict[str, Any]) -> None:
        """Update internal states and Markov blanket.
        
        Args:
            states: Dictionary of state arrays to update
        """
        # Update Markov blanket states
        self.markov_blanket.update(states)
        
        # Update internal state tracking
        self._last_update = np.datetime64('now')
        
        # Convert numpy arrays to lists for storage
        serializable_states = {}
        for key, value in states.items():
            if isinstance(value, np.ndarray):
                serializable_states[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_states[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in value.items()}
            else:
                serializable_states[key] = value
                
        self._state_history.append({
            'timestamp': self._last_update,
            'states': serializable_states
        })
        
        # Update internal and external states
        if 'internal_states' in states:
            self.internal_state = states['internal_states']
        if 'external_states' in states:
            self.external_state = states['external_states']
            
        # Minimize free energy if we have all required states
        if all(key in states for key in ['internal_states', 'external_states', 'sensory_states']):
            self.free_energy.minimize(
                internal_state=states['internal_states'],
                external_state=states['external_states'],
                observation=states['sensory_states']
            ) 