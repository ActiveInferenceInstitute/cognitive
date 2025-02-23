class MarkovBlanket:
    """Represents the Markov blanket of a Generic Thing."""
    
    def __init__(self):
        """Initialize Markov blanket with empty states."""
        self.sensory_states = {}
        self.internal_states = {}
        self.active_states = {}
        self.free_energies = []
        self.sensory_weights = None
        self.active_weights = None
        self.state_history = []
        self.learning_rate = 0.01
        
    def set_weights(self, sensory_weights: np.ndarray, active_weights: np.ndarray):
        """Set connection weights for state updates."""
        # Validate weight dimensions
        if len(sensory_weights.shape) != 2 or len(active_weights.shape) != 2:
            raise ValueError("Weights must be 2D arrays")
        if sensory_weights.shape[1] != active_weights.shape[1]:
            raise ValueError("Weight matrices must have compatible dimensions")
        self.sensory_weights = sensory_weights
        self.active_weights = active_weights
    
    def update(self, observation: Dict[str, Any]):
        """Update states based on new observation."""
        if not isinstance(observation, dict):
            raise TypeError("Observation must be a dictionary")
        
        # Initialize states if empty
        if not self.sensory_states:
            self.sensory_states = {}
        if not self.internal_states:
            self.internal_states = {}
        if not self.active_states:
            self.active_states = {}
            
        # Update sensory states
        if 'sensory_states' in observation:
            self.sensory_states = observation['sensory_states'].copy()
        elif all(isinstance(v, (int, float, np.ndarray)) for v in observation.values()):
            self.sensory_states = observation.copy()
        else:
            raise ValueError("Invalid observation format")
            
        # Update internal states if provided
        if 'internal_states' in observation:
            self.internal_states = observation['internal_states'].copy()
        else:
            self._update_internal_states()
            
        # Update active states if provided
        if 'active_states' in observation:
            self.active_states = observation['active_states'].copy()
        else:
            self._update_active_states()
            
        # Update free energies if provided
        if 'free_energies' in observation:
            self.free_energies = observation['free_energies'].copy() if isinstance(observation['free_energies'], list) else [observation['free_energies']]
        
        # Minimize free energy
        self._minimize_free_energy()
            
        # Record state history
        self.state_history.append({
            'sensory_states': self.sensory_states.copy(),
            'internal_states': self.internal_states.copy(),
            'active_states': self.active_states.copy(),
            'free_energies': self.free_energies.copy()
        })
    
    def _update_internal_states(self):
        """Compute internal states from sensory input."""
        if self.sensory_weights is None:
            return
            
        # Convert sensory states to vector format
        if isinstance(self.sensory_states, dict):
            sensory_vec = np.array(list(self.sensory_states.values())).reshape(-1, 1)
        else:
            sensory_vec = np.array(self.sensory_states).reshape(-1, 1)
            
        # Compute internal states using transposed weights
        internal_vec = np.dot(self.sensory_weights.T, sensory_vec)
        self.internal_states = {
            f'state_{i}': float(v) 
            for i, v in enumerate(internal_vec.flatten())
        }
    
    def _update_active_states(self):
        """Compute active states from internal states."""
        if self.active_weights is None:
            return
            
        # Convert internal states to vector format
        if isinstance(self.internal_states, dict):
            internal_vec = np.array(list(self.internal_states.values())).reshape(-1, 1)
        else:
            internal_vec = np.array(self.internal_states).reshape(-1, 1)
            
        # Compute active states
        active_vec = np.dot(self.active_weights.T, internal_vec)
        self.active_states = {
            f'state_{i}': float(v) 
            for i, v in enumerate(active_vec.flatten())
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the Markov blanket."""
        return {
            'sensory_states': self.sensory_states,
            'internal_states': self.internal_states,
            'active_states': self.active_states,
            'free_energies': self.free_energies
        }
    
    def _minimize_free_energy(self):
        """Minimize free energy by updating internal states."""
        # Convert states to vectors
        if isinstance(self.sensory_states, dict):
            sensory_vec = np.array(list(self.sensory_states.values())).reshape(-1, 1)
        else:
            sensory_vec = np.array(self.sensory_states).reshape(-1, 1)
            
        if isinstance(self.internal_states, dict):
            internal_vec = np.array(list(self.internal_states.values())).reshape(-1, 1)
        else:
            internal_vec = np.array(self.internal_states).reshape(-1, 1)
            
        # Compute prediction error using correct matrix dimensions
        prediction = np.dot(self.sensory_weights, internal_vec)
        prediction_error = sensory_vec - prediction
        
        # Compute gradient with transposed weights
        gradient = -np.dot(self.sensory_weights.T, prediction_error)
        
        # Update internal states
        internal_vec -= self.learning_rate * gradient
        
        # Update state dictionary
        self.internal_states = {
            f'state_{i}': float(v) 
            for i, v in enumerate(internal_vec.flatten())
        }
        
        # Compute current free energy
        current_fe = float(np.sum(prediction_error ** 2))
        self.free_energies.append(current_fe)

    def _compute_free_energy(self):
        """Compute free energy from current states."""
        # Convert states to vectors
        if isinstance(self.sensory_states, dict):
            sensory_vec = np.array(list(self.sensory_states.values())).reshape(-1, 1)
        else:
            sensory_vec = np.array(self.sensory_states).reshape(-1, 1)
            
        if isinstance(self.internal_states, dict):
            internal_vec = np.array(list(self.internal_states.values())).reshape(-1, 1)
        else:
            internal_vec = np.array(self.internal_states).reshape(-1, 1)
            
        # Compute prediction error using correct matrix dimensions
        prediction = np.dot(self.sensory_weights, internal_vec)
        prediction_error = sensory_vec - prediction
        
        # Compute current free energy
        current_fe = float(np.sum(prediction_error ** 2))
        self.free_energies.append(current_fe) 