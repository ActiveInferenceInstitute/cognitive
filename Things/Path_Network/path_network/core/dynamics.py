"""
Environmental dynamics for the Path Network simulation.
Generates nested sinusoidal waves that represent the global water level.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class WaveComponent:
    """Parameters for a single sinusoidal wave component."""
    amplitude: float
    frequency: float
    phase: float = 0.0

@dataclass
class DynamicsConfig:
    """Configuration for the environmental dynamics."""
    base_components: List[WaveComponent] = None
    noise_std: float = 0.05
    time_scale: float = 1.0
    
    def __post_init__(self):
        if self.base_components is None:
            # Default wave components
            self.base_components = [
                WaveComponent(amplitude=1.0, frequency=0.1),  # Slow wave
                WaveComponent(amplitude=0.5, frequency=0.5),  # Medium wave
                WaveComponent(amplitude=0.2, frequency=2.0)   # Fast wave
            ]

class EnvironmentalDynamics:
    """
    Generates and manages the environmental dynamics (water level) through
    nested sinusoidal waves with optional noise and perturbations.
    """
    
    def __init__(self, config: DynamicsConfig):
        self.config = config
        self.time = 0.0
        self.history: List[float] = []
        self.perturbation: Optional[Tuple[float, float, float]] = None
        
    def step(self) -> float:
        """
        Compute the next water level value.
        
        Returns:
            float: The current global water level
        """
        # Base level from superposition of waves
        level = sum(
            component.amplitude * np.sin(
                2 * np.pi * component.frequency * self.time * self.config.time_scale
                + component.phase
            )
            for component in self.config.base_components
        )
        
        # Add noise
        if self.config.noise_std > 0:
            level += np.random.normal(0, self.config.noise_std)
        
        # Add any active perturbation
        if self.perturbation is not None:
            magnitude, duration, decay = self.perturbation
            if self.time < duration:
                level += magnitude * (1 - self.time / duration)
            else:
                # Remove perturbation after it expires
                self.perturbation = None
        
        # Update time and history
        self.time += 1
        self.history.append(level)
        
        return level
    
    def add_perturbation(
        self,
        magnitude: float,
        duration: float,
        decay: float = 1.0
    ) -> None:
        """
        Add a temporary perturbation to the water level.
        
        Args:
            magnitude: Size of the perturbation
            duration: How long the perturbation lasts
            decay: How quickly the perturbation decays (1.0 = linear)
        """
        self.perturbation = (magnitude, duration, decay)
    
    def add_wave_component(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0
    ) -> None:
        """
        Add a new wave component to the dynamics.
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency
            phase: Wave phase shift
        """
        self.config.base_components.append(
            WaveComponent(amplitude, frequency, phase)
        )
    
    def get_history(self) -> List[float]:
        """Get the history of water levels."""
        return self.history.copy()
    
    def get_wave_components(self) -> List[WaveComponent]:
        """Get the current wave components."""
        return self.config.base_components.copy()
    
    def reset(self) -> None:
        """Reset the dynamics to initial state."""
        self.time = 0.0
        self.history.clear()
        self.perturbation = None 