"""
Simulation framework for Earth Systems Active Inference.
Provides tools for running and analyzing multi-scale earth system simulations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

from .earth_systems import (
    SystemState, EcologicalState, ClimateState, HumanImpactState,
    NestedController, EarthSystemMetrics
)
from .interventions import (
    InterventionStrategy, InterventionConstraints, InterventionOutcome,
    InterventionCoordinator
)
from .visualization import MultiScaleViz, StateSpaceViz, InterventionViz

@dataclass
class SimulationConfig:
    """Configuration for earth system simulation."""
    duration: float
    time_step: float
    spatial_resolution: Dict[str, float]
    noise_levels: Dict[str, float]
    intervention_frequency: float
    random_seed: Optional[int] = None

@dataclass
class SimulationState:
    """Complete simulation state."""
    time: float
    system_state: SystemState
    controller_state: Dict[str, Any]
    intervention_history: List[Dict[str, Any]]
    metrics_history: List[Dict[str, float]]

class EarthSystemSimulator:
    """Simulator for earth system dynamics."""
    
    def __init__(self,
                 config: SimulationConfig,
                 controller: NestedController,
                 intervention_coordinator: InterventionCoordinator):
        self.config = config
        self.controller = controller
        self.coordinator = intervention_coordinator
        self.rng = np.random.default_rng(config.random_seed)
        self.state = self._initialize_state()
        
    def run_simulation(self) -> SimulationState:
        """Run complete simulation."""
        current_time = 0.0
        
        # Setup progress bar
        n_steps = int(self.config.duration / self.config.time_step)
        pbar = tqdm(total=n_steps, desc="Running simulation")
        
        while current_time < self.config.duration:
            # Update system state
            self._update_system_state(current_time)
            
            # Check for interventions
            if self._should_intervene(current_time):
                self._apply_interventions()
            
            # Record metrics
            self._record_metrics()
            
            # Update time
            current_time += self.config.time_step
            pbar.update(1)
        
        pbar.close()
        return self.state
    
    def _initialize_state(self) -> SimulationState:
        """Initialize simulation state."""
        return SimulationState(
            time=0.0,
            system_state=self._create_initial_state(),
            controller_state={},
            intervention_history=[],
            metrics_history=[]
        )
    
    def _create_initial_state(self) -> SystemState:
        """Create initial system state."""
        return SystemState(
            ecological=self._initialize_ecological_state(),
            climate=self._initialize_climate_state(),
            human=self._initialize_human_state(),
            timestamp=0.0,
            scale="global"
        )
    
    def _initialize_ecological_state(self) -> EcologicalState:
        """Initialize ecological state."""
        return EcologicalState(
            biodiversity={"richness": 0.7, "evenness": 0.6},
            biomass={"total": 0.8, "distribution": 0.7},
            soil_health={"organic_matter": 0.6, "microbial": 0.7},
            water_cycles={"quality": 0.8, "flow": 0.7},
            energy_flows={"primary": 0.8, "secondary": 0.7},
            resilience={"recovery": 0.6, "resistance": 0.7}
        )
    
    def _initialize_climate_state(self) -> ClimateState:
        """Initialize climate state."""
        return ClimateState(
            temperature={"mean": 0.5, "variance": 0.1},
            precipitation={"total": 0.6, "pattern": 0.7},
            wind_patterns={"speed": 0.5, "direction": 0.6},
            carbon_cycles={"storage": 0.7, "flux": 0.6},
            energy_balance={"radiation": 0.7, "albedo": 0.6}
        )
    
    def _initialize_human_state(self) -> HumanImpactState:
        """Initialize human impact state."""
        return HumanImpactState(
            land_use={"natural": 0.6, "modified": 0.4},
            resource_extraction={"rate": 0.5, "efficiency": 0.6},
            pollution_levels={"air": 0.3, "water": 0.4},
            restoration_efforts={"coverage": 0.5, "effectiveness": 0.6},
            social_indicators={"awareness": 0.5, "action": 0.4}
        )
    
    def _update_system_state(self, current_time: float):
        """Update system state for one time step."""
        # Get controller actions
        actions = self.controller.update(
            {scale: self.state.system_state for scale in ["micro", "meso", "macro"]}
        )
        
        # Apply actions to state
        self._apply_actions(actions)
        
        # Apply natural dynamics
        self._apply_natural_dynamics()
        
        # Add noise
        self._add_noise()
        
        # Update timestamp
        self.state.system_state.timestamp = current_time
    
    def _should_intervene(self, current_time: float) -> bool:
        """Determine if intervention is needed."""
        if current_time % self.config.intervention_frequency == 0:
            metrics = EarthSystemMetrics.compute_metrics(
                self.state.system_state
            )
            return any(v < 0.5 for v in metrics.values())
        return False
    
    def _apply_interventions(self):
        """Apply interventions to system state."""
        # Get current predictions
        predictions = self._generate_predictions()
        
        # Create constraints
        constraints = self._create_intervention_constraints()
        
        # Select interventions
        interventions = self.coordinator.select_interventions(
            self.state.system_state,
            predictions,
            constraints
        )
        
        # Apply interventions
        self._apply_intervention_effects(interventions)
        
        # Record intervention
        self.state.intervention_history.append(interventions)
    
    def _generate_predictions(self) -> Dict[str, np.ndarray]:
        """Generate system state predictions."""
        # Simple linear extrapolation for now
        return {
            "ecological": self._extrapolate_state(
                self.state.system_state.ecological
            ),
            "climate": self._extrapolate_state(
                self.state.system_state.climate
            ),
            "human": self._extrapolate_state(
                self.state.system_state.human
            )
        }
    
    def _create_intervention_constraints(self) -> InterventionConstraints:
        """Create intervention constraints."""
        return InterventionConstraints(
            resource_limits={"budget": 1000.0, "manpower": 100.0},
            time_constraints={"immediate": 7.0, "short_term": 30.0},
            spatial_bounds=[[-1000.0, 1000.0], [-1000.0, 1000.0]],
            social_constraints={"acceptance": 0.6},
            ecological_thresholds={"biodiversity": 0.3}
        )
    
    def _apply_intervention_effects(self, interventions: Dict[str, Any]):
        """Apply intervention effects to system state."""
        for intervention_type, actions in interventions.items():
            if intervention_type == "ecological":
                self._apply_ecological_intervention(actions)
            elif intervention_type == "climate":
                self._apply_climate_intervention(actions)
            elif intervention_type == "social":
                self._apply_social_intervention(actions)
    
    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """Apply controller actions to system state."""
        # Apply temporal scale actions
        for scale, action in actions.items():
            if scale.startswith("temporal"):
                self._apply_temporal_action(scale, action)
            elif scale.startswith("spatial"):
                self._apply_spatial_action(scale, action)
    
    def _apply_natural_dynamics(self):
        """Apply natural system dynamics."""
        # Simple dynamics for now
        self._update_ecological_dynamics()
        self._update_climate_dynamics()
        self._update_human_dynamics()
    
    def _add_noise(self):
        """Add noise to system state."""
        self._add_ecological_noise()
        self._add_climate_noise()
        self._add_human_noise()
    
    def _record_metrics(self):
        """Record current system metrics."""
        metrics = EarthSystemMetrics.compute_metrics(
            self.state.system_state
        )
        self.state.metrics_history.append(metrics)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results."""
        return {
            "metrics_summary": self._analyze_metrics(),
            "intervention_analysis": self._analyze_interventions(),
            "stability_analysis": self._analyze_stability(),
            "resilience_analysis": self._analyze_resilience()
        }
    
    def visualize_results(self, save_path: Optional[Path] = None):
        """Visualize simulation results."""
        # Create visualizations
        temporal_fig = MultiScaleViz.plot_temporal_hierarchy(
            self._get_state_history(),
            self._get_metrics_history(),
            (0.0, self.config.duration)
        )
        
        state_space_fig = StateSpaceViz.plot_state_space(
            self.state.system_state,
            self._get_state_history()["global"],
            self._generate_predictions()
        )
        
        intervention_fig = InterventionViz.plot_intervention_impacts(
            self._get_initial_state(),
            self.state.system_state,
            self.state.intervention_history[-1]
            if self.state.intervention_history else {}
        )
        
        # Save if path provided
        if save_path:
            temporal_fig.savefig(save_path / "temporal_hierarchy.png")
            state_space_fig.write_html(str(save_path / "state_space.html"))
            intervention_fig.write_html(str(save_path / "interventions.html"))
        
        return {
            "temporal": temporal_fig,
            "state_space": state_space_fig,
            "interventions": intervention_fig
        } 