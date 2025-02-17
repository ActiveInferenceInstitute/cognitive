"""
Stewardship model implementation for BioFirm framework.
Handles bioregional evaluation and intervention strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from abc import ABC, abstractmethod

from .state_spaces import BioregionalState
from .observation import ObservationModel, HierarchicalObservation
from .transition import TransitionModel, HierarchicalTransition

@dataclass
class Intervention:
    """Represents a bioregional intervention."""
    name: str
    type: str  # PROTECT, RESTORE, ENHANCE, TRANSFORM
    target_variables: List[str]
    scale: str
    duration: float
    intensity: float
    resources: Dict[str, float]
    stakeholders: List[str]
    expected_outcomes: Dict[str, float]
    uncertainty: float
    constraints: Dict[str, Any]

@dataclass
class StewardshipMetrics:
    """Comprehensive bioregional performance tracking."""
    ecological_metrics: Dict[str, float] = field(default_factory=lambda: {
        "biodiversity_index": 0.0,
        "ecosystem_health": 0.0,
        "habitat_connectivity": 0.0,
        "species_persistence": 0.0,
        "ecological_resilience": 0.0
    })
    
    climate_metrics: Dict[str, float] = field(default_factory=lambda: {
        "carbon_sequestration": 0.0,
        "water_regulation": 0.0,
        "microclimate_stability": 0.0,
        "extreme_event_buffer": 0.0
    })
    
    social_metrics: Dict[str, float] = field(default_factory=lambda: {
        "community_participation": 0.0,
        "knowledge_integration": 0.0,
        "cultural_preservation": 0.0,
        "governance_effectiveness": 0.0
    })
    
    economic_metrics: Dict[str, float] = field(default_factory=lambda: {
        "sustainable_value": 0.0,
        "resource_efficiency": 0.0,
        "green_jobs": 0.0,
        "ecosystem_services_value": 0.0
    })
    
    stewardship_metrics: Dict[str, float] = field(default_factory=lambda: {
        "management_effectiveness": 0.0,
        "stakeholder_engagement": 0.0,
        "adaptive_capacity": 0.0,
        "cross_scale_coordination": 0.0
    })

class StewardshipMode(ABC):
    """Abstract base class for stewardship modes."""
    
    @abstractmethod
    def evaluate_state(self,
                      current_state: BioregionalState,
                      target_state: BioregionalState) -> float:
        """Evaluate current state against stewardship goals."""
        pass
    
    @abstractmethod
    def propose_interventions(self,
                            state: BioregionalState,
                            constraints: Dict[str, Any]) -> List[Intervention]:
        """Propose context-appropriate interventions."""
        pass

class AdaptiveComanagement(StewardshipMode):
    """Implements adaptive comanagement stewardship approach."""
    
    def __init__(self,
                 stakeholder_weights: Dict[str, float],
                 learning_rate: float = 0.1):
        self.stakeholder_weights = stakeholder_weights
        self.learning_rate = learning_rate
        self.intervention_history: List[Tuple[Intervention, float]] = []
        
    def evaluate_state(self,
                      current_state: BioregionalState,
                      target_state: BioregionalState) -> float:
        """Evaluate state using weighted stakeholder preferences."""
        # Convert states to vectors
        current_vec = current_state.to_vector()
        target_vec = target_state.to_vector()
        
        # Calculate weighted evaluation
        evaluation = 0.0
        for stakeholder, weight in self.stakeholder_weights.items():
            stakeholder_eval = self._stakeholder_evaluation(
                current_vec, target_vec, stakeholder
            )
            evaluation += weight * stakeholder_eval
            
        return evaluation
        
    def _stakeholder_evaluation(self,
                              current: np.ndarray,
                              target: np.ndarray,
                              stakeholder: str) -> float:
        """Compute stakeholder-specific evaluation."""
        # Could be customized per stakeholder type
        return -np.sum((current - target) ** 2)
        
    def propose_interventions(self,
                            state: BioregionalState,
                            constraints: Dict[str, Any]) -> List[Intervention]:
        """Propose interventions based on collective knowledge."""
        proposed = []
        
        # Consider past successful interventions
        successful_patterns = self._analyze_history()
        
        # Generate candidate interventions
        candidates = self._generate_candidates(state, successful_patterns)
        
        # Filter by constraints
        for candidate in candidates:
            if self._satisfies_constraints(candidate, constraints):
                proposed.append(candidate)
                
        return proposed
        
    def _analyze_history(self) -> Dict[str, float]:
        """Analyze intervention history for patterns."""
        if not self.intervention_history:
            return {}
            
        patterns = {}
        for intervention, outcome in self.intervention_history:
            key = f"{intervention.type}_{intervention.scale}"
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(outcome)
            
        return {k: np.mean(v) for k, v in patterns.items()}
        
    def _generate_candidates(self,
                           state: BioregionalState,
                           patterns: Dict[str, float]) -> List[Intervention]:
        """Generate candidate interventions."""
        candidates = []
        
        # Generate based on state assessment
        if state.ecological_state["biodiversity"] < 0.5:
            candidates.append(
                Intervention(
                    name="Biodiversity Enhancement",
                    type="ENHANCE",
                    target_variables=["biodiversity", "habitat_connectivity"],
                    scale="local",
                    duration=2.0,
                    intensity=0.7,
                    resources={"budget": 100000, "staff_hours": 2000},
                    stakeholders=["local_communities", "scientists"],
                    expected_outcomes={"biodiversity": 0.2},
                    uncertainty=0.3,
                    constraints={"social_acceptance": 0.8}
                )
            )
            
        # Add more candidates based on patterns
        for pattern, success_rate in patterns.items():
            if success_rate > 0.7:  # High success threshold
                intervention_type, scale = pattern.split("_")
                candidates.append(
                    self._create_intervention_from_pattern(
                        intervention_type, scale, state
                    )
                )
                
        return candidates
        
    def _satisfies_constraints(self,
                             intervention: Intervention,
                             constraints: Dict[str, Any]) -> bool:
        """Check if intervention satisfies constraints."""
        if "budget_limit" in constraints:
            if intervention.resources["budget"] > constraints["budget_limit"]:
                return False
                
        if "time_horizon" in constraints:
            if intervention.duration > float(constraints["time_horizon"][:-1]):
                return False
                
        if "social_acceptance" in constraints:
            if intervention.constraints["social_acceptance"] < constraints["social_acceptance"]:
                return False
                
        return True
        
    def _create_intervention_from_pattern(self,
                                        intervention_type: str,
                                        scale: str,
                                        state: BioregionalState) -> Intervention:
        """Create intervention based on successful pattern."""
        # Template method - could be customized
        return Intervention(
            name=f"{intervention_type} at {scale}",
            type=intervention_type,
            target_variables=self._select_targets(state),
            scale=scale,
            duration=1.0,
            intensity=0.5,
            resources={"budget": 50000, "staff_hours": 1000},
            stakeholders=list(self.stakeholder_weights.keys()),
            expected_outcomes=self._estimate_outcomes(state),
            uncertainty=0.4,
            constraints={"social_acceptance": 0.7}
        )
        
    def _select_targets(self, state: BioregionalState) -> List[str]:
        """Select intervention targets based on state."""
        targets = []
        
        # Add variables below threshold
        for domain in ["ecological", "climate", "social", "economic"]:
            state_dict = getattr(state, f"{domain}_state")
            for var, value in state_dict.items():
                if value < 0.4:  # Low performance threshold
                    targets.append(f"{domain}.{var}")
                    
        return targets[:3]  # Limit to top 3 targets
        
    def _estimate_outcomes(self, state: BioregionalState) -> Dict[str, float]:
        """Estimate intervention outcomes."""
        outcomes = {}
        
        # Simple improvement estimates
        for domain in ["ecological", "climate", "social", "economic"]:
            state_dict = getattr(state, f"{domain}_state")
            for var, value in state_dict.items():
                if value < 0.7:  # Room for improvement
                    outcomes[f"{domain}.{var}"] = min(0.2, 1.0 - value)
                    
        return outcomes

class BioregionalStewardship:
    """Main stewardship coordination class."""
    
    def __init__(self,
                 observation_model: HierarchicalObservation,
                 transition_model: HierarchicalTransition,
                 stewardship_mode: StewardshipMode):
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.stewardship_mode = stewardship_mode
        self.metrics = StewardshipMetrics()
        
    def evaluate_system(self,
                       states: Dict[str, BioregionalState],
                       target_states: Dict[str, BioregionalState]
                       ) -> Dict[str, float]:
        """Evaluate system state across scales."""
        evaluations = {}
        
        for scale, state in states.items():
            evaluations[scale] = self.stewardship_mode.evaluate_state(
                state, target_states[scale]
            )
            
        return evaluations
        
    def plan_interventions(self,
                          states: Dict[str, BioregionalState],
                          constraints: Dict[str, Any]
                          ) -> Dict[str, List[Intervention]]:
        """Plan interventions across scales."""
        interventions = {}
        
        for scale, state in states.items():
            scale_constraints = self._adjust_constraints(constraints, scale)
            interventions[scale] = self.stewardship_mode.propose_interventions(
                state, scale_constraints
            )
            
        return interventions
        
    def _adjust_constraints(self,
                          constraints: Dict[str, Any],
                          scale: str) -> Dict[str, Any]:
        """Adjust constraints based on scale."""
        scale_factors = {
            "local": 0.2,
            "landscape": 0.3,
            "regional": 0.5,
            "bioregional": 1.0
        }
        
        adjusted = constraints.copy()
        if "budget_limit" in adjusted:
            adjusted["budget_limit"] *= scale_factors[scale]
            
        return adjusted
        
    def update_metrics(self,
                      states: Dict[str, BioregionalState],
                      interventions: Dict[str, List[Intervention]]):
        """Update stewardship metrics."""
        # Update ecological metrics
        self.metrics.ecological_metrics.update(
            self._compute_ecological_metrics(states)
        )
        
        # Update climate metrics
        self.metrics.climate_metrics.update(
            self._compute_climate_metrics(states)
        )
        
        # Update social metrics
        self.metrics.social_metrics.update(
            self._compute_social_metrics(states, interventions)
        )
        
        # Update economic metrics
        self.metrics.economic_metrics.update(
            self._compute_economic_metrics(states, interventions)
        )
        
        # Update stewardship metrics
        self.metrics.stewardship_metrics.update(
            self._compute_stewardship_metrics(states, interventions)
        )
        
    def _compute_ecological_metrics(self,
                                  states: Dict[str, BioregionalState]
                                  ) -> Dict[str, float]:
        """Compute ecological metrics across scales."""
        metrics = {}
        
        # Aggregate across scales
        biodiversity = np.mean([
            state.ecological_state["biodiversity"]
            for state in states.values()
        ])
        metrics["biodiversity_index"] = biodiversity
        
        # Add other ecological metrics
        return metrics
        
    def _compute_climate_metrics(self,
                               states: Dict[str, BioregionalState]
                               ) -> Dict[str, float]:
        """Compute climate metrics across scales."""
        metrics = {}
        
        # Aggregate across scales
        carbon = np.mean([
            state.climate_state["carbon_storage"]
            for state in states.values()
        ])
        metrics["carbon_sequestration"] = carbon
        
        # Add other climate metrics
        return metrics
        
    def _compute_social_metrics(self,
                              states: Dict[str, BioregionalState],
                              interventions: Dict[str, List[Intervention]]
                              ) -> Dict[str, float]:
        """Compute social metrics."""
        metrics = {}
        
        # Calculate participation
        participation = np.mean([
            state.social_state["community_engagement"]
            for state in states.values()
        ])
        metrics["community_participation"] = participation
        
        # Add other social metrics
        return metrics
        
    def _compute_economic_metrics(self,
                                states: Dict[str, BioregionalState],
                                interventions: Dict[str, List[Intervention]]
                                ) -> Dict[str, float]:
        """Compute economic metrics."""
        metrics = {}
        
        # Calculate sustainable value
        value = np.mean([
            state.economic_state["sustainable_livelihoods"]
            for state in states.values()
        ])
        metrics["sustainable_value"] = value
        
        # Add other economic metrics
        return metrics
        
    def _compute_stewardship_metrics(self,
                                   states: Dict[str, BioregionalState],
                                   interventions: Dict[str, List[Intervention]]
                                   ) -> Dict[str, float]:
        """Compute stewardship effectiveness metrics."""
        metrics = {}
        
        # Calculate management effectiveness
        n_interventions = sum(len(i) for i in interventions.values())
        if n_interventions > 0:
            effectiveness = np.mean([
                intervention.expected_outcomes.get("success_rate", 0.5)
                for scale_interventions in interventions.values()
                for intervention in scale_interventions
            ])
            metrics["management_effectiveness"] = effectiveness
            
        # Add other stewardship metrics
        return metrics 