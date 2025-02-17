"""
Intervention strategies for Earth Systems Active Inference.
Provides implementations for different intervention approaches and their selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from .earth_systems import (
    SystemState, EcologicalState, ClimateState, HumanImpactState,
    EarthSystemMetrics
)

@dataclass
class InterventionConstraints:
    """Constraints for intervention selection."""
    resource_limits: Dict[str, float]
    time_constraints: Dict[str, float]
    spatial_bounds: Dict[str, List[float]]
    social_constraints: Dict[str, Any]
    ecological_thresholds: Dict[str, float]

@dataclass
class InterventionOutcome:
    """Predicted outcome of an intervention."""
    state_change: SystemState
    confidence: float
    resource_usage: Dict[str, float]
    timeline: Dict[str, float]
    side_effects: Dict[str, float]

class InterventionStrategy(ABC):
    """Abstract base class for intervention strategies."""
    
    @abstractmethod
    def select_actions(self,
                      current_state: SystemState,
                      predictions: Dict[str, np.ndarray],
                      constraints: InterventionConstraints) -> Dict[str, Any]:
        """Select appropriate interventions."""
        pass
    
    @abstractmethod
    def evaluate_outcome(self,
                        before: SystemState,
                        after: SystemState,
                        intervention: Dict[str, Any]) -> InterventionOutcome:
        """Evaluate intervention outcome."""
        pass

class EcologicalRestoration(InterventionStrategy):
    """Ecological restoration intervention strategy."""
    
    def __init__(self,
                 restoration_params: Dict[str, Any],
                 risk_tolerance: float = 0.3):
        self.params = restoration_params
        self.risk_tolerance = risk_tolerance
        
    def select_actions(self,
                      current_state: SystemState,
                      predictions: Dict[str, np.ndarray],
                      constraints: InterventionConstraints) -> Dict[str, Any]:
        """Select restoration actions."""
        actions = {}
        
        # Assess ecological needs
        biodiversity_needs = self._assess_biodiversity_needs(current_state)
        soil_needs = self._assess_soil_needs(current_state)
        water_needs = self._assess_water_needs(current_state)
        
        # Prioritize interventions
        priorities = self._prioritize_needs(
            biodiversity_needs,
            soil_needs,
            water_needs,
            constraints
        )
        
        # Select specific actions
        for priority in priorities:
            if priority == "biodiversity":
                actions.update(self._plan_biodiversity_actions(
                    current_state, constraints
                ))
            elif priority == "soil":
                actions.update(self._plan_soil_actions(
                    current_state, constraints
                ))
            elif priority == "water":
                actions.update(self._plan_water_actions(
                    current_state, constraints
                ))
        
        return actions
    
    def evaluate_outcome(self,
                        before: SystemState,
                        after: SystemState,
                        intervention: Dict[str, Any]) -> InterventionOutcome:
        """Evaluate restoration outcome."""
        # Calculate state changes
        biodiversity_change = self._calculate_biodiversity_change(before, after)
        soil_change = self._calculate_soil_change(before, after)
        water_change = self._calculate_water_change(before, after)
        
        # Assess confidence
        confidence = self._assess_confidence(
            before, after, intervention
        )
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(intervention)
        
        # Estimate timeline
        timeline = self._estimate_timeline(intervention)
        
        # Assess side effects
        side_effects = self._assess_side_effects(before, after)
        
        return InterventionOutcome(
            state_change=after,
            confidence=confidence,
            resource_usage=resource_usage,
            timeline=timeline,
            side_effects=side_effects
        )
    
    def _assess_biodiversity_needs(self, state: SystemState) -> Dict[str, float]:
        """Assess biodiversity restoration needs."""
        return {
            "species_richness": 1.0 - np.mean(list(state.ecological.biodiversity.values())),
            "habitat_quality": 1.0 - state.ecological.resilience.get("habitat", 0.0),
            "connectivity": 1.0 - state.ecological.resilience.get("connectivity", 0.0)
        }
    
    def _assess_soil_needs(self, state: SystemState) -> Dict[str, float]:
        """Assess soil restoration needs."""
        return {
            "organic_matter": 1.0 - state.ecological.soil_health.get("organic_matter", 0.0),
            "microbial_activity": 1.0 - state.ecological.soil_health.get("microbial", 0.0),
            "structure": 1.0 - state.ecological.soil_health.get("structure", 0.0)
        }
    
    def _assess_water_needs(self, state: SystemState) -> Dict[str, float]:
        """Assess water system needs."""
        return {
            "quality": 1.0 - state.ecological.water_cycles.get("quality", 0.0),
            "flow": 1.0 - state.ecological.water_cycles.get("flow", 0.0),
            "retention": 1.0 - state.ecological.water_cycles.get("retention", 0.0)
        }

class ClimateIntervention(InterventionStrategy):
    """Climate intervention strategy."""
    
    def __init__(self,
                 climate_params: Dict[str, Any],
                 uncertainty_threshold: float = 0.2):
        self.params = climate_params
        self.uncertainty_threshold = uncertainty_threshold
    
    def select_actions(self,
                      current_state: SystemState,
                      predictions: Dict[str, np.ndarray],
                      constraints: InterventionConstraints) -> Dict[str, Any]:
        """Select climate interventions."""
        actions = {}
        
        # Assess climate needs
        temperature_needs = self._assess_temperature_needs(current_state)
        carbon_needs = self._assess_carbon_needs(current_state)
        water_needs = self._assess_water_needs(current_state)
        
        # Check prediction uncertainties
        uncertainties = self._calculate_uncertainties(predictions)
        
        # Select interventions based on needs and uncertainties
        if uncertainties["temperature"] < self.uncertainty_threshold:
            actions.update(self._plan_temperature_actions(
                temperature_needs, constraints
            ))
        
        if uncertainties["carbon"] < self.uncertainty_threshold:
            actions.update(self._plan_carbon_actions(
                carbon_needs, constraints
            ))
        
        if uncertainties["water"] < self.uncertainty_threshold:
            actions.update(self._plan_water_actions(
                water_needs, constraints
            ))
        
        return actions
    
    def evaluate_outcome(self,
                        before: SystemState,
                        after: SystemState,
                        intervention: Dict[str, Any]) -> InterventionOutcome:
        """Evaluate climate intervention outcome."""
        # Calculate state changes
        temperature_change = self._calculate_temperature_change(before, after)
        carbon_change = self._calculate_carbon_change(before, after)
        water_change = self._calculate_water_change(before, after)
        
        # Assess confidence
        confidence = self._assess_confidence(
            before, after, intervention
        )
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(intervention)
        
        # Estimate timeline
        timeline = self._estimate_timeline(intervention)
        
        # Assess side effects
        side_effects = self._assess_side_effects(before, after)
        
        return InterventionOutcome(
            state_change=after,
            confidence=confidence,
            resource_usage=resource_usage,
            timeline=timeline,
            side_effects=side_effects
        )

class SocialIntervention(InterventionStrategy):
    """Social intervention strategy."""
    
    def __init__(self,
                 social_params: Dict[str, Any],
                 community_threshold: float = 0.4):
        self.params = social_params
        self.community_threshold = community_threshold
    
    def select_actions(self,
                      current_state: SystemState,
                      predictions: Dict[str, np.ndarray],
                      constraints: InterventionConstraints) -> Dict[str, Any]:
        """Select social interventions."""
        actions = {}
        
        # Assess social needs
        community_needs = self._assess_community_needs(current_state)
        education_needs = self._assess_education_needs(current_state)
        policy_needs = self._assess_policy_needs(current_state)
        
        # Check community readiness
        readiness = self._assess_community_readiness(current_state)
        
        # Select interventions based on needs and readiness
        if readiness > self.community_threshold:
            if community_needs["engagement"] > 0.5:
                actions.update(self._plan_community_actions(
                    community_needs, constraints
                ))
            
            if education_needs["awareness"] > 0.5:
                actions.update(self._plan_education_actions(
                    education_needs, constraints
                ))
            
            if policy_needs["reform"] > 0.5:
                actions.update(self._plan_policy_actions(
                    policy_needs, constraints
                ))
        
        return actions
    
    def evaluate_outcome(self,
                        before: SystemState,
                        after: SystemState,
                        intervention: Dict[str, Any]) -> InterventionOutcome:
        """Evaluate social intervention outcome."""
        # Calculate state changes
        community_change = self._calculate_community_change(before, after)
        education_change = self._calculate_education_change(before, after)
        policy_change = self._calculate_policy_change(before, after)
        
        # Assess confidence
        confidence = self._assess_confidence(
            before, after, intervention
        )
        
        # Calculate resource usage
        resource_usage = self._calculate_resource_usage(intervention)
        
        # Estimate timeline
        timeline = self._estimate_timeline(intervention)
        
        # Assess side effects
        side_effects = self._assess_side_effects(before, after)
        
        return InterventionOutcome(
            state_change=after,
            confidence=confidence,
            resource_usage=resource_usage,
            timeline=timeline,
            side_effects=side_effects
        )

class InterventionCoordinator:
    """Coordinates multiple intervention strategies."""
    
    def __init__(self,
                 strategies: Dict[str, InterventionStrategy],
                 weights: Optional[Dict[str, float]] = None):
        self.strategies = strategies
        self.weights = weights or {k: 1.0 for k in strategies.keys()}
    
    def select_interventions(self,
                           current_state: SystemState,
                           predictions: Dict[str, np.ndarray],
                           constraints: InterventionConstraints) -> Dict[str, Any]:
        """Select coordinated interventions."""
        all_actions = {}
        
        # Get actions from each strategy
        for name, strategy in self.strategies.items():
            actions = strategy.select_actions(
                current_state, predictions, constraints
            )
            all_actions[name] = actions
        
        # Resolve conflicts and synergies
        coordinated_actions = self._resolve_conflicts(all_actions)
        
        # Optimize resource allocation
        optimized_actions = self._optimize_resources(
            coordinated_actions, constraints
        )
        
        return optimized_actions
    
    def evaluate_outcomes(self,
                         before: SystemState,
                         after: SystemState,
                         interventions: Dict[str, Any]) -> Dict[str, InterventionOutcome]:
        """Evaluate outcomes of all interventions."""
        outcomes = {}
        
        for name, strategy in self.strategies.items():
            if name in interventions:
                outcome = strategy.evaluate_outcome(
                    before, after, interventions[name]
                )
                outcomes[name] = outcome
        
        return outcomes
    
    def _resolve_conflicts(self,
                         actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between different interventions."""
        resolved = {}
        
        # Identify conflicts
        conflicts = self._identify_conflicts(actions)
        
        # Resolve each conflict
        for conflict in conflicts:
            resolution = self._resolve_conflict(
                conflict, actions
            )
            resolved.update(resolution)
        
        return resolved
    
    def _optimize_resources(self,
                          actions: Dict[str, Any],
                          constraints: InterventionConstraints) -> Dict[str, Any]:
        """Optimize resource allocation across interventions."""
        # Calculate resource requirements
        requirements = self._calculate_requirements(actions)
        
        # Check against constraints
        if self._check_constraints(requirements, constraints):
            return actions
        
        # Optimize if constraints are exceeded
        return self._optimize_allocation(
            actions, requirements, constraints
        ) 