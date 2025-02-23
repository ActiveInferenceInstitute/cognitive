"""
Implementation of Federated Inference for Generic Things.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from .message_passing import Message

@dataclass
class FederatedInference:
    """
    Implements federated inference across Generic Things.
    
    Handles:
    - Distributed belief updates
    - Consensus formation
    - Evidence accumulation
    - Parameter sharing
    """
    
    # Local belief state
    beliefs: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Confidence/precision for each belief
    precisions: Dict[str, float] = field(default_factory=dict)
    
    # Accumulated evidence from other things
    evidence: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Learning rate for belief updates
    learning_rate: float = 0.1
    
    def update_beliefs(self, messages: List[Message]) -> None:
        """
        Update beliefs based on received messages.
        
        Args:
            messages: List of received messages containing evidence
        """
        for msg in messages:
            content = msg.content
            source = msg.source_id
            
            # Store evidence
            if source not in self.evidence:
                self.evidence[source] = []
            self.evidence[source].append(content)
            
            # Update beliefs for each variable
            for var, value in content.items():
                if var not in self.beliefs:
                    self.beliefs[var] = np.array(value)
                    self.precisions[var] = 1.0
                else:
                    # Weighted update based on precision and learning rate
                    precision = self.precisions[var]
                    current_belief = self.beliefs[var]
                    new_value = np.array(value)
                    
                    # Update belief with learning rate effect
                    # Higher learning rate means faster convergence to new value
                    # The weight of the new value increases with learning rate
                    self.beliefs[var] = (
                        self.learning_rate * new_value +
                        (1 - self.learning_rate) * current_belief
                    )
                    
                    # Update precision - higher learning rate means more confidence
                    self.precisions[var] *= (1 + self.learning_rate)
    
    def form_consensus(self) -> Dict[str, np.ndarray]:
        """
        Form consensus beliefs across all evidence.
        
        Returns:
            Dictionary of consensus beliefs
        """
        consensus = {}
        
        # For each variable we have beliefs about
        for var in self.beliefs:
            # Collect all evidence for this variable
            all_evidence = []
            for source_evidence in self.evidence.values():
                for evidence_item in source_evidence:
                    if var in evidence_item:
                        all_evidence.append(np.array(evidence_item[var]))
            
            if all_evidence:
                # Form consensus through weighted average
                weights = np.ones(len(all_evidence)) / len(all_evidence)
                consensus[var] = np.average(all_evidence, weights=weights, axis=0)
            else:
                consensus[var] = self.beliefs[var]
                
        return consensus
    
    def get_belief_state(self) -> Dict[str, Any]:
        """
        Get current belief state including consensus.
        
        Returns:
            Dictionary containing current beliefs and consensus
        """
        return {
            'beliefs': self.beliefs,
            'precisions': self.precisions,
            'consensus': self.form_consensus(),
            'num_evidence_sources': len(self.evidence)
        }
    
    def reset_evidence(self) -> None:
        """Reset accumulated evidence while maintaining current beliefs."""
        self.evidence.clear()
    
    def compute_uncertainty(self) -> Dict[str, float]:
        """
        Compute uncertainty for each belief.
        
        Returns:
            Dictionary of uncertainty values for each variable
        """
        uncertainties = {}
        for var in self.beliefs:
            # Simple uncertainty based on inverse precision
            uncertainties[var] = 1.0 / (self.precisions[var] + 1e-6)
        return uncertainties
    
    def share_parameters(self, 
                        source_id: str,
                        parameters: Dict[str, np.ndarray]) -> Message:
        """
        Create message for parameter sharing.
        
        Args:
            source_id: ID of the source thing
            parameters: Parameters to share
            
        Returns:
            Message containing shared parameters
        """
        return Message(
            source_id=source_id,
            target_id="broadcast",  # Special target for parameter sharing
            content=parameters,
            message_type="parameters",
            timestamp=np.datetime64('now')
        ) 