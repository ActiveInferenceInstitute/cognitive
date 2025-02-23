"""Tests for Federated Inference functionality."""

import unittest
import logging
import numpy as np
from typing import Dict, Any
from generic_thing.inference import FederatedInference
from generic_thing.message_passing import Message
from generic_thing.tests.test_utils import (
    log_test_case,
    generate_test_data,
    plot_belief_state,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class TestFederatedInference(unittest.TestCase):
    """Test suite for FederatedInference class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.inference = FederatedInference()
        logger.info("Created FederatedInference test instance")
    
    @log_test_case
    def test_initialization(self):
        """Test proper initialization of FederatedInference."""
        self.assertEqual(len(self.inference.beliefs), 0)
        self.assertEqual(len(self.inference.precisions), 0)
        self.assertEqual(len(self.inference.evidence), 0)
        self.assertEqual(self.inference.learning_rate, 0.1)
    
    @log_test_case
    def test_belief_updates(self):
        """Test updating beliefs based on received messages."""
        # Create test messages
        messages = [
            Message(
                source_id=f"source_{i}",
                target_id="target",
                content={'x': i, 'y': -i},
                message_type="update",
                timestamp=np.datetime64('now')
            )
            for i in range(3)
        ]
        
        # Process and update beliefs
        for message in messages:
            self.inference.update_beliefs([message])
            
        # Verify belief updates
        self.assertIn('x', self.inference.beliefs)
        self.assertIn('y', self.inference.beliefs)
        self.assertEqual(len(self.inference.evidence), 3)
        
        # Verify precision updates
        self.assertGreater(self.inference.precisions['x'], 1.0)
        self.assertGreater(self.inference.precisions['y'], 1.0)
        
        # Verify message history
        self.assertEqual(len(self.inference.evidence), 3)
        for source_id in [f"source_{i}" for i in range(3)]:
            self.assertIn(source_id, self.inference.evidence)
        
        logger.info(f"Updated beliefs: {self.inference.beliefs}")
    
    @log_test_case
    def test_consensus_formation(self):
        """Test forming consensus from multiple evidence sources."""
        # Add evidence from multiple sources
        sources = ['source1', 'source2', 'source3']
        test_var = 'z'
        true_value = 1.0
        
        for source in sources:
            message = Message(
                source_id=source,
                target_id="target",
                content={test_var: true_value + np.random.normal(0, 0.1)},
                message_type="update",
                timestamp=np.datetime64('now')
            )
            self.inference.update_beliefs([message])
        
        # Form consensus
        consensus = self.inference.form_consensus()
        
        # Verify consensus
        self.assertIn(test_var, consensus)
        self.assertAlmostEqual(consensus[test_var], true_value, delta=0.2)
        
        logger.info(f"Formed consensus: {consensus}")
    
    @log_test_case
    def test_uncertainty_computation(self):
        """Test computation of belief uncertainties."""
        # Add some beliefs with different precisions
        variables = ['a', 'b', 'c']
        for i, var in enumerate(variables):
            self.inference.beliefs[var] = np.array([i])
            self.inference.precisions[var] = 1.0 / (i + 1)
        
        # Compute uncertainties
        uncertainties = self.inference.compute_uncertainty()
        
        # Verify uncertainties
        for i, var in enumerate(variables):
            self.assertIn(var, uncertainties)
            self.assertGreater(uncertainties[var], 0.0)
            if i > 0:
                # Uncertainty should increase with decreasing precision
                self.assertGreater(
                    uncertainties[var],
                    uncertainties[variables[i-1]]
                )
        
        # Visualize belief states and uncertainties
        plot_belief_state(
            self.inference.beliefs,
            uncertainties,
            timestamp_filename("belief_uncertainty")
        )
    
    @log_test_case
    def test_evidence_accumulation(self):
        """Test accumulation of evidence from multiple sources."""
        # Generate evidence from multiple sources
        n_sources = 5
        n_observations = 3
        
        for i in range(n_sources):
            source = f"source_{i}"
            for _ in range(n_observations):
                message = Message(
                    source_id=source,
                    target_id="target",
                    content=generate_test_data(size=2),
                    message_type="update",
                    timestamp=np.datetime64('now')
                )
                self.inference.update_beliefs([message])
        
        # Verify evidence accumulation
        self.assertEqual(len(self.inference.evidence), n_sources)
        for evidence_list in self.inference.evidence.values():
            self.assertEqual(len(evidence_list), n_observations)
    
    @log_test_case
    def test_parameter_sharing(self):
        """Test creation of parameter sharing messages."""
        # Create parameters to share
        parameters = {
            'weights': np.random.randn(3, 3),
            'bias': np.random.randn(3)
        }
        
        # Create sharing message
        message = self.inference.share_parameters(
            source_id="source",
            parameters=parameters
        )
        
        # Verify message
        self.assertEqual(message.source_id, "source")
        self.assertEqual(message.target_id, "broadcast")
        self.assertEqual(message.message_type, "parameters")
        self.assertEqual(message.content, parameters)
    
    @log_test_case
    def test_belief_reset(self):
        """Test resetting accumulated evidence."""
        # Add some evidence
        message = Message(
            source_id="source",
            target_id="target",
            content=generate_test_data(),
            message_type="update",
            timestamp=np.datetime64('now')
        )
        self.inference.update_beliefs([message])
        
        # Store initial beliefs
        initial_beliefs = dict(self.inference.beliefs)
        
        # Reset evidence
        self.inference.reset_evidence()
        
        # Verify reset
        self.assertEqual(len(self.inference.evidence), 0)
        # Beliefs should remain unchanged
        self.assertEqual(self.inference.beliefs, initial_beliefs)
    
    @log_test_case
    def test_learning_rate_effects(self):
        """Test effects of different learning rates."""
        # Test variable
        var = 'test'
        true_value = 1.0
        initial_value = 0.0
        
        # Test different learning rates
        learning_rates = [0.1, 0.5, 0.9]
        results = {}
        
        for lr in learning_rates:
            # Create new inference instance with specific learning rate
            inference = FederatedInference(
                beliefs={var: np.array(initial_value)},
                precisions={var: 1.0},
                learning_rate=lr
            )
            
            # Add same evidence
            message = Message(
                source_id="source",
                target_id="target",
                content={var: true_value},
                message_type="update",
                timestamp=np.datetime64('now')
            )
            inference.update_beliefs([message])
            
            # Store result
            results[lr] = inference.beliefs[var]
        
        # Verify learning rate effects - higher learning rates should lead to faster convergence
        for lr1, lr2 in zip(learning_rates[:-1], learning_rates[1:]):
            self.assertGreater(
                abs(results[lr2] - initial_value),  # Change from true value to initial value
                abs(results[lr1] - initial_value)
            )

if __name__ == '__main__':
    unittest.main() 