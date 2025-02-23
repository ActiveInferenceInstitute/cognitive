"""Tests for interactions between Generic Things."""

import unittest
import logging
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from generic_thing.visualization import Visualizer
from generic_thing.core import GenericThing
from generic_thing.markov_blanket import MarkovBlanket
from generic_thing.message_passing import MessagePassing, Message
from generic_thing.tests.test_utils import (
    log_test_case,
    generate_test_data,
    plot_message_network,
    plot_thing_hierarchy,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class TestThingInteractions(unittest.TestCase):
    """Test suite for interactions between Generic Things."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz = Visualizer()
        
        # Create a network of interacting things
        self.parent = GenericThing(id="parent", name="Parent")
        self.child1 = GenericThing(id="child1", name="Child 1")
        self.child2 = GenericThing(id="child2", name="Child 2")
        self.grandchild1 = GenericThing(id="grandchild1", name="Grandchild 1")
        self.grandchild2 = GenericThing(id="grandchild2", name="Grandchild 2")
        
        # Set up hierarchy
        self.parent.add_child(self.child1)
        self.parent.add_child(self.child2)
        self.child1.add_child(self.grandchild1)
        self.child2.add_child(self.grandchild2)
        
        # Initialize message passing connections
        self.setup_message_connections()
        
        logger.info("Created test network of interacting Things")
    
    def setup_message_connections(self):
        """Set up message passing connections between Things."""
        # Parent-child connections
        self.parent.message_passing.connect(self.parent.id, self.child1.id)
        self.parent.message_passing.connect(self.parent.id, self.child2.id)
        self.child1.message_passing.connect(self.child1.id, self.parent.id)
        self.child2.message_passing.connect(self.child2.id, self.parent.id)
        
        # Sibling connections
        self.child1.message_passing.connect(self.child1.id, self.child2.id)
        self.child2.message_passing.connect(self.child2.id, self.child1.id)
        
        # Grandchild connections
        self.child1.message_passing.connect(self.child1.id, self.grandchild1.id)
        self.child2.message_passing.connect(self.child2.id, self.grandchild2.id)
        self.grandchild1.message_passing.connect(self.grandchild1.id, self.child1.id)
        self.grandchild2.message_passing.connect(self.grandchild2.id, self.child2.id)
    
    @log_test_case
    def test_hierarchical_message_propagation(self):
        """Test message propagation through the hierarchy."""
        # Generate test data
        data = generate_test_data()
        
        # Parent sends message down the hierarchy
        message = Message(
            source_id=self.parent.id,
            target_id="broadcast",
            content=data,
            message_type="update",
            timestamp=np.datetime64('now')
        )
        
        # Send message from parent and process outgoing messages
        self.parent.message_passing.send_message(message)
        self.parent.message_passing.process_outgoing()
        
        # Process messages at each child level
        for child in [self.child1, self.child2]:
            child.message_passing.process_outgoing()
            
            # Verify message reception
            received = child.message_passing.receive(child.id)
            self.assertTrue(len(received) > 0, f"No messages received by {child.id}")
            
            # Verify message source and target
            self.assertTrue(
                any(msg.source_id == self.parent.id for msg in received),
                f"No messages from parent received by {child.id}"
            )
            self.assertTrue(
                all(msg.target_id in [child.id, "broadcast"] for msg in received),
                f"Message with invalid target received by {child.id}"
            )
            
            # Forward broadcast messages to grandchildren
            for msg in received:
                if msg.target_id == "broadcast":
                    forward_msg = Message(
                        source_id=child.id,
                        target_id="broadcast",
                        content=msg.content,
                        message_type=msg.message_type,
                        timestamp=np.datetime64('now')
                    )
                    child.message_passing.send_message(forward_msg)
            child.message_passing.process_outgoing()
        
        # Process and verify grandchild messages
        for grandchild, parent in [(self.grandchild1, self.child1), (self.grandchild2, self.child2)]:
            grandchild.message_passing.process_outgoing()
            received = grandchild.message_passing.receive(grandchild.id)
            
            self.assertTrue(len(received) > 0, f"No messages received by {grandchild.id}")
            self.assertTrue(
                any(msg.source_id == parent.id for msg in received),
                f"No messages from parent received by {grandchild.id}"
            )
            self.assertTrue(
                all(msg.target_id in [grandchild.id, "broadcast"] for msg in received),
                f"Message with invalid target received by {grandchild.id}"
            )
    
    @log_test_case
    def test_markov_blanket_interactions(self):
        """Test interactions between Markov blankets."""
        # Set up initial states with proper dimensions
        initial_states = {
            'internal_states': {'state1': 0.5, 'state2': -0.3},
            'external_states': {'state1': 0.2, 'state2': 0.4},
            'sensory_states': {'state1': 0.1, 'state2': -0.2},
            'active_states': {'state1': -0.1, 'state2': 0.3},
            'free_energies': [0.5]
        }
        
        # Update states for parent and child1
        self.parent.update_state(initial_states)
        self.child1.update_state(initial_states)
        
        # Create test message with Markov blanket data
        message = Message(
            source_id=self.parent.id,
            target_id=self.child1.id,
            content={
                'internal_states': initial_states['internal_states'],
                'active_states': initial_states['active_states']
            },
            message_type="mb_update",
            timestamp=np.datetime64('now')
        )
        
        # Send message and process
        self.parent.message_passing.send_message(message)
        self.parent.message_passing.process_outgoing()
        
        # Verify message reception
        received = self.child1.message_passing.receive(self.child1.id)
        self.assertTrue(len(received) > 0, "No messages received by child1")
        
        # Update child's states based on received message
        if received:
            msg = received[0]
            update_states = {
                'internal_states': msg.content['internal_states'],
                'active_states': msg.content['active_states']
            }
            self.child1.update_state(update_states)
            
            # Verify state updates
            self.assertEqual(
                self.child1.markov_blanket.internal_states,
                msg.content['internal_states']
            )
            self.assertEqual(
                self.child1.markov_blanket.active_states,
                msg.content['active_states']
            )
    
    def plot_interacting_markov_blankets(self):
        """Create visualization of interacting Markov blankets."""
        plt.figure(figsize=(15, 10))
        
        # Create positions for each Thing's Markov blanket
        positions = {
            'parent': (0.5, 0.8),
            'child1': (0.3, 0.5),
            'child2': (0.7, 0.5),
            'grandchild1': (0.2, 0.2),
            'grandchild2': (0.8, 0.2)
        }
        
        # Plot each Markov blanket components as circles
        for thing, pos in zip(
            [self.parent, self.child1, self.child2, self.grandchild1, self.grandchild2],
            positions.values()
        ):
            circle_radius = 0.1
            sensory_circle = plt.Circle(
                (pos[0] - circle_radius, pos[1]),
                circle_radius,
                color='blue',
                alpha=0.3,
                label='Sensory'
            )
            internal_circle = plt.Circle(
                (pos[0], pos[1]),
                circle_radius,
                color='green',
                alpha=0.3,
                label='Internal'
            )
            active_circle = plt.Circle(
                (pos[0] + circle_radius, pos[1]),
                circle_radius,
                color='red',
                alpha=0.3,
                label='Active'
            )
            
            plt.gca().add_patch(sensory_circle)
            plt.gca().add_patch(internal_circle)
            plt.gca().add_patch(active_circle)
            
            # Add labels
            plt.text(pos[0], pos[1] + 0.15, thing.name, 
                    horizontalalignment='center')
        
        # Draw connections between Things
        for (thing1, pos1), (thing2, pos2) in zip(
            list(positions.items())[:-1],
            list(positions.items())[1:]
        ):
            plt.arrow(pos1[0], pos1[1], 
                     pos2[0] - pos1[0], pos2[1] - pos1[1],
                     head_width=0.02, head_length=0.02,
                     fc='gray', ec='gray', alpha=0.5)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title('Interacting Markov Blankets in Thing Hierarchy')
        
        # Save the visualization
        plt.savefig(self.viz._ensure_viz_path('interacting_markov_blankets.png'))
        plt.close()
    
    @log_test_case
    def test_collective_free_energy_minimization(self):
        """Test collective free energy minimization across Things."""
        # Initialize states with proper dimensions
        initial_states = {
            'internal_states': {'state1': 0.5, 'state2': -0.3},
            'external_states': {'state1': 0.2, 'state2': 0.4},
            'sensory_states': {'state1': 0.1, 'state2': -0.2},
            'active_states': {'state1': -0.1, 'state2': 0.3},
            'free_energies': [0.5]
        }
        
        # Set initial states for all things
        for thing in [self.parent, self.child1, self.child2]:
            thing.update_state(initial_states)
        
        # Record initial collective free energy
        initial_fe = sum(
            thing.markov_blanket.free_energies[0]
            for thing in [self.parent, self.child1, self.child2]
        )
        
        # Perform free energy minimization steps
        for _ in range(5):
            # Exchange messages between things
            for source in [self.parent, self.child1, self.child2]:
                # Create message with current states
                message = Message(
                    source_id=source.id,
                    target_id="broadcast",
                    content={
                        'internal_states': source.markov_blanket.internal_states,
                        'active_states': source.markov_blanket.active_states,
                        'free_energies': source.markov_blanket.free_energies
                    },
                    message_type="fe_update",
                    timestamp=np.datetime64('now')
                )
                source.message_passing.send_message(message)
                source.message_passing.process_outgoing()
            
            # Process received messages and update states
            for thing in [self.parent, self.child1, self.child2]:
                received = thing.message_passing.receive(thing.id)
                if received:
                    # Combine received states
                    combined_states = {
                        'internal_states': {
                            k: np.mean([msg.content['internal_states'][k] for msg in received])
                            for k in received[0].content['internal_states']
                        },
                        'active_states': {
                            k: np.mean([msg.content['active_states'][k] for msg in received])
                            for k in received[0].content['active_states']
                        },
                        'free_energies': [
                            np.mean([msg.content['free_energies'][0] for msg in received])
                        ]
                    }
                    thing.update_state(combined_states)
        
        # Calculate final collective free energy
        final_fe = sum(
            thing.markov_blanket.free_energies[0]
            for thing in [self.parent, self.child1, self.child2]
        )
        
        # Verify free energy has decreased
        self.assertLess(
            final_fe,
            initial_fe,
            f"Free energy did not decrease: initial={initial_fe}, final={final_fe}"
        )
    
    @log_test_case
    def test_nested_belief_propagation(self):
        """Test belief propagation through nested Thing structure."""
        # Initialize beliefs at each level
        beliefs = {
            'parent': {'x': 0.5, 'y': -0.3},
            'child1': {'x': 0.2, 'y': 0.4},
            'child2': {'x': 0.1, 'y': -0.2},
            'grandchild1': {'x': -0.1, 'y': 0.3},
            'grandchild2': {'x': 0.4, 'y': -0.4}
        }
        
        # Create messages for belief updates
        for thing_id, belief in beliefs.items():
            thing = getattr(self, thing_id)
            message = Message(
                source_id=thing.id,
                target_id="broadcast",
                content={'beliefs': belief},
                message_type="belief_update",
                timestamp=np.datetime64('now')
            )
            thing.message_passing.send_message(message)
            thing.message_passing.process_outgoing()
        
        # Verify belief propagation
        for thing_id in ['child1', 'child2', 'grandchild1', 'grandchild2']:
            thing = getattr(self, thing_id)
            received = thing.message_passing.receive(thing.id)
            self.assertTrue(len(received) > 0, f"No messages received by {thing_id}")
            
            # Verify message content
            for msg in received:
                self.assertIn('beliefs', msg.content)
                self.assertIn('x', msg.content['beliefs'])
                self.assertIn('y', msg.content['beliefs'])
    
    def plot_nested_beliefs(self, beliefs: Dict[str, Dict[str, np.ndarray]]):
        """Create visualization of nested belief structure."""
        plt.figure(figsize=(15, 10))
        
        # Create positions for each Thing's beliefs
        positions = {
            'parent': (0.5, 0.8),
            'child1': (0.3, 0.5),
            'child2': (0.7, 0.5),
            'grandchild1': (0.2, 0.2),
            'grandchild2': (0.8, 0.2)
        }
        
        # Plot beliefs as heatmaps
        for name, pos in positions.items():
            belief_data = beliefs[name]['x']
            
            # Create small heatmap for each Thing's beliefs
            ax = plt.axes([pos[0]-0.1, pos[1]-0.1, 0.2, 0.2])
            sns.heatmap(
                belief_data.reshape(-1, 1),
                cmap='viridis',
                cbar=False,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(name)
        
        # Draw arrows indicating belief flow
        for (name1, pos1), (name2, pos2) in zip(
            list(positions.items())[:-1],
            list(positions.items())[1:]
        ):
            plt.arrow(pos1[0], pos1[1], 
                     pos2[0] - pos1[0], pos2[1] - pos1[1],
                     head_width=0.02, head_length=0.02,
                     fc='gray', ec='gray', alpha=0.5)
        
        plt.title('Nested Belief Structure')
        
        # Save the visualization
        plt.savefig(self.viz._ensure_viz_path('nested_beliefs.png'))
        plt.close()
    
    def tearDown(self):
        """Clean up test resources."""
        plt.close('all')

if __name__ == '__main__':
    unittest.main() 