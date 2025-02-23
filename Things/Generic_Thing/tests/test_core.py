"""Tests for Generic Thing core functionality."""

import unittest
import logging
from typing import Dict, Any
import numpy as np
from generic_thing.core import GenericThing
from generic_thing.tests.test_utils import (
    log_test_case,
    plot_thing_hierarchy,
    generate_test_data,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class TestGenericThing(unittest.TestCase):
    """Test suite for GenericThing class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.thing = GenericThing(
            id="test_thing",
            name="Test Thing",
            description="A thing for testing"
        )
        logger.info("Created test thing instance")
    
    @log_test_case
    def test_initialization(self):
        """Test proper initialization of GenericThing."""
        self.assertEqual(self.thing.id, "test_thing")
        self.assertEqual(self.thing.name, "Test Thing")
        self.assertEqual(self.thing.description, "A thing for testing")
        self.assertIsNotNone(self.thing.markov_blanket)
        self.assertIsNotNone(self.thing.free_energy)
        self.assertIsNotNone(self.thing.message_passing)
    
    @log_test_case
    def test_hierarchical_structure(self):
        """Test hierarchical relationships between things."""
        # Create a hierarchy of things
        parent = GenericThing(id="parent", name="Parent")
        child1 = GenericThing(id="child1", name="Child 1")
        child2 = GenericThing(id="child2", name="Child 2")
        grandchild = GenericThing(id="grandchild", name="Grandchild")
        
        # Build hierarchy
        parent.add_child(child1)
        parent.add_child(child2)
        child1.add_child(grandchild)
        
        # Test relationships
        self.assertIn(child1, parent.children)
        self.assertIn(child2, parent.children)
        self.assertIn(grandchild, child1.children)
        self.assertEqual(child1.parent, parent)
        self.assertEqual(grandchild.parent, child1)
        
        # Visualize hierarchy
        plot_thing_hierarchy(parent, timestamp_filename("hierarchy_test"))
    
    @log_test_case
    def test_state_update(self):
        """Test state updates through observations."""
        # Generate test data
        observation = generate_test_data()
        
        # Update state
        self.thing.update(observation)
        
        # Verify state changes
        state = self.thing.get_state()
        self.assertIsNotNone(state['markov_blanket'])
        self.assertIsNotNone(state['free_energy'])
        
        logger.info(f"Updated state with observation: {observation}")
    
    @log_test_case
    def test_action_generation(self):
        """Test action generation based on state."""
        # Set up internal and external states
        self.thing.internal_state = {'x': 0.5, 'y': -0.3}
        self.thing.external_state = {'x': 0.8, 'y': -0.1}
        
        # Generate actions
        actions = self.thing.act()
        
        # Verify actions
        self.assertIsInstance(actions, dict)
        self.assertIn('action_x', actions)
        self.assertIn('action_y', actions)
        
        logger.info(f"Generated actions: {actions}")
    
    @log_test_case
    def test_child_management(self):
        """Test adding and removing child things."""
        child = GenericThing(id="child", name="Child")
        
        # Add child
        self.thing.add_child(child)
        self.assertIn(child, self.thing.children)
        self.assertEqual(child.parent, self.thing)
        
        # Remove child
        self.thing.remove_child(child)
        self.assertNotIn(child, self.thing.children)
        self.assertIsNone(child.parent)
    
    @log_test_case
    def test_state_persistence(self):
        """Test persistence of state across updates."""
        # Initial state
        self.thing.internal_state = {'a': 1.0}
        self.thing.external_state = {'b': 2.0}
        
        # Update with new observation
        self.thing.update({'c': 3.0})
        
        # Verify state persistence
        self.assertEqual(self.thing.internal_state['a'], 1.0)
        self.assertEqual(self.thing.external_state['b'], 2.0)
    
    @log_test_case
    def test_error_handling(self):
        """Test error handling in state updates."""
        # Test with invalid observation
        with self.assertRaises(TypeError):
            self.thing.update(None)
        
        # Test with invalid state access
        with self.assertRaises(KeyError):
            _ = self.thing.internal_state['nonexistent']

if __name__ == '__main__':
    unittest.main() 