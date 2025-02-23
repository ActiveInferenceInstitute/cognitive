"""Tests for Message Passing functionality."""

import unittest
import logging
import numpy as np
from typing import Dict, Any
from generic_thing.message_passing import MessagePassing, Message
from generic_thing.tests.test_utils import (
    log_test_case,
    plot_message_network,
    generate_test_data,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class MockThing:
    """Mock Thing class for testing message passing."""
    def __init__(self, thing_id: str):
        self.id = thing_id

class TestMessagePassing(unittest.TestCase):
    """Test suite for MessagePassing class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock things
        self.thing1 = MockThing("thing1")
        self.thing2 = MockThing("thing2")
        self.thing3 = MockThing("thing3")
        
        # Create message passing instances for each thing
        self.mp1 = MessagePassing(id=self.thing1.id)
        self.mp2 = MessagePassing(id=self.thing2.id)
        self.mp3 = MessagePassing(id=self.thing3.id)
        
        logger.info("Created MessagePassing test instances")
    
    @log_test_case
    def test_initialization(self):
        """Test proper initialization of MessagePassing."""
        self.assertEqual(len(self.mp1.connections), 0)
        self.assertEqual(len(self.mp1.incoming_queue), 0)
        self.assertEqual(len(self.mp1.outgoing_queue), 0)
        self.assertEqual(len(self.mp1.message_history), 0)
    
    @log_test_case
    def test_connection_management(self):
        """Test managing connections between things."""
        # Connect things
        self.mp1.connect(self.thing1.id, self.thing2.id)
        self.mp2.connect(self.thing2.id, self.thing3.id)
        
        # Verify connections
        self.assertIn(self.thing2.id, self.mp1.connections[self.thing1.id])
        self.assertIn(self.thing1.id, self.mp1.connections[self.thing2.id])
        self.assertIn(self.thing3.id, self.mp2.connections[self.thing2.id])
        
        # Visualize network
        plot_message_network(
            self.mp1,
            timestamp_filename("message_network")
        )
        
        # Disconnect things
        self.mp1.disconnect(self.thing1.id, self.thing2.id)
        
        # Verify disconnection
        self.assertNotIn(self.thing2.id, self.mp1.connections[self.thing1.id])
        self.assertNotIn(self.thing1.id, self.mp1.connections[self.thing2.id])
    
    @log_test_case
    def test_message_propagation(self):
        """Test message propagation between things."""
        # Connect things
        self.mp1.connect(self.thing1.id, self.thing2.id)
        
        # Create test message
        message = Message(
            source_id=self.thing1.id,
            target_id=self.thing2.id,
            content=generate_test_data(),
            message_type="test"
        )
        
        # Send message
        self.mp1.send_message(message)
        
        # Verify message in outgoing queue
        self.assertEqual(len(self.mp1.outgoing_queue), 1)
        self.assertEqual(len(self.mp1.message_history), 1)
        
        # Process outgoing messages
        self.mp1.process_outgoing()
        
        # Verify message moved to incoming queue
        self.assertEqual(len(self.mp1.outgoing_queue), 0)
        self.assertEqual(len(self.mp1.incoming_queue), 1)
        
        # Receive messages for thing2
        received = self.mp1.receive(self.thing2.id)
        
        # Verify received messages
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].target_id, self.thing2.id)
        self.assertEqual(received[0].source_id, self.thing1.id)
        
        logger.info(f"Propagated message: {message}")
    
    @log_test_case
    def test_message_receiving(self):
        """Test receiving messages by target things."""
        # Connect things
        self.mp1.connect(self.thing1.id, self.thing2.id)
        
        # Create and propagate messages
        for i in range(3):
            message = {f"data_{i}": i}
            self.mp1.send_message(Message(
                source_id=self.thing1.id,
                target_id=self.thing2.id,
                content=message,
                message_type="test"
            ))
        
        # Process outgoing messages
        self.mp1.process_outgoing()
        
        # Receive messages for thing2
        received = self.mp1.receive(self.thing2.id)
        
        # Verify received messages
        self.assertEqual(len(received), 3)
        self.assertTrue(all(msg.target_id == self.thing2.id for msg in received))
        self.assertTrue(all(msg.source_id == self.thing1.id for msg in received))
        
        logger.info(f"Received {len(received)} messages")
    
    @log_test_case
    def test_message_filtering(self):
        """Test filtering messages by target."""
        # Connect multiple things
        self.mp1.connect(self.thing1.id, self.thing2.id)
        self.mp1.connect(self.thing1.id, self.thing3.id)
        
        # Send messages to different targets
        message1 = {"data": 1}
        message2 = {"data": 2}
        
        self.mp1.send_message(Message(
            source_id=self.thing1.id,
            target_id=self.thing2.id,
            content=message1,
            message_type="test"
        ))  # Goes to both
        self.mp2.send_message(Message(
            source_id=self.thing2.id,
            target_id=self.thing1.id,
            content=message2,
            message_type="test"
        ))  # Goes to thing1
        
        # Process messages
        self.mp1.process_outgoing()
        
        # Receive messages for thing2
        thing2_messages = self.mp1.receive(self.thing2.id)
        
        # Verify filtering
        self.assertEqual(len(thing2_messages), 1)
        self.assertEqual(thing2_messages[0].source_id, self.thing1.id)
    
    @log_test_case
    def test_message_history(self):
        """Test message history tracking."""
        # Connect things
        self.mp1.connect(self.thing1.id, self.thing2.id)
        
        # Send sequence of messages
        n_messages = 5
        for i in range(n_messages):
            message = {f"data_{i}": i}
            self.mp1.send_message(Message(
                source_id=self.thing1.id,
                target_id=self.thing2.id,
                content=message,
                message_type="test"
            ))
        
        # Verify history
        self.assertEqual(len(self.mp1.message_history), n_messages)
        
        # Verify history contents
        for i, msg in enumerate(self.mp1.message_history):
            self.assertEqual(msg.source_id, self.thing1.id)
            self.assertEqual(msg.target_id, self.thing2.id)
            self.assertEqual(msg.content[f"data_{i}"], i)
    
    @log_test_case
    def test_error_handling(self):
        """Test error handling in message passing."""
        # Test sending message with invalid source ID
        with self.assertRaises(ValueError):
            self.mp1.send_message(Message(
                source_id="invalid_id",  # Using invalid source ID
                target_id=self.thing2.id,
                content={"data": 1},
                message_type="test"
            ))
        
        # Test sending message with invalid content type
        with self.assertRaises(TypeError):
            # Create a message with non-serializable content (a function)
            def test_func(): pass
            self.mp1.send_message(Message(
                source_id=self.thing1.id,
                target_id=self.thing2.id,
                content=test_func,  # Non-serializable content
                message_type="test",
                timestamp=np.datetime64('now')
            ))
        
        # Test receiving message with invalid target ID
        with self.assertRaises(ValueError):
            self.mp1.receive_message(Message(
                source_id=self.thing2.id,
                target_id="invalid_id",  # Using invalid target ID
                content={"data": 1},
                message_type="test",
                timestamp=np.datetime64('now')
            ))

if __name__ == '__main__':
    unittest.main() 