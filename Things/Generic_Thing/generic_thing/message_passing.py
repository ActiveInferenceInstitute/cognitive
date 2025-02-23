"""
Implementation of Message Passing for Generic Things.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Deque
from collections import deque
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """
    Represents a message passed between Generic Things.
    """
    source_id: str
    target_id: str
    content: Dict[str, Any]
    message_type: str
    timestamp: datetime = field(default_factory=lambda: np.datetime64('now'))

@dataclass
class MessagePassing:
    """
    Implements message passing between Generic Things.
    
    Handles:
    - Federated inference through message exchange
    - Belief propagation across Markov blankets
    - Hierarchical message passing in nested structures
    """
    
    id: str
    connections: Dict[str, List[str]] = field(default_factory=dict)
    received_messages: List[Message] = field(default_factory=list)
    sent_messages: List[Message] = field(default_factory=list)
    incoming_queue: Deque[Message] = field(default_factory=deque)
    outgoing_queue: Deque[Message] = field(default_factory=deque)
    message_history: List[Message] = field(default_factory=list)
    
    def _is_valid_target(self, target_id: str) -> bool:
        """Check if target_id is valid in the connection graph.
        
        Args:
            target_id: ID to validate
            
        Returns:
            bool: True if target is valid, False otherwise
        """
        # Special cases
        if target_id == "broadcast" or target_id == self.id:
            return True
            
        # Check direct connections from this node
        if self.id in self.connections and target_id in self.connections[self.id]:
            return True
            
        # Check if target exists in any connection list
        for node, targets in self.connections.items():
            if target_id in targets:
                return True
                
        return False
    
    def process_outgoing(self) -> None:
        """Process all messages in the outgoing queue."""
        processed = []
        while self.outgoing_queue:
            message = self.outgoing_queue.popleft()
            if self._is_valid_target(message.target_id) or message.target_id == "broadcast":
                self.receive_message(message)
                logger.info(f"Processed outgoing message from {message.source_id} to {message.target_id}")
            else:
                # Put the message back in the queue if not processed
                self.outgoing_queue.append(message)
                logger.warning(f"Invalid target ID in outgoing message: {message.target_id}")
                
            processed.append(message)
            
            # Break if we've seen all messages to avoid infinite loop
            if len(processed) >= len(self.outgoing_queue) + 1:
                break
    
    def receive(self, target_id: str) -> List[Message]:
        """Receive all messages intended for a specific target.
        
        Args:
            target_id: ID of the target Thing
            
        Returns:
            List of messages for the target
        """
        if not self._is_valid_target(target_id):
            logger.warning(f"Invalid target ID for receiving: {target_id}")
            return []
            
        messages = []
        remaining = deque()
        
        while self.incoming_queue:
            message = self.incoming_queue.popleft()
            if message.target_id == target_id or message.target_id == "broadcast":
                messages.append(message)
                self.received_messages.append(message)
                self.message_history.append(message)
            else:
                remaining.append(message)
                
        self.incoming_queue = remaining
        logger.info(f"Retrieved {len(messages)} messages for {target_id}")
        return messages
    
    def connect(self, source_id: str, target_id: str) -> None:
        """Add a bidirectional connection between two Things.
        
        Args:
            source_id: ID of the source Thing
            target_id: ID of the target Thing
        """
        # Initialize connection lists if needed
        if source_id not in self.connections:
            self.connections[source_id] = []
        if target_id not in self.connections:
            self.connections[target_id] = []
            
        # Add bidirectional connections
        if target_id not in self.connections[source_id]:
            self.connections[source_id].append(target_id)
            logger.info(f"Added connection from {source_id} to {target_id}")
            
        if source_id not in self.connections[target_id]:
            self.connections[target_id].append(source_id)
            logger.info(f"Added connection from {target_id} to {source_id}")
            
        # Ensure both message passing instances are updated
        if hasattr(self, 'id'):
            if self.id == source_id:
                if target_id not in self.connections[source_id]:
                    self.connections[source_id].append(target_id)
            elif self.id == target_id:
                if source_id not in self.connections[target_id]:
                    self.connections[target_id].append(source_id)
    
    def disconnect(self, source_id: str, target_id: str) -> None:
        """Remove a bidirectional connection between two Things.
        
        Args:
            source_id: ID of the source Thing
            target_id: ID of the target Thing
        """
        # Remove forward connection
        if source_id in self.connections and target_id in self.connections[source_id]:
            self.connections[source_id].remove(target_id)
            logger.info(f"Removed connection from {source_id} to {target_id}")
            
        # Remove reverse connection
        if target_id in self.connections and source_id in self.connections[target_id]:
            self.connections[target_id].remove(source_id)
            logger.info(f"Removed connection from {target_id} to {source_id}")
            
        # Keep empty connection lists (don't delete them)
        if source_id not in self.connections:
            self.connections[source_id] = []
        if target_id not in self.connections:
            self.connections[target_id] = []
            
        # Ensure both message passing instances are updated
        if hasattr(self, 'id'):
            if self.id == source_id:
                if target_id in self.connections[source_id]:
                    self.connections[source_id].remove(target_id)
            elif self.id == target_id:
                if source_id in self.connections[target_id]:
                    self.connections[target_id].remove(source_id)
    
    def send_message(self, message: Message) -> None:
        """Send a message to the outgoing queue.
        
        Args:
            message: Message to send
            
        Raises:
            ValueError: If source or target ID is invalid
            TypeError: If message content is not JSON serializable
        """
        # Validate source ID
        if message.source_id != self.id:
            raise ValueError(f"Invalid source ID: {message.source_id}")
            
        # Initialize connections for source if not exists
        if message.source_id not in self.connections:
            self.connections[message.source_id] = []
            
        # For broadcast messages
        if message.target_id == "broadcast":
            # Get all connected nodes except the source
            targets = set()
            for node, connections in self.connections.items():
                targets.update(connections)
            targets.discard(message.source_id)
            
            # Send to each target
            for target_id in targets:
                broadcast_msg = Message(
                    source_id=message.source_id,
                    target_id=target_id,
                    content=message.content,
                    message_type=message.message_type,
                    timestamp=message.timestamp
                )
                self.outgoing_queue.append(broadcast_msg)
                self.sent_messages.append(broadcast_msg)
                logger.info(f"Sent broadcast message from {message.source_id} to {target_id}")
        else:
            # For direct messages, validate target exists in connection graph
            if not self._is_valid_target(message.target_id):
                # Check if target exists in any connection list
                target_found = False
                for node, targets in self.connections.items():
                    if message.target_id in targets or node == message.target_id:
                        target_found = True
                        break
                        
                if not target_found:
                    raise ValueError(f"Invalid target ID: {message.target_id}")
            
            # Add message to outgoing queue and sent messages
            self.outgoing_queue.append(message)
            self.sent_messages.append(message)
            logger.info(f"Sent message from {message.source_id} to {message.target_id}")
            
        # Add to message history
        self.message_history.append(message)
        
        # Process outgoing queue immediately
        self.process_outgoing()
    
    def receive_message(self, message: Message) -> None:
        """Receive a message directly into the incoming queue.
        
        Args:
            message: Message to receive
            
        Raises:
            ValueError: If target ID is invalid
        """
        if not self._is_valid_target(message.target_id) and message.target_id != "broadcast":
            raise ValueError(f"Invalid target ID for receiving: {message.target_id}")
            
        # Convert lists back to numpy arrays
        content = {}
        for key, value in message.content.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                content[key] = np.array(value)
            elif isinstance(value, dict):
                content[key] = {}
                for k, v in value.items():
                    if isinstance(v, list) or isinstance(v, np.ndarray):
                        content[key][k] = np.array(v)
                    else:
                        content[key][k] = v
            else:
                content[key] = value
                
        # Create new message with numpy arrays
        array_message = Message(
            source_id=message.source_id,
            target_id=message.target_id,
            content=content,
            message_type=message.message_type,
            timestamp=message.timestamp
        )
            
        self.incoming_queue.append(array_message)
        logger.info(f"Received message for {message.target_id}")
        
        # Only add to history when actually receiving, not just queueing
        if message.target_id == self.id or message.target_id == "broadcast":
            self.received_messages.append(array_message)
            self.message_history.append(array_message)
    
    def filter_messages(self, target_id: Optional[str] = None, message_type: Optional[str] = None) -> List[Message]:
        """Filter messages based on target ID and/or message type.
        
        Args:
            target_id: Optional target ID to filter by
            message_type: Optional message type to filter by
            
        Returns:
            List of filtered messages
        """
        if target_id and not self._is_valid_target(target_id) and target_id != self.id:
            raise ValueError(f"Invalid target ID: {target_id}")
            
        filtered = self.received_messages.copy()
        
        if target_id:
            filtered = [m for m in filtered if m.target_id == target_id or 
                       m.target_id == "broadcast" or
                       (m.target_id == self.id and target_id == self.id)]
            
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
            
        return filtered
    
    def propagate(self, source: Any, message: Dict[str, Any]) -> None:
        """Propagate a message to all connected Things.
        
        Args:
            source: Thing sending the message
            message: Message content to propagate
        """
        if source.id not in self.connections:
            raise KeyError(f"No connections found for Thing {source.id}")
        
        for target_id in self.connections[source.id]:
            msg = Message(
                source_id=source.id,
                target_id=target_id,
                content=message,
                message_type="propagate",
                timestamp=np.datetime64('now')
            )
            self.send_message(msg)
    
    def has_connections(self) -> bool:
        """Check if this Thing has any connections."""
        return bool(self.connections.get(self.id, []))
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of message passing."""
        return {
            'id': self.id,
            'connections': self.connections,
            'n_received': len(self.received_messages),
            'n_sent': len(self.sent_messages),
            'incoming_queue_size': len(self.incoming_queue),
            'outgoing_queue_size': len(self.outgoing_queue),
            'history_size': len(self.message_history)
        } 