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
        """Check if a target ID is valid in the connection graph.
        
        Args:
            target_id: ID to validate
            
        Returns:
            True if target is valid, False otherwise
        """
        # Special cases
        if target_id == "broadcast" or target_id == self.id:
            return True
            
        # Check if target exists in source's connections
        if self.id in self.connections and target_id in self.connections[self.id]:
            return True
            
        # Check if target exists in any connections
        for source_id, targets in self.connections.items():
            if target_id == source_id or target_id in targets:
                return True
                
        return False
    
    def process_outgoing(self) -> None:
        """Process all messages in the outgoing queue."""
        processed_ids = set()
        
        while self.outgoing_queue:
            message = self.outgoing_queue.popleft()
            msg_id = (message.source_id, message.target_id, message.timestamp)
            
            # Skip if we've already processed this message
            if msg_id in processed_ids:
                continue
                
            processed_ids.add(msg_id)
            
            # For broadcast messages
            if message.target_id == "broadcast":
                # Get all connected nodes except the source
                targets = set()
                for node, connections in self.connections.items():
                    targets.update(connections)
                    if node != message.source_id:
                        targets.add(node)
                targets.discard(message.source_id)
                
                # Send to each target
                for target_id in targets:
                    if target_id == self.id:  # If this node is a target
                        self.receive_message(message)
                    else:  # Forward to other targets
                        broadcast_msg = Message(
                            source_id=message.source_id,
                            target_id=target_id,
                            content=message.content.copy(),
                            message_type=message.message_type,
                            timestamp=message.timestamp
                        )
                        self.incoming_queue.append(broadcast_msg)
                logger.info(f"Processed broadcast message from {message.source_id}")
            else:
                # For direct messages
                if self._is_valid_target(message.target_id):
                    if message.target_id == self.id:  # Message for this node
                        self.receive_message(message)
                    else:  # Forward to target
                        self.incoming_queue.append(message)
                    logger.info(f"Processed outgoing message from {message.source_id} to {message.target_id}")
                else:
                    logger.warning(f"Invalid target ID in outgoing message: {message.target_id}")
    
    def receive(self, target_id: str) -> List[Message]:
        """Receive all messages intended for a specific target.
        
        Args:
            target_id: ID of the target Thing
            
        Returns:
            List of messages for the target
        """
        if not self._is_valid_target(target_id) and target_id != self.id:
            logger.warning(f"Invalid target ID for receiving: {target_id}")
            return []
            
        messages = []
        remaining = deque()
        
        # Process any pending outgoing messages first
        self.process_outgoing()
        
        # Collect messages for the target
        while self.incoming_queue:
            message = self.incoming_queue.popleft()
            if message.target_id == target_id or message.target_id == "broadcast":
                messages.append(message)
                # Add to received messages if not already there
                if message not in self.received_messages:
                    self.received_messages.append(message)
                    self.message_history.append(message)
            else:
                remaining.append(message)
                
        # Restore remaining messages
        self.incoming_queue = remaining
        
        # Process any new outgoing messages that may have been generated
        self.process_outgoing()
        
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
                if node != message.source_id:
                    targets.add(node)
            targets.discard(message.source_id)
            
            # Send to each target
            for target_id in targets:
                broadcast_msg = Message(
                    source_id=message.source_id,
                    target_id=target_id,
                    content=message.content.copy(),
                    message_type=message.message_type,
                    timestamp=message.timestamp
                )
                # Add to outgoing queue and sent messages
                self.outgoing_queue.append(broadcast_msg)
                if broadcast_msg not in self.sent_messages:
                    self.sent_messages.append(broadcast_msg)
                    self.message_history.append(broadcast_msg)
                logger.info(f"Sent broadcast message from {message.source_id} to {target_id}")
        else:
            # For direct messages, validate target exists in connection graph
            if not self._is_valid_target(message.target_id):
                raise ValueError(f"Invalid target ID: {message.target_id}")
            
            # Create a copy of the message with copied content
            direct_msg = Message(
                source_id=message.source_id,
                target_id=message.target_id,
                content=message.content.copy(),
                message_type=message.message_type,
                timestamp=message.timestamp
            )
            
            # Add message to outgoing queue and sent messages
            self.outgoing_queue.append(direct_msg)
            if direct_msg not in self.sent_messages:
                self.sent_messages.append(direct_msg)
                self.message_history.append(direct_msg)
            logger.info(f"Sent message from {message.source_id} to {message.target_id}")
            
        # Process outgoing queue immediately
        self.process_outgoing()
    
    def receive_message(self, message: Message) -> None:
        """Receive a message directly into the incoming queue.
        
        Args:
            message: Message to receive
            
        Raises:
            ValueError: If target ID is invalid
        """
        # For broadcast messages or direct messages to this node
        if message.target_id == "broadcast" or message.target_id == self.id:
            # Convert message content to numpy arrays
            content = {}
            for key, value in message.content.items():
                if isinstance(value, dict):
                    content[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (list, np.ndarray)):
                            content[key][k] = np.array(v)
                        else:
                            content[key][k] = v
                elif isinstance(value, (list, np.ndarray)):
                    content[key] = np.array(value)
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
                
            # Add to incoming queue and received messages
            self.incoming_queue.append(array_message)
            if array_message not in self.received_messages:
                self.received_messages.append(array_message)
                self.message_history.append(array_message)
            logger.info(f"Received message for {message.target_id}")
            
            # Process the message immediately
            self.process_outgoing()
        else:
            # For messages to other nodes, validate target and forward
            if not self._is_valid_target(message.target_id):
                raise ValueError(f"Invalid target ID: {message.target_id}")
            
            # Add to incoming queue for forwarding
            self.incoming_queue.append(message)
            logger.info(f"Forwarding message to {message.target_id}")
    
    def filter_messages(self, target_id: Optional[str] = None, message_type: Optional[str] = None) -> List[Message]:
        """Filter messages based on target ID and/or message type.
        
        Args:
            target_id: Optional target ID to filter by
            message_type: Optional message type to filter by
            
        Returns:
            List of filtered messages
        """
        # Process any pending messages first
        self.process_outgoing()
        
        # If target_id is specified, validate it
        if target_id:
            if not self._is_valid_target(target_id) and target_id != self.id and target_id != "broadcast":
                raise ValueError(f"Invalid target ID: {target_id}")
        
        # Start with all received messages
        filtered = self.received_messages.copy()
        
        # Filter by target ID if specified
        if target_id:
            filtered = [m for m in filtered if 
                       m.target_id == target_id or 
                       m.target_id == "broadcast" or
                       (target_id == self.id and m.target_id == self.id)]
            
        # Filter by message type if specified
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