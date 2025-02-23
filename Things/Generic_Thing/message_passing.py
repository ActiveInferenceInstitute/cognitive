from typing import List, Dict, Any
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from logger import logger

@dataclass
class Message:
    """Message class for communication between Things."""
    source_id: str
    target_id: str
    content: Any
    message_type: str
    timestamp: datetime = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validate content is serializable
        try:
            import json
            json.dumps(self.content)
        except (TypeError, ValueError):
            raise TypeError("Message content must be JSON serializable")

class MessagePassing:
    """Handles message passing between Generic Things."""
    
    def __init__(self, id: str, max_history: int = 1000):
        """Initialize message passing system."""
        self.id = id
        self.max_history = max_history
        self.connections = {}  # Initialize empty connections dict
        self.received_messages = []
        self.sent_messages = []
        self.incoming_queue = deque()
        self.outgoing_queue = deque()
        self.message_history = deque(maxlen=max_history)
        
    def connect(self, source_id: str, target_id: str):
        """Create bidirectional connection between two things."""
        # Initialize connection lists if needed
        if source_id not in self.connections:
            self.connections[source_id] = []
        if target_id not in self.connections:
            self.connections[target_id] = []
            
        # Add bidirectional connection
        if target_id not in self.connections[source_id]:
            self.connections[source_id].append(target_id)
            logger.info(f"Added connection from {source_id} to {target_id}")
            
        if source_id not in self.connections[target_id]:
            self.connections[target_id].append(source_id)
            logger.info(f"Added connection from {target_id} to {source_id}")
    
    def disconnect(self, source_id: str, target_id: str):
        """Remove bidirectional connection between two things."""
        if source_id in self.connections and target_id in self.connections[source_id]:
            self.connections[source_id].remove(target_id)
            logger.info(f"Removed connection from {source_id} to {target_id}")
            
        if target_id in self.connections and source_id in self.connections[target_id]:
            self.connections[target_id].remove(source_id)
            logger.info(f"Removed connection from {target_id} to {source_id}")
    
    def send_message(self, message: Message):
        """Send message to target(s)."""
        if not isinstance(message, Message):
            raise TypeError("Message must be a Message object")
            
        # Add source to connections if not present
        if message.source_id not in self.connections:
            self.connections[message.source_id] = []
            
        # Handle broadcast messages
        if message.target_id == "broadcast":
            # Create copy of message for each connection to preserve original
            targets = self.connections[message.source_id].copy()
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
                # Update message history immediately
                self.message_history.append(broadcast_msg)
            logger.info(f"Sent broadcast message from {message.source_id} to {len(targets)} targets")
        else:
            # Validate target exists
            if message.target_id not in self.connections:
                self.connect(message.source_id, message.target_id)
                
            # Add message to outgoing queue and history
            self.outgoing_queue.append(message)
            self.sent_messages.append(message)
            self.message_history.append(message)
            logger.info(f"Sent message from {message.source_id} to {message.target_id}")
    
    def process_outgoing(self):
        """Process all outgoing messages."""
        processed = []
        while self.outgoing_queue:
            message = self.outgoing_queue.popleft()
            # Validate target still exists
            if message.target_id in self.connections:
                self.incoming_queue.append(message)
                processed.append(message)
                logger.info(f"Processed outgoing message from {message.source_id} to {message.target_id}")
        return processed
    
    def receive(self, target_id: str) -> List[Message]:
        """Receive all messages for target."""
        # Add target to connections if not present
        if target_id not in self.connections:
            self.connections[target_id] = []
            
        # Filter messages for target
        received = [
            msg for msg in self.incoming_queue 
            if msg.target_id == target_id or msg.target_id == "broadcast"
        ]
        
        # Remove received messages from queue
        self.incoming_queue = deque([
            msg for msg in self.incoming_queue 
            if msg not in received
        ])
        
        # Add to received messages history
        self.received_messages.extend(received)
        
        logger.info(f"Retrieved {len(received)} messages for {target_id}")
        return received
    
    def get_message_history(self) -> List[Message]:
        """Get history of all messages."""
        return list(self.message_history) 