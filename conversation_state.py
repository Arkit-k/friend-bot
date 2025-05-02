"""
Conversation State Manager for Friendship Bot

This module manages conversation states to prevent interruptions
and maintain focused conversations with individual users.
"""

import time
import json
import os
from enum import Enum
from typing import Dict, Optional, List, Tuple

class ConversationStatus(Enum):
    """Enum representing the status of a conversation."""
    IDLE = "idle"  # Not in an active conversation
    ACTIVE = "active"  # In an active conversation
    PROTECTED = "protected"  # In a protected conversation that can't be interrupted

class ConversationStateManager:
    """
    Manages conversation states to prevent interruptions and maintain
    focused conversations with individual users.
    """
    
    def __init__(self, state_file="data/conversation_states.json", idle_timeout=300):
        """
        Initialize the conversation state manager.
        
        Args:
            state_file: Path to the file for storing conversation states
            idle_timeout: Time in seconds after which a conversation is considered idle
        """
        self.state_file = state_file
        self.idle_timeout = idle_timeout
        self.states = {}
        self.load_states()
        
        # Create directory for state file if it doesn't exist
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    def load_states(self):
        """Load conversation states from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    
                    # Convert string keys to integers (user IDs)
                    self.states = {int(k): v for k, v in data.items()}
                    
                    # Update timestamps for loaded states
                    current_time = time.time()
                    for user_id, state in self.states.items():
                        # If the last activity was too long ago, reset to IDLE
                        if current_time - state.get("last_activity", 0) > self.idle_timeout:
                            state["status"] = ConversationStatus.IDLE.value
            except Exception as e:
                print(f"Error loading conversation states: {e}")
                self.states = {}
    
    def save_states(self):
        """Save conversation states to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.states, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation states: {e}")
    
    def get_state(self, user_id):
        """
        Get the conversation state for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dictionary containing the user's conversation state
        """
        if user_id not in self.states:
            # Create default state for new users
            self.states[user_id] = {
                "status": ConversationStatus.IDLE.value,
                "last_activity": time.time(),
                "current_channel": None,
                "protected_since": None
            }
        
        return self.states[user_id]
    
    def update_activity(self, user_id, channel_id=None):
        """
        Update the last activity timestamp for a user.
        
        Args:
            user_id: The ID of the user
            channel_id: The ID of the channel (if applicable)
        """
        state = self.get_state(user_id)
        state["last_activity"] = time.time()
        
        if channel_id is not None:
            state["current_channel"] = channel_id
        
        self.save_states()
    
    def start_protected_conversation(self, user_id, channel_id):
        """
        Start a protected conversation with a user.
        
        Args:
            user_id: The ID of the user
            channel_id: The ID of the channel
            
        Returns:
            Boolean indicating success
        """
        state = self.get_state(user_id)
        
        # Check if the user is already in a protected conversation
        if state["status"] == ConversationStatus.PROTECTED.value:
            return False
        
        # Start protected conversation
        state["status"] = ConversationStatus.PROTECTED.value
        state["last_activity"] = time.time()
        state["current_channel"] = channel_id
        state["protected_since"] = time.time()
        
        self.save_states()
        return True
    
    def end_protected_conversation(self, user_id):
        """
        End a protected conversation with a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Boolean indicating success
        """
        state = self.get_state(user_id)
        
        # Check if the user is in a protected conversation
        if state["status"] != ConversationStatus.PROTECTED.value:
            return False
        
        # End protected conversation
        state["status"] = ConversationStatus.ACTIVE.value
        state["last_activity"] = time.time()
        state["protected_since"] = None
        
        self.save_states()
        return True
    
    def can_interrupt(self, user_id, channel_id):
        """
        Check if a user can interrupt the current conversation in a channel.
        
        Args:
            user_id: The ID of the user
            channel_id: The ID of the channel
            
        Returns:
            Tuple of (can_interrupt, current_user_id)
        """
        # Check if any user has a protected conversation in this channel
        for current_user_id, state in self.states.items():
            # Skip if it's the same user
            if current_user_id == user_id:
                continue
            
            # Check if another user has a protected conversation in this channel
            if (state["status"] == ConversationStatus.PROTECTED.value and 
                state["current_channel"] == channel_id):
                
                # Check if the conversation is still active (not timed out)
                if time.time() - state["last_activity"] <= self.idle_timeout:
                    return False, current_user_id
        
        return True, None
    
    def get_active_users_in_channel(self, channel_id):
        """
        Get a list of users with active conversations in a channel.
        
        Args:
            channel_id: The ID of the channel
            
        Returns:
            List of user IDs
        """
        active_users = []
        
        for user_id, state in self.states.items():
            if (state["status"] in [ConversationStatus.ACTIVE.value, ConversationStatus.PROTECTED.value] and 
                state["current_channel"] == channel_id and
                time.time() - state["last_activity"] <= self.idle_timeout):
                active_users.append(user_id)
        
        return active_users
    
    def cleanup_idle_conversations(self):
        """Clean up idle conversations."""
        current_time = time.time()
        
        for user_id, state in list(self.states.items()):
            # If the last activity was too long ago, reset to IDLE
            if current_time - state.get("last_activity", 0) > self.idle_timeout:
                state["status"] = ConversationStatus.IDLE.value
                state["protected_since"] = None
        
        self.save_states()
    
    def get_conversation_duration(self, user_id):
        """
        Get the duration of a user's current conversation.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Duration in seconds or None if not in a conversation
        """
        state = self.get_state(user_id)
        
        if state["status"] == ConversationStatus.IDLE.value:
            return None
        
        if state["status"] == ConversationStatus.PROTECTED.value and state["protected_since"]:
            return time.time() - state["protected_since"]
        
        return time.time() - state["last_activity"]
