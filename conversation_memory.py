import time
import os
from collections import defaultdict
from user_profiler import UserProfiler

class ConversationMemory:
    """
    A class to manage conversation history for the friendship bot.
    Stores messages for each user and provides methods to retrieve conversation context.
    Integrates with UserProfiler to track user preferences and adapt responses.
    """

    def __init__(self, max_history=20, expiry_time=86400):
        """
        Initialize the conversation memory.

        Args:
            max_history (int): Maximum number of messages to store per user
            expiry_time (int): Time in seconds after which messages expire (default: 24 hours)
        """
        self.conversations = defaultdict(list)
        self.max_history = max_history
        self.expiry_time = expiry_time
        self.user_profiler = UserProfiler()

        # Create directory for user profiles
        os.makedirs('data/user_profiles', exist_ok=True)

    def add_message(self, user_id, content, is_bot=False, emotion=None):
        """
        Add a message to the conversation history and update user profile.

        Args:
            user_id: The ID of the user or bot
            content: The message content
            is_bot: Whether the message is from the bot
            emotion: The detected emotion (if available)
        """
        # Create message object
        message = {
            'user_id': user_id,
            'content': content,
            'timestamp': time.time(),
            'is_bot': is_bot
        }

        # Add to conversation history
        self.conversations[user_id].append(message)

        # Trim history if it exceeds max_history
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]

        # Clean up expired messages
        self._clean_expired_messages()

        # Update user profile if the message is from a user (not the bot)
        if not is_bot:
            history = self.get_history(user_id)
            self.user_profiler.update_profile(user_id, content, emotion, history)

    def get_history(self, user_id, limit=None):
        """
        Get conversation history for a user.

        Args:
            user_id: The ID of the user
            limit: Maximum number of messages to return (default: all)

        Returns:
            List of message objects
        """
        # Clean up expired messages first
        self._clean_expired_messages()

        # Get history for the user
        history = self.conversations[user_id]

        # Apply limit if specified
        if limit and limit > 0:
            history = history[-limit:]

        return history

    def clear_history(self, user_id):
        """
        Clear conversation history for a user.

        Args:
            user_id: The ID of the user
        """
        self.conversations[user_id] = []

    def _clean_expired_messages(self):
        """Remove expired messages from all conversations."""
        current_time = time.time()

        for user_id in self.conversations:
            # Filter out expired messages
            self.conversations[user_id] = [
                msg for msg in self.conversations[user_id]
                if current_time - msg['timestamp'] < self.expiry_time
            ]

    def get_sentiment_history(self, user_id, limit=5):
        """
        Get a history of user messages for sentiment analysis.

        Args:
            user_id: The ID of the user
            limit: Maximum number of messages to return

        Returns:
            List of message contents
        """
        history = self.get_history(user_id)

        # Filter for only user messages (not bot responses)
        user_messages = [msg['content'] for msg in history if msg['user_id'] == user_id]

        # Apply limit
        if limit and limit > 0:
            user_messages = user_messages[-limit:]

        return user_messages

    def get_conversation_summary(self, user_id):
        """
        Get a summary of the conversation with a user.

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary with conversation statistics
        """
        history = self.get_history(user_id)

        if not history:
            return {
                'message_count': 0,
                'user_message_count': 0,
                'bot_message_count': 0,
                'conversation_duration': 0,
                'average_response_time': 0
            }

        # Count messages
        user_messages = [msg for msg in history if msg['user_id'] == user_id]
        bot_messages = [msg for msg in history if msg['user_id'] != user_id]

        # Calculate conversation duration
        if len(history) >= 2:
            first_msg_time = history[0]['timestamp']
            last_msg_time = history[-1]['timestamp']
            duration = last_msg_time - first_msg_time
        else:
            duration = 0

        # Calculate average response time
        response_times = []
        for i in range(1, len(history)):
            if history[i]['user_id'] != history[i-1]['user_id']:
                response_time = history[i]['timestamp'] - history[i-1]['timestamp']
                response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            'message_count': len(history),
            'user_message_count': len(user_messages),
            'bot_message_count': len(bot_messages),
            'conversation_duration': duration,
            'average_response_time': avg_response_time
        }

    def get_user_profile(self, user_id):
        """
        Get a user's profile.

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary containing the user's profile
        """
        return self.user_profiler.load_profile(user_id)

    def get_adaptation_recommendations(self, user_id):
        """
        Get adaptation recommendations for a user.

        Args:
            user_id: The ID of the user

        Returns:
            Dictionary with adaptation recommendations
        """
        return self.user_profiler.get_adaptation_recommendations(user_id)

    def set_user_preference(self, user_id, category, preference, value):
        """
        Set an explicit preference for a user.

        Args:
            user_id: The ID of the user
            category: The preference category
            preference: The specific preference
            value: The preference value

        Returns:
            Updated user profile
        """
        return self.user_profiler.set_explicit_preference(user_id, category, preference, value)
