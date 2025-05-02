"""
User Profiler for Friendship Bot

This module analyzes user messages and interactions to build a profile of the user's
communication style, preferences, and personality traits. The bot uses this profile
to adapt its responses to better match the user's needs and preferences.
"""

import re
import json
import os
import nltk
from collections import Counter
import numpy as np
from datetime import datetime, timedelta

# Create necessary directories
os.makedirs('data/user_profiles', exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Communication style dimensions
COMMUNICATION_STYLES = {
    "formality": {
        "formal": ["would", "could", "should", "perhaps", "therefore", "however", "nevertheless", 
                  "furthermore", "additionally", "consequently", "regarding", "concerning"],
        "informal": ["yeah", "cool", "awesome", "lol", "haha", "btw", "gonna", "wanna", "kinda", 
                    "sorta", "like", "totally", "literally", "u", "ur", "r", "y", "k", "ok"]
    },
    "verbosity": {
        "verbose": [],  # Determined by message length
        "concise": []   # Determined by message length
    },
    "emotionality": {
        "emotional": ["feel", "feeling", "felt", "emotion", "emotional", "happy", "sad", "angry", 
                     "upset", "excited", "worried", "anxious", "love", "hate", "miss", "afraid",
                     "scared", "nervous", "stressed", "overwhelmed", "thrilled", "delighted"],
        "logical": ["think", "thinking", "thought", "logic", "logical", "reason", "reasonable", 
                   "analysis", "analyze", "consider", "evaluate", "assess", "determine", "conclude",
                   "therefore", "thus", "hence", "consequently", "because", "since", "due to"]
    },
    "assertiveness": {
        "assertive": ["need", "want", "must", "should", "definitely", "absolutely", "certainly", 
                     "surely", "clearly", "obviously", "undoubtedly", "unquestionably", "will", 
                     "won't", "can't", "don't"],
        "tentative": ["maybe", "perhaps", "possibly", "might", "may", "could", "would", "sometimes", 
                     "occasionally", "potentially", "conceivably", "presumably", "seemingly", "apparently"]
    },
    "positivity": {
        "positive": ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific", 
                    "awesome", "brilliant", "outstanding", "superb", "perfect", "happy", "glad", 
                    "pleased", "delighted", "excited", "thrilled", "love", "enjoy", "appreciate"],
        "negative": ["bad", "terrible", "awful", "horrible", "poor", "disappointing", "unfortunate", 
                    "sad", "unhappy", "upset", "angry", "frustrated", "annoyed", "irritated", "hate", 
                    "dislike", "despise", "resent", "regret", "sorry"]
    },
    "humor": {
        "humorous": ["lol", "haha", "hehe", "lmao", "rofl", "funny", "joke", "hilarious", "amusing", 
                    "laugh", "laughing", "laughed", "humor", "humorous", "comedy", "comedic", "witty", 
                    "silly", "goofy", "ridiculous", "absurd", "😂", "🤣", "😆", "😄", "😅"],
        "serious": ["serious", "seriously", "important", "critical", "crucial", "essential", "vital", 
                   "significant", "substantial", "considerable", "meaningful", "relevant", "pertinent"]
    }
}

# Topic preferences
TOPIC_CATEGORIES = {
    "personal": ["family", "friend", "relationship", "feeling", "emotion", "life", "experience", 
                "memory", "childhood", "grow", "growing", "grew", "personal", "private", "secret"],
    "professional": ["work", "job", "career", "profession", "business", "company", "office", "colleague", 
                    "coworker", "boss", "manager", "employee", "client", "customer", "project", "task"],
    "intellectual": ["book", "read", "reading", "write", "writing", "wrote", "learn", "learning", 
                    "learned", "study", "studying", "studied", "knowledge", "idea", "concept", "theory", 
                    "philosophy", "science", "scientific", "research", "academic", "intellectual"],
    "recreational": ["game", "play", "playing", "played", "sport", "exercise", "workout", "hobby", 
                    "interest", "music", "movie", "film", "show", "tv", "television", "watch", "watching", 
                    "watched", "listen", "listening", "listened", "travel", "traveling", "travelled"],
    "social": ["friend", "friendship", "relationship", "date", "dating", "party", "event", "gathering", 
              "meet", "meeting", "met", "talk", "talking", "talked", "chat", "chatting", "chatted", 
              "conversation", "discuss", "discussing", "discussed", "social", "socialize", "socializing"],
    "emotional": ["feel", "feeling", "felt", "emotion", "emotional", "happy", "sad", "angry", "upset", 
                 "excited", "worried", "anxious", "love", "hate", "miss", "afraid", "scared", "nervous", 
                 "stressed", "overwhelmed", "thrilled", "delighted"]
}

# Response preferences
RESPONSE_PREFERENCES = {
    "advice_seeking": ["advice", "suggest", "suggestion", "recommend", "recommendation", "help", 
                      "guidance", "guide", "tip", "hint", "what should", "how should", "what would", 
                      "how would", "what could", "how could", "what can", "how can"],
    "validation_seeking": ["right?", "don't you think?", "isn't it?", "wouldn't you?", "don't you?", 
                          "am i right?", "do you agree?", "what do you think?", "your thoughts?", 
                          "your opinion?", "make sense?", "sound good?"],
    "emotional_support": ["sad", "upset", "worried", "anxious", "stressed", "overwhelmed", "scared", 
                         "afraid", "nervous", "depressed", "lonely", "alone", "isolated", "hurt", 
                         "painful", "suffering", "struggle", "struggling", "difficult", "hard", "tough"],
    "practical_support": ["help", "assist", "support", "aid", "guidance", "direction", "instruction", 
                         "information", "resource", "tool", "strategy", "plan", "approach", "method", 
                         "technique", "solution", "resolve", "fix", "solve", "address", "tackle", "handle"],
    "intellectual_discussion": ["think", "thought", "idea", "concept", "theory", "philosophy", "perspective", 
                               "viewpoint", "opinion", "belief", "consider", "consideration", "reflect", 
                               "reflection", "analyze", "analysis", "evaluate", "evaluation", "assess", 
                               "assessment", "examine", "examination", "explore", "exploration"]
}

# Interaction preferences
INTERACTION_PREFERENCES = {
    "response_length": {
        "long": [],  # Determined by user's average message length
        "short": []  # Determined by user's average message length
    },
    "response_time": {
        "quick": [],  # Determined by user's response patterns
        "thoughtful": []  # Determined by user's response patterns
    },
    "conversation_pacing": {
        "rapid": [],  # Determined by message frequency
        "measured": []  # Determined by message frequency
    },
    "initiative": {
        "user_led": [],  # Determined by who initiates topics
        "bot_led": []  # Determined by who initiates topics
    }
}

class UserProfiler:
    """
    Analyzes user messages and interactions to build a profile of the user's
    communication style, preferences, and personality traits.
    """
    
    def __init__(self, profile_dir='data/user_profiles'):
        """
        Initialize the user profiler.
        
        Args:
            profile_dir: Directory to store user profiles
        """
        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)
    
    def get_profile_path(self, user_id):
        """Get the file path for a user's profile."""
        return os.path.join(self.profile_dir, f"user_{user_id}.json")
    
    def load_profile(self, user_id):
        """
        Load a user's profile from disk.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dictionary containing the user's profile
        """
        profile_path = self.get_profile_path(user_id)
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading profile for user {user_id}: {e}")
        
        # Return default profile if no profile exists or there was an error
        return self.create_default_profile()
    
    def save_profile(self, user_id, profile):
        """
        Save a user's profile to disk.
        
        Args:
            user_id: The ID of the user
            profile: Dictionary containing the user's profile
        """
        profile_path = self.get_profile_path(user_id)
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile for user {user_id}: {e}")
    
    def create_default_profile(self):
        """
        Create a default user profile.
        
        Returns:
            Dictionary containing a default user profile
        """
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
            "average_message_length": 0,
            "communication_style": {
                "formality": 0.5,  # 0 = very informal, 1 = very formal
                "verbosity": 0.5,  # 0 = very concise, 1 = very verbose
                "emotionality": 0.5,  # 0 = very logical, 1 = very emotional
                "assertiveness": 0.5,  # 0 = very tentative, 1 = very assertive
                "positivity": 0.5,  # 0 = very negative, 1 = very positive
                "humor": 0.5  # 0 = very serious, 1 = very humorous
            },
            "topic_preferences": {
                "personal": 0.5,
                "professional": 0.5,
                "intellectual": 0.5,
                "recreational": 0.5,
                "social": 0.5,
                "emotional": 0.5
            },
            "response_preferences": {
                "advice_seeking": 0.5,
                "validation_seeking": 0.5,
                "emotional_support": 0.5,
                "practical_support": 0.5,
                "intellectual_discussion": 0.5
            },
            "interaction_preferences": {
                "response_length": 0.5,  # 0 = very short, 1 = very long
                "response_time": 0.5,  # 0 = very quick, 1 = very thoughtful
                "conversation_pacing": 0.5,  # 0 = very rapid, 1 = very measured
                "initiative": 0.5  # 0 = very user-led, 1 = very bot-led
            },
            "explicit_preferences": {
                # Preferences explicitly set by the user
            },
            "detected_interests": {},
            "detected_dislikes": {},
            "conversation_history_stats": {
                "total_conversations": 0,
                "total_messages": 0,
                "average_conversation_length": 0,
                "peak_activity_times": {}
            },
            "emotional_patterns": {
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "surprise": 0,
                "neutral": 0
            }
        }
    
    def update_profile(self, user_id, message, emotion=None, conversation_history=None):
        """
        Update a user's profile based on a new message.
        
        Args:
            user_id: The ID of the user
            message: The message content
            emotion: The detected emotion (if available)
            conversation_history: The conversation history (if available)
            
        Returns:
            Updated user profile
        """
        # Load existing profile or create a new one
        profile = self.load_profile(user_id)
        
        # Update basic stats
        profile["updated_at"] = datetime.now().isoformat()
        profile["message_count"] += 1
        
        # Update average message length
        current_length = len(message.split())
        profile["average_message_length"] = (
            (profile["average_message_length"] * (profile["message_count"] - 1) + current_length) / 
            profile["message_count"]
        )
        
        # Update communication style
        self._update_communication_style(profile, message)
        
        # Update topic preferences
        self._update_topic_preferences(profile, message)
        
        # Update response preferences
        self._update_response_preferences(profile, message)
        
        # Update interaction preferences
        if conversation_history:
            self._update_interaction_preferences(profile, message, conversation_history)
        
        # Update emotional patterns
        if emotion:
            self._update_emotional_patterns(profile, emotion)
        
        # Update interests and dislikes
        self._update_interests_and_dislikes(profile, message)
        
        # Save the updated profile
        self.save_profile(user_id, profile)
        
        return profile
    
    def _update_communication_style(self, profile, message):
        """Update the communication style section of the profile."""
        message_lower = message.lower()
        words = nltk.word_tokenize(message_lower)
        
        # Update formality
        formal_count = sum(1 for word in words if word in COMMUNICATION_STYLES["formality"]["formal"])
        informal_count = sum(1 for word in words if word in COMMUNICATION_STYLES["formality"]["informal"])
        
        if formal_count + informal_count > 0:
            formality_score = formal_count / (formal_count + informal_count)
            # Smooth the update to avoid drastic changes
            profile["communication_style"]["formality"] = (
                0.9 * profile["communication_style"]["formality"] + 0.1 * formality_score
            )
        
        # Update verbosity based on message length
        message_length = len(words)
        verbosity_score = min(1.0, message_length / 30)  # Normalize: 30+ words is considered verbose
        profile["communication_style"]["verbosity"] = (
            0.9 * profile["communication_style"]["verbosity"] + 0.1 * verbosity_score
        )
        
        # Update emotionality
        emotional_count = sum(1 for word in words if word in COMMUNICATION_STYLES["emotionality"]["emotional"])
        logical_count = sum(1 for word in words if word in COMMUNICATION_STYLES["emotionality"]["logical"])
        
        if emotional_count + logical_count > 0:
            emotionality_score = emotional_count / (emotional_count + logical_count)
            profile["communication_style"]["emotionality"] = (
                0.9 * profile["communication_style"]["emotionality"] + 0.1 * emotionality_score
            )
        
        # Update assertiveness
        assertive_count = sum(1 for word in words if word in COMMUNICATION_STYLES["assertiveness"]["assertive"])
        tentative_count = sum(1 for word in words if word in COMMUNICATION_STYLES["assertiveness"]["tentative"])
        
        if assertive_count + tentative_count > 0:
            assertiveness_score = assertive_count / (assertive_count + tentative_count)
            profile["communication_style"]["assertiveness"] = (
                0.9 * profile["communication_style"]["assertiveness"] + 0.1 * assertiveness_score
            )
        
        # Update positivity
        positive_count = sum(1 for word in words if word in COMMUNICATION_STYLES["positivity"]["positive"])
        negative_count = sum(1 for word in words if word in COMMUNICATION_STYLES["positivity"]["negative"])
        
        if positive_count + negative_count > 0:
            positivity_score = positive_count / (positive_count + negative_count)
            profile["communication_style"]["positivity"] = (
                0.9 * profile["communication_style"]["positivity"] + 0.1 * positivity_score
            )
        
        # Update humor
        humorous_count = sum(1 for word in words if word in COMMUNICATION_STYLES["humor"]["humorous"])
        serious_count = sum(1 for word in words if word in COMMUNICATION_STYLES["humor"]["serious"])
        
        if humorous_count + serious_count > 0:
            humor_score = humorous_count / (humorous_count + serious_count)
            profile["communication_style"]["humor"] = (
                0.9 * profile["communication_style"]["humor"] + 0.1 * humor_score
            )
    
    def _update_topic_preferences(self, profile, message):
        """Update the topic preferences section of the profile."""
        message_lower = message.lower()
        words = nltk.word_tokenize(message_lower)
        
        # Count occurrences of topic-related words
        topic_counts = {}
        for topic, keywords in TOPIC_CATEGORIES.items():
            topic_counts[topic] = sum(1 for word in words if word in keywords)
        
        # Update topic preferences
        total_count = sum(topic_counts.values())
        if total_count > 0:
            for topic, count in topic_counts.items():
                if count > 0:
                    # Calculate the topic score and update the profile
                    topic_score = count / total_count
                    profile["topic_preferences"][topic] = (
                        0.9 * profile["topic_preferences"][topic] + 0.1 * topic_score
                    )
    
    def _update_response_preferences(self, profile, message):
        """Update the response preferences section of the profile."""
        message_lower = message.lower()
        
        # Check for response preference indicators
        for preference, indicators in RESPONSE_PREFERENCES.items():
            # Count occurrences of preference indicators
            indicator_count = sum(1 for indicator in indicators if indicator in message_lower)
            
            if indicator_count > 0:
                # Increase the preference score
                preference_score = min(1.0, indicator_count / 3)  # Normalize: 3+ indicators is strong
                profile["response_preferences"][preference] = (
                    0.9 * profile["response_preferences"][preference] + 0.1 * preference_score
                )
    
    def _update_interaction_preferences(self, profile, message, conversation_history):
        """Update the interaction preferences section of the profile."""
        # Update response length preference based on user's message length
        message_length = len(message.split())
        length_score = min(1.0, message_length / 30)  # Normalize: 30+ words is considered long
        profile["interaction_preferences"]["response_length"] = (
            0.9 * profile["interaction_preferences"]["response_length"] + 0.1 * length_score
        )
        
        # Update conversation pacing based on message frequency
        if len(conversation_history) >= 2:
            # Calculate time between messages
            user_messages = [msg for msg in conversation_history if msg['user_id'] == conversation_history[0]['user_id']]
            if len(user_messages) >= 2:
                time_diffs = []
                for i in range(1, len(user_messages)):
                    time_diff = user_messages[i]['timestamp'] - user_messages[i-1]['timestamp']
                    time_diffs.append(time_diff)
                
                if time_diffs:
                    avg_time_diff = sum(time_diffs) / len(time_diffs)
                    # Normalize: 60+ seconds is considered measured pacing
                    pacing_score = min(1.0, avg_time_diff / 60)
                    profile["interaction_preferences"]["conversation_pacing"] = (
                        0.9 * profile["interaction_preferences"]["conversation_pacing"] + 0.1 * pacing_score
                    )
        
        # Update initiative preference based on conversation flow
        # This is a simplified approach; a more sophisticated analysis would be better
        if len(conversation_history) >= 3:
            # Count how often the user asks questions
            user_messages = [msg for msg in conversation_history if msg['user_id'] == conversation_history[0]['user_id']]
            question_count = sum(1 for msg in user_messages if '?' in msg['content'])
            question_ratio = question_count / len(user_messages) if user_messages else 0
            
            # Higher question ratio indicates user-led conversations
            initiative_score = 1.0 - min(1.0, question_ratio * 2)  # Normalize and invert
            profile["interaction_preferences"]["initiative"] = (
                0.9 * profile["interaction_preferences"]["initiative"] + 0.1 * initiative_score
            )
    
    def _update_emotional_patterns(self, profile, emotion):
        """Update the emotional patterns section of the profile."""
        emotion_lower = emotion.lower()
        
        # Increment the count for the detected emotion
        if emotion_lower in profile["emotional_patterns"]:
            profile["emotional_patterns"][emotion_lower] += 1
        else:
            # Default to neutral if the emotion is not recognized
            profile["emotional_patterns"]["neutral"] += 1
    
    def _update_interests_and_dislikes(self, profile, message):
        """Update the detected interests and dislikes based on message content."""
        message_lower = message.lower()
        
        # Look for explicit mentions of likes and dislikes
        like_patterns = [
            r"i (?:really |absolutely |totally |kind of |kinda |)(?:like|love|enjoy|adore|am into) ([^.,!?]*)",
            r"i'm (?:really |absolutely |totally |kind of |kinda |)(?:into|fond of|passionate about) ([^.,!?]*)",
            r"([^.,!?]*) (?:is|are) (?:my favorite|my passion|what i love)"
        ]
        
        dislike_patterns = [
            r"i (?:really |absolutely |totally |kind of |kinda |)(?:dislike|hate|don't like|can't stand|detest) ([^.,!?]*)",
            r"i'm (?:really |absolutely |totally |kind of |kinda |)(?:not into|not fond of|tired of|sick of) ([^.,!?]*)",
            r"([^.,!?]*) (?:annoys|bothers|irritates|frustrates) me"
        ]
        
        # Extract interests
        for pattern in like_patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                interest = match.strip()
                if interest and len(interest) > 2:  # Avoid very short matches
                    if interest in profile["detected_interests"]:
                        profile["detected_interests"][interest] += 1
                    else:
                        profile["detected_interests"][interest] = 1
        
        # Extract dislikes
        for pattern in dislike_patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                dislike = match.strip()
                if dislike and len(dislike) > 2:  # Avoid very short matches
                    if dislike in profile["detected_dislikes"]:
                        profile["detected_dislikes"][dislike] += 1
                    else:
                        profile["detected_dislikes"][dislike] = 1
    
    def get_communication_style_adaptation(self, profile):
        """
        Get adaptation recommendations based on the user's communication style.
        
        Args:
            profile: The user's profile
            
        Returns:
            Dictionary with adaptation recommendations
        """
        style = profile["communication_style"]
        
        adaptations = {
            "formality": style["formality"],
            "verbosity": style["verbosity"],
            "emotionality": style["emotionality"],
            "assertiveness": style["assertiveness"],
            "positivity": style["positivity"],
            "humor": style["humor"]
        }
        
        return adaptations
    
    def get_topic_recommendations(self, profile, limit=3):
        """
        Get topic recommendations based on the user's preferences.
        
        Args:
            profile: The user's profile
            limit: Maximum number of topics to recommend
            
        Returns:
            List of recommended topics
        """
        topics = profile["topic_preferences"]
        
        # Sort topics by preference score
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top N topics
        return [topic for topic, score in sorted_topics[:limit]]
    
    def get_response_style_recommendations(self, profile):
        """
        Get response style recommendations based on the user's preferences.
        
        Args:
            profile: The user's profile
            
        Returns:
            Dictionary with response style recommendations
        """
        response_prefs = profile["response_preferences"]
        interaction_prefs = profile["interaction_preferences"]
        
        # Determine the dominant response style
        dominant_style = max(response_prefs.items(), key=lambda x: x[1])[0]
        
        recommendations = {
            "dominant_style": dominant_style,
            "response_length": interaction_prefs["response_length"],
            "conversation_pacing": interaction_prefs["conversation_pacing"],
            "initiative": interaction_prefs["initiative"]
        }
        
        return recommendations
    
    def get_interests_and_dislikes(self, profile, limit=5):
        """
        Get the user's top interests and dislikes.
        
        Args:
            profile: The user's profile
            limit: Maximum number of interests/dislikes to return
            
        Returns:
            Dictionary with interests and dislikes
        """
        # Sort interests and dislikes by frequency
        sorted_interests = sorted(profile["detected_interests"].items(), key=lambda x: x[1], reverse=True)
        sorted_dislikes = sorted(profile["detected_dislikes"].items(), key=lambda x: x[1], reverse=True)
        
        return {
            "interests": [interest for interest, count in sorted_interests[:limit]],
            "dislikes": [dislike for dislike, count in sorted_dislikes[:limit]]
        }
    
    def set_explicit_preference(self, user_id, category, preference, value):
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
        profile = self.load_profile(user_id)
        
        if "explicit_preferences" not in profile:
            profile["explicit_preferences"] = {}
        
        if category not in profile["explicit_preferences"]:
            profile["explicit_preferences"][category] = {}
        
        profile["explicit_preferences"][category][preference] = value
        profile["updated_at"] = datetime.now().isoformat()
        
        self.save_profile(user_id, profile)
        
        return profile
    
    def get_adaptation_recommendations(self, user_id):
        """
        Get comprehensive adaptation recommendations for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dictionary with adaptation recommendations
        """
        profile = self.load_profile(user_id)
        
        # Get communication style adaptations
        communication_style = self.get_communication_style_adaptation(profile)
        
        # Get topic recommendations
        topics = self.get_topic_recommendations(profile)
        
        # Get response style recommendations
        response_style = self.get_response_style_recommendations(profile)
        
        # Get interests and dislikes
        interests_dislikes = self.get_interests_and_dislikes(profile)
        
        # Combine all recommendations
        recommendations = {
            "communication_style": communication_style,
            "recommended_topics": topics,
            "response_style": response_style,
            "interests": interests_dislikes["interests"],
            "dislikes": interests_dislikes["dislikes"],
            "explicit_preferences": profile.get("explicit_preferences", {})
        }
        
        return recommendations
