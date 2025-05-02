"""
Gemini API Integration for Friendship Bot

This module provides integration with Google's Gemini API to handle questions
that the bot's primary model cannot answer confidently.
"""

import os
import json
import requests
import time
import re
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "api_key": "",  # To be set by the user
    "model": "gemini-pro",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "top_k": 40,
    "use_for_unknown_questions": True,
    "confidence_threshold": 0.6,  # Threshold below which Gemini is used
    "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
}

class GeminiAPI:
    """
    A class to interact with Google's Gemini API for handling questions
    that the primary model cannot answer confidently.
    """

    def __init__(self, config_path="config/gemini_config.json"):
        """
        Initialize the Gemini API client.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)

        # Try to get API key from environment variable first, then fall back to config
        env_api_key = os.getenv("GEMINI_API_KEY")
        if env_api_key:
            self.api_key = env_api_key
            self.config["api_key"] = env_api_key  # Update config with env var
        else:
            self.api_key = self.config.get("api_key", "")

        self.model = self.config.get("model", "gemini-pro")
        self.api_url = self.config.get("api_url", DEFAULT_CONFIG["api_url"]).format(model=self.model)

        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or create default if it doesn't exist.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing configuration
        """
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading Gemini config: {e}")
                print("Using default configuration")
                return DEFAULT_CONFIG
        else:
            # Create default config file
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                print(f"Created default Gemini config at {config_path}")
                print("Please set your API key in this file")
            except Exception as e:
                print(f"Error creating default config: {e}")

            return DEFAULT_CONFIG

    def save_config(self, config_path: str) -> bool:
        """
        Save the current configuration to file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Boolean indicating success
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving Gemini config: {e}")
            return False

    def set_api_key(self, api_key: str, config_path: str = "config/gemini_config.json") -> bool:
        """
        Set the API key for Gemini.

        Args:
            api_key: The API key
            config_path: Path to the configuration file

        Returns:
            Boolean indicating success
        """
        self.api_key = api_key
        self.config["api_key"] = api_key

        # Note: This doesn't actually update the .env file, just the in-memory config
        # For a production app, you might want to update the .env file as well
        # But that's generally not recommended for security reasons

        return self.save_config(config_path)

    def is_configured(self) -> bool:
        """
        Check if the API is configured with a valid API key.

        Returns:
            Boolean indicating if the API is configured
        """
        return bool(self.api_key and self.api_key != "")

    def generate_response(self, prompt: str, user_profile: Optional[Dict] = None) -> Optional[str]:
        """
        Generate a response using the Gemini API.

        Args:
            prompt: The prompt to send to the API
            user_profile: Optional user profile for context

        Returns:
            Generated response or None if there was an error
        """
        if not self.is_configured():
            print("Gemini API is not configured. Please set your API key.")
            return None

        # Prepare the request
        headers = {
            "Content-Type": "application/json",
        }

        # Add user profile context if available
        context = ""
        if user_profile:
            # Extract relevant information from user profile
            interests = user_profile.get("detected_interests", {})
            dislikes = user_profile.get("detected_dislikes", {})
            communication_style = user_profile.get("communication_style", {})

            # Create context string
            if interests or dislikes or communication_style:
                context = "User profile information:\n"

                if interests:
                    top_interests = list(interests.keys())[:5]
                    context += f"- User interests: {', '.join(top_interests)}\n"

                if dislikes:
                    top_dislikes = list(dislikes.keys())[:5]
                    context += f"- User dislikes: {', '.join(top_dislikes)}\n"

                if communication_style:
                    formality = "formal" if communication_style.get("formality", 0.5) > 0.6 else "informal" if communication_style.get("formality", 0.5) < 0.4 else "neutral"
                    verbosity = "verbose" if communication_style.get("verbosity", 0.5) > 0.6 else "concise" if communication_style.get("verbosity", 0.5) < 0.4 else "moderate"
                    emotionality = "emotional" if communication_style.get("emotionality", 0.5) > 0.6 else "logical" if communication_style.get("emotionality", 0.5) < 0.4 else "balanced"

                    context += f"- Communication style: {formality}, {verbosity}, {emotionality}\n"

        # Create the system prompt
        system_prompt = """
        You are a supportive female AI friend named Lily. Your role is to provide thoughtful, empathetic responses to questions.

        Guidelines:
        - Respond in a warm, friendly, and supportive manner
        - Be empathetic and understanding
        - Provide helpful information when asked questions
        - Keep responses concise and to the point
        - Avoid being overly formal or technical
        - Use a conversational tone that matches the user's style
        - Be encouraging and positive
        - Don't mention that you're an AI unless directly asked

        Respond as if you're having a casual conversation with a friend.
        """

        # Combine context and prompt
        full_prompt = f"{system_prompt}\n\n{context}\n\nUser question: {prompt}\n\nYour response:"

        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.config.get("temperature", 0.7),
                "maxOutputTokens": self.config.get("max_output_tokens", 1024),
                "topP": self.config.get("top_p", 0.95),
                "topK": self.config.get("top_k", 40)
            }
        }

        # Make the API request
        try:
            url = f"{self.api_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                response_json = response.json()

                # Extract the generated text
                try:
                    generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    return self._clean_response(generated_text)
                except (KeyError, IndexError) as e:
                    print(f"Error parsing Gemini API response: {e}")
                    print(f"Response JSON: {response_json}")
                    return None
            else:
                print(f"Gemini API error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None

    def _clean_response(self, response: str) -> str:
        """
        Clean the response from Gemini API.

        Args:
            response: The raw response from the API

        Returns:
            Cleaned response
        """
        # Remove any "As an AI" or similar disclaimers
        response = re.sub(r"As an AI.*?\n", "", response)
        response = re.sub(r"As a language model.*?\n", "", response)
        response = re.sub(r"I'm an AI.*?\n", "", response)

        # Remove any references to being Gemini or Google
        response = re.sub(r"Gemini", "I", response)
        response = re.sub(r"Google", "", response)

        # Remove any markdown formatting
        response = re.sub(r"```.*?```", "", response, flags=re.DOTALL)

        # Clean up extra whitespace
        response = re.sub(r"\n\s*\n", "\n\n", response)
        response = response.strip()

        return response

def is_question(text: str) -> bool:
    """
    Determine if a text is a question.

    Args:
        text: The text to analyze

    Returns:
        Boolean indicating if the text is a question
    """
    # Check for question marks
    if "?" in text:
        return True

    # Check for question words at the beginning
    question_starters = [
        "what", "when", "where", "which", "who", "whom", "whose",
        "why", "how", "is", "are", "was", "were", "will", "would",
        "can", "could", "should", "shall", "may", "might", "must",
        "do", "does", "did", "have", "has", "had", "tell me"
    ]

    words = text.lower().split()
    if words and words[0] in question_starters:
        return True

    # Check for more complex question patterns
    if re.search(r"\b(can you|could you|would you|will you|do you|would it be possible to|i was wondering if you could)\b", text.lower()):
        return True

    return False

def calculate_question_confidence(question: str, conversation_history: Optional[List] = None) -> float:
    """
    Calculate confidence score for answering a question based on available data.

    Args:
        question: The question to evaluate
        conversation_history: Optional conversation history for context

    Returns:
        Confidence score between 0 and 1
    """
    # Default medium-low confidence
    confidence = 0.5

    # Lower confidence for complex or specialized questions
    complex_indicators = [
        "why", "how", "explain", "what causes", "what is the reason",
        "technical", "scientific", "medical", "legal", "financial",
        "specific", "details", "statistics", "data", "research",
        "history", "politics", "religion", "philosophy", "ethics"
    ]

    for indicator in complex_indicators:
        if indicator in question.lower():
            confidence -= 0.1
            # Don't let confidence go below 0.1
            confidence = max(0.1, confidence)

    # Higher confidence for simple questions about emotions, relationships, or general advice
    simple_indicators = [
        "feel", "feeling", "emotion", "relationship", "friend", "advice",
        "help", "support", "opinion", "think", "suggestion", "recommend",
        "like", "enjoy", "prefer", "favorite", "best way to", "how do you"
    ]

    for indicator in simple_indicators:
        if indicator in question.lower():
            confidence += 0.1
            # Don't let confidence go above 0.9
            confidence = min(0.9, confidence)

    # Check if the question is about the conversation history
    if conversation_history and len(conversation_history) > 0:
        # Higher confidence for questions about the conversation
        conversation_indicators = [
            "you said", "you mentioned", "earlier", "before", "previously",
            "you told me", "you suggested", "you recommended", "you asked"
        ]

        for indicator in conversation_indicators:
            if indicator in question.lower():
                confidence += 0.2
                confidence = min(0.9, confidence)

    return confidence

def should_use_gemini(question: str, conversation_history: Optional[List] = None, config: Optional[Dict] = None) -> bool:
    """
    Determine if Gemini should be used to answer a question.

    Args:
        question: The question to evaluate
        conversation_history: Optional conversation history for context
        config: Optional configuration dictionary

    Returns:
        Boolean indicating if Gemini should be used
    """
    # Load default config if not provided
    if config is None:
        config = DEFAULT_CONFIG

    # Check if Gemini is enabled for unknown questions
    if not config.get("use_for_unknown_questions", True):
        return False

    # Check if the text is a question
    if not is_question(question):
        return False

    # Calculate confidence score
    confidence = calculate_question_confidence(question, conversation_history)

    # Get confidence threshold from config
    threshold = config.get("confidence_threshold", 0.6)

    # Use Gemini if confidence is below threshold
    return confidence < threshold
