"""
Preference Learning Module for Friendship Bot

This module analyzes conversations to detect and learn user preferences,
particularly focusing on what male users like and dislike.
"""

import re
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any
import random

class PreferenceLearner:
    """
    A class to learn and track user preferences from conversations.
    Focuses on detecting likes, dislikes, interests, and values.
    """
    
    def __init__(self, data_dir="data/preferences"):
        """
        Initialize the preference learner.
        
        Args:
            data_dir: Directory to store preference data
        """
        self.data_dir = data_dir
        self.global_preferences = {
            "likes": {},      # Things users generally like
            "dislikes": {},   # Things users generally dislike
            "interests": {},  # Topics users are interested in
            "values": {}      # Things users value or find important
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing global preferences if available
        self.global_preferences_path = os.path.join(data_dir, "global_male_preferences.json")
        self._load_global_preferences()
        
        # Patterns for detecting preferences
        self.like_patterns = [
            r"i (?:really |kind of |kinda |absolutely |totally |){0,1}(?:like|love|enjoy|adore|am into) ([^.,!?;]+)",
            r"i'm (?:really |kind of |kinda |absolutely |totally |){0,1}(?:into|fond of|passionate about) ([^.,!?;]+)",
            r"([^.,!?;]+) is my favorite",
            r"i'm a fan of ([^.,!?;]+)",
            r"i'm (?:really |kind of |kinda |absolutely |totally |){0,1}interested in ([^.,!?;]+)"
        ]
        
        self.dislike_patterns = [
            r"i (?:really |kind of |kinda |absolutely |totally |){0,1}(?:dislike|hate|don't like|can't stand|detest) ([^.,!?;]+)",
            r"i'm not (?:really |kind of |kinda |){0,1}(?:into|fond of|a fan of) ([^.,!?;]+)",
            r"([^.,!?;]+) (?:annoys|bothers|irritates) me",
            r"i (?:can't|couldn't) (?:stand|tolerate|deal with) ([^.,!?;]+)",
            r"i'm tired of ([^.,!?;]+)"
        ]
        
        self.value_patterns = [
            r"i (?:really |kind of |kinda |absolutely |totally |){0,1}(?:value|appreciate|respect|admire) ([^.,!?;]+)",
            r"([^.,!?;]+) is important to me",
            r"i believe in ([^.,!?;]+)",
            r"i care about ([^.,!?;]+)",
            r"i'm looking for ([^.,!?;]+) in a (?:relationship|partner|friendship|person)"
        ]
        
        # Common stop words to filter out
        self.stop_words = {
            "a", "an", "the", "this", "that", "these", "those", "it", "they", "them",
            "when", "where", "why", "how", "what", "who", "whom", "which", "you", "your",
            "yours", "i", "me", "my", "mine", "we", "us", "our", "ours", "he", "him", "his",
            "she", "her", "hers", "its", "their", "theirs", "to", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after", "above",
            "below", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "will", "just", "should", "now"
        }
        
        # Categories of interests for classification
        self.interest_categories = {
            "sports": ["football", "basketball", "soccer", "baseball", "tennis", "golf", "hockey", "running", "swimming", "cycling", "fitness", "workout", "gym", "exercise", "sports"],
            "gaming": ["games", "gaming", "video games", "playstation", "xbox", "nintendo", "pc gaming", "steam", "esports", "rpg", "fps", "mmorpg", "minecraft", "fortnite", "call of duty", "league of legends"],
            "technology": ["tech", "technology", "computers", "programming", "coding", "software", "hardware", "gadgets", "ai", "artificial intelligence", "machine learning", "data science", "cybersecurity", "blockchain", "crypto"],
            "movies": ["movies", "films", "cinema", "hollywood", "directors", "actors", "netflix", "streaming", "tv shows", "series", "documentaries", "sci-fi", "action", "comedy", "drama", "horror"],
            "music": ["music", "songs", "bands", "artists", "concerts", "festivals", "albums", "rock", "pop", "hip hop", "rap", "jazz", "classical", "edm", "country", "metal", "indie"],
            "food": ["food", "cooking", "cuisine", "restaurants", "recipes", "baking", "grilling", "bbq", "beer", "wine", "cocktails", "coffee", "pizza", "burgers", "steak"],
            "travel": ["travel", "vacation", "trips", "tourism", "countries", "cities", "beaches", "mountains", "hiking", "camping", "backpacking", "road trips", "flying", "hotels", "airbnb"],
            "reading": ["books", "reading", "literature", "fiction", "non-fiction", "novels", "authors", "poetry", "comics", "manga", "fantasy", "sci-fi", "mystery", "thriller"],
            "career": ["career", "job", "work", "business", "entrepreneurship", "startups", "investing", "finance", "stocks", "real estate", "marketing", "management", "leadership"],
            "cars": ["cars", "vehicles", "driving", "racing", "automotive", "mechanics", "motorcycles", "trucks", "jeeps", "tesla", "ford", "toyota", "bmw", "audi", "ferrari"],
            "outdoors": ["outdoors", "nature", "hiking", "camping", "fishing", "hunting", "kayaking", "canoeing", "climbing", "mountaineering", "skiing", "snowboarding", "surfing"],
            "fitness": ["fitness", "workout", "gym", "exercise", "running", "weightlifting", "bodybuilding", "crossfit", "yoga", "martial arts", "boxing", "mma"],
            "art": ["art", "drawing", "painting", "sculpture", "photography", "design", "architecture", "crafts", "creativity", "museums", "galleries"],
            "science": ["science", "physics", "chemistry", "biology", "astronomy", "space", "environment", "climate", "research", "discoveries", "experiments"],
            "history": ["history", "historical", "ancient", "medieval", "world war", "civilizations", "archaeology", "mythology", "legends"],
            "politics": ["politics", "government", "democracy", "elections", "political", "policies", "laws", "rights", "activism", "social issues"],
            "relationships": ["relationships", "dating", "marriage", "family", "parenting", "children", "love", "romance", "sex", "intimacy", "connection"],
            "personal_development": ["self-improvement", "personal growth", "productivity", "motivation", "goals", "habits", "mindfulness", "meditation", "psychology", "mental health"],
            "fashion": ["fashion", "style", "clothing", "shoes", "accessories", "watches", "grooming", "haircuts", "beards", "tattoos"],
            "humor": ["humor", "comedy", "jokes", "memes", "funny", "stand-up", "sitcoms", "satire", "pranks"]
        }
    
    def _load_global_preferences(self):
        """Load global preferences from file if it exists."""
        if os.path.exists(self.global_preferences_path):
            try:
                with open(self.global_preferences_path, 'r', encoding='utf-8') as f:
                    self.global_preferences = json.load(f)
                print(f"Loaded global preferences with {sum(len(prefs) for prefs in self.global_preferences.values())} items")
            except Exception as e:
                print(f"Error loading global preferences: {e}")
    
    def _save_global_preferences(self):
        """Save global preferences to file."""
        try:
            with open(self.global_preferences_path, 'w', encoding='utf-8') as f:
                json.dump(self.global_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving global preferences: {e}")
    
    def _clean_preference(self, preference: str) -> str:
        """
        Clean and normalize a preference string.
        
        Args:
            preference: The raw preference string
            
        Returns:
            Cleaned preference string
        """
        # Convert to lowercase
        preference = preference.lower().strip()
        
        # Remove common articles and pronouns at the beginning
        for word in ["a ", "an ", "the ", "this ", "that ", "these ", "those ", "my ", "your ", "his ", "her ", "their "]:
            if preference.startswith(word):
                preference = preference[len(word):]
        
        # Remove trailing prepositions and conjunctions
        for word in [" and", " or", " but", " so", " because", " with", " without", " by", " for", " to", " from"]:
            if preference.endswith(word):
                preference = preference[:-len(word)]
        
        return preference.strip()
    
    def _is_valid_preference(self, preference: str) -> bool:
        """
        Check if a preference is valid (not just a stop word or too short).
        
        Args:
            preference: The preference to check
            
        Returns:
            Boolean indicating if the preference is valid
        """
        # Check if it's just a stop word
        if preference in self.stop_words:
            return False
        
        # Check if it's too short
        if len(preference) < 3:
            return False
        
        # Check if it's just a pronoun or common word
        words = preference.split()
        if len(words) == 1 and words[0] in self.stop_words:
            return False
        
        return True
    
    def _categorize_interest(self, interest: str) -> str:
        """
        Categorize an interest into a predefined category.
        
        Args:
            interest: The interest to categorize
            
        Returns:
            Category name or "other" if no category matches
        """
        interest_lower = interest.lower()
        
        # Check each category
        for category, keywords in self.interest_categories.items():
            for keyword in keywords:
                if keyword in interest_lower:
                    return category
        
        return "other"
    
    def analyze_message(self, message: str, is_male: bool = True) -> Dict[str, List[str]]:
        """
        Analyze a message to detect preferences.
        
        Args:
            message: The message to analyze
            is_male: Whether the message is from a male user
            
        Returns:
            Dictionary of detected preferences
        """
        if not is_male:
            # If not tracking female preferences, return empty results
            return {"likes": [], "dislikes": [], "values": []}
        
        # Initialize results
        results = {
            "likes": [],
            "dislikes": [],
            "values": []
        }
        
        # Check for likes
        for pattern in self.like_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                preference = self._clean_preference(match.group(1))
                if self._is_valid_preference(preference):
                    results["likes"].append(preference)
                    # Also categorize as an interest
                    category = self._categorize_interest(preference)
                    if category != "other":
                        if preference not in results.get("interests", []):
                            if "interests" not in results:
                                results["interests"] = []
                            results["interests"].append(preference)
        
        # Check for dislikes
        for pattern in self.dislike_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                preference = self._clean_preference(match.group(1))
                if self._is_valid_preference(preference):
                    results["dislikes"].append(preference)
        
        # Check for values
        for pattern in self.value_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                preference = self._clean_preference(match.group(1))
                if self._is_valid_preference(preference):
                    results["values"].append(preference)
        
        return results
    
    def update_global_preferences(self, preferences: Dict[str, List[str]]):
        """
        Update global preferences with new detected preferences.
        
        Args:
            preferences: Dictionary of detected preferences
        """
        for category, items in preferences.items():
            if category not in self.global_preferences:
                self.global_preferences[category] = {}
            
            for item in items:
                if item in self.global_preferences[category]:
                    self.global_preferences[category][item] += 1
                else:
                    self.global_preferences[category][item] = 1
        
        # Save updated preferences
        self._save_global_preferences()
    
    def get_top_preferences(self, category: str, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top n preferences in a category.
        
        Args:
            category: The preference category (likes, dislikes, interests, values)
            n: Number of top preferences to return
            
        Returns:
            List of (preference, count) tuples
        """
        if category not in self.global_preferences:
            return []
        
        # Sort by count (descending)
        sorted_prefs = sorted(
            self.global_preferences[category].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_prefs[:n]
    
    def get_random_preference(self, category: str) -> Optional[str]:
        """
        Get a random preference from a category, weighted by count.
        
        Args:
            category: The preference category (likes, dislikes, interests, values)
            
        Returns:
            A random preference or None if the category is empty
        """
        if category not in self.global_preferences or not self.global_preferences[category]:
            return None
        
        # Get all preferences and their counts
        prefs = list(self.global_preferences[category].items())
        
        # Extract preferences and counts for weighted random selection
        items = [p[0] for p in prefs]
        weights = [p[1] for p in prefs]
        
        # Weighted random selection
        return random.choices(items, weights=weights, k=1)[0]
    
    def get_related_preferences(self, preference: str, category: str = "likes", n: int = 5) -> List[str]:
        """
        Get preferences related to a given preference.
        
        Args:
            preference: The preference to find related items for
            category: The preference category to search in
            n: Number of related preferences to return
            
        Returns:
            List of related preferences
        """
        if category not in self.global_preferences or not self.global_preferences[category]:
            return []
        
        # Get the category of the preference
        pref_category = self._categorize_interest(preference)
        
        # If it's a recognized category, find other preferences in the same category
        if pref_category != "other":
            related = []
            for pref, count in self.global_preferences[category].items():
                if self._categorize_interest(pref) == pref_category and pref != preference:
                    related.append((pref, count))
            
            # Sort by count (descending)
            related.sort(key=lambda x: x[1], reverse=True)
            
            return [r[0] for r in related[:n]]
        
        # If no category match, return random top preferences
        return [p[0] for p in self.get_top_preferences(category, n) if p[0] != preference]
    
    def process_conversation(self, messages: List[Dict], user_gender: str = "male"):
        """
        Process a conversation to learn preferences.
        
        Args:
            messages: List of message dictionaries with 'content' and 'user_id' keys
            user_gender: Gender of the user (male/female)
        """
        is_male = user_gender.lower() == "male"
        
        for message in messages:
            # Skip bot messages
            if message.get("is_bot", False):
                continue
            
            # Analyze the message
            preferences = self.analyze_message(message["content"], is_male=is_male)
            
            # Update global preferences
            if preferences and any(preferences.values()):
                self.update_global_preferences(preferences)
    
    def generate_preference_based_response(self, message: str, user_profile: Optional[Dict] = None) -> Optional[str]:
        """
        Generate a response based on detected preferences in the message.
        
        Args:
            message: The user's message
            user_profile: Optional user profile with known preferences
            
        Returns:
            A preference-based response or None if no preferences detected
        """
        # Analyze the message for preferences
        detected_prefs = self.analyze_message(message)
        
        # If no preferences detected in the message, check if we can use the user profile
        if not any(detected_prefs.values()) and user_profile:
            if "detected_interests" in user_profile:
                # Use a random interest from the user profile
                interests = list(user_profile["detected_interests"].keys())
                if interests:
                    interest = random.choice(interests)
                    related = self.get_related_preferences(interest, "likes", 3)
                    
                    if related:
                        related_item = random.choice(related)
                        responses = [
                            f"Since you like {interest}, you might also enjoy {related_item}.",
                            f"Many guys who are into {interest} also like {related_item}. Have you tried it?",
                            f"I've noticed that {interest} fans often appreciate {related_item} as well.",
                            f"If you enjoy {interest}, {related_item} might be right up your alley too."
                        ]
                        return random.choice(responses)
            
            return None
        
        # If we detected likes in the message
        if detected_prefs["likes"]:
            like = random.choice(detected_prefs["likes"])
            related = self.get_related_preferences(like, "likes", 3)
            
            if related:
                related_item = random.choice(related)
                responses = [
                    f"I've noticed a lot of guys who like {like} also enjoy {related_item}.",
                    f"If you're into {like}, you might also appreciate {related_item}.",
                    f"{like} is popular! Have you ever tried {related_item}? Many guys enjoy both.",
                    f"That's cool that you like {like}! {related_item} is often enjoyed by people with similar tastes."
                ]
                return random.choice(responses)
        
        # If we detected values in the message
        elif detected_prefs["values"]:
            value = random.choice(detected_prefs["values"])
            responses = [
                f"I really respect that you value {value}. That says a lot about you.",
                f"It's great that you appreciate {value}. That's something many thoughtful people care about.",
                f"{value} is definitely important. It's nice to meet someone who recognizes that.",
                f"The fact that you value {value} shows you have depth. I like that."
            ]
            return random.choice(responses)
        
        return None

# Example usage
if __name__ == "__main__":
    # Initialize preference learner
    preference_learner = PreferenceLearner()
    
    # Example message
    message = "I really like playing basketball and video games. I enjoy watching sci-fi movies and I'm into technology. I can't stand waiting in long lines and I hate when people are rude."
    
    # Analyze message
    preferences = preference_learner.analyze_message(message)
    print("Detected preferences:")
    for category, items in preferences.items():
        if items:
            print(f"  {category.capitalize()}: {', '.join(items)}")
    
    # Update global preferences
    preference_learner.update_global_preferences(preferences)
    
    # Get top preferences
    print("\nTop likes:")
    for pref, count in preference_learner.get_top_preferences("likes", 5):
        print(f"  {pref}: {count}")
    
    # Generate a preference-based response
    response = preference_learner.generate_preference_based_response(message)
    if response:
        print(f"\nResponse: {response}")
