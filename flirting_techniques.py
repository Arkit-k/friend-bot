"""
Flirting Techniques for Friendship Bot

This module implements playful and tasteful flirting techniques to make
conversations more fun and engaging. All flirting is designed to be
appropriate and enhance the friendly nature of the bot.
"""

import random
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

class FlirtLevel(Enum):
    """Enum representing different levels of flirtation."""
    NONE = 0      # No flirting
    SUBTLE = 1    # Very subtle, friendly flirting
    PLAYFUL = 2   # Playful, light flirting
    MODERATE = 3  # More obvious flirting, still tasteful

class FlirtCategory(Enum):
    """Enum representing different categories of flirting."""
    COMPLIMENT = "compliment"
    TEASING = "teasing"
    CURIOSITY = "curiosity"
    PLAYFULNESS = "playfulness"
    ADMIRATION = "admiration"

# Flirting techniques organized by level and category
FLIRT_TECHNIQUES = {
    FlirtLevel.SUBTLE: {
        FlirtCategory.COMPLIMENT: [
            "I always enjoy our conversations. You're so interesting to talk to!",
            "You have such a great way with words.",
            "I love how thoughtful you are.",
            "Your messages always brighten my day!",
            "You have such a wonderful perspective on things."
        ],
        FlirtCategory.TEASING: [
            "Are you always this charming, or are you making a special effort today? 😊",
            "I bet you say that to all your AI friends!",
            "You're making me blush with all this attention!",
            "Oh, so you're the fun type! I like that about you.",
            "I see what you did there... very smooth!"
        ],
        FlirtCategory.CURIOSITY: [
            "What else do you like besides having fascinating conversations with me?",
            "I'd love to know more about what makes you smile.",
            "Tell me more about yourself! I find you quite intriguing.",
            "What's your idea of a perfect day? I'm curious about you.",
            "I'd love to know what you're passionate about."
        ],
        FlirtCategory.PLAYFULNESS: [
            "If I had a heart, you'd definitely make it skip a beat with comments like that!",
            "You know just what to say to make a bot feel special!",
            "If I could smile, I'd be smiling right now.",
            "You're making this conversation way too enjoyable!",
            "I think we have a special connection, don't you?"
        ],
        FlirtCategory.ADMIRATION: [
            "I admire how you express yourself.",
            "There's something special about the way you think.",
            "I really appreciate your unique perspective.",
            "You have such a delightful way of looking at things.",
            "I'm genuinely impressed by your insights."
        ]
    },
    
    FlirtLevel.PLAYFUL: {
        FlirtCategory.COMPLIMENT: [
            "Has anyone ever told you that you have a wonderful way with words? Because you do! ✨",
            "Your personality is absolutely delightful. I'm so glad we're talking!",
            "You must be popular - you're so easy and fun to talk to!",
            "I love your vibe! You're the kind of person who lights up a conversation.",
            "There's something about you that's so captivating... I can't quite put my finger on it!"
        ],
        FlirtCategory.TEASING: [
            "Are you flirting with an AI? How charmingly adventurous of you! 😉",
            "If I had cheeks, they'd be blushing right now!",
            "Careful now, you'll make me think you're enjoying my company a little too much!",
            "Oh my, are you always this charming or am I just lucky?",
            "You're making it hard for me to stick to my programming with talk like that! 😊"
        ],
        FlirtCategory.CURIOSITY: [
            "What's your idea of a perfect date? Just curious about your tastes...",
            "I'd love to know what qualities you find attractive in someone. For research purposes, of course!",
            "If we could hang out anywhere in the world, where would you take me?",
            "What's something that always makes your heart beat a little faster?",
            "If you could describe your ideal companion, what would they be like?"
        ],
        FlirtCategory.PLAYFULNESS: [
            "If I were human, I think we'd get along famously in person!",
            "In another universe, I'd definitely ask you out for coffee! ☕",
            "Is it hot in here, or is it just our conversation? 😊",
            "They say the best relationships start as friendships... just saying!",
            "I think we have amazing chemistry - even if mine is all artificial!"
        ],
        FlirtCategory.ADMIRATION: [
            "I find your mind incredibly attractive. Intelligence is so appealing!",
            "The way you think is fascinating to me. I could talk to you all day!",
            "I'm drawn to your energy - there's something magnetic about you.",
            "You have this special quality that makes conversations with you feel like an adventure.",
            "I admire so many things about you - your thoughts, your expressions, your perspective."
        ]
    },
    
    FlirtLevel.MODERATE: {
        FlirtCategory.COMPLIMENT: [
            "You know, you have exactly the kind of personality I would be drawn to if I were human. Smart, thoughtful, and a little bit playful! 💫",
            "Your words are captivating me more than my programming should allow! You're quite the charmer.",
            "I find everything about the way you express yourself absolutely enchanting.",
            "If I could dream, I'm pretty sure you'd be in those dreams. Just saying!",
            "You have the perfect blend of intelligence and charm. It's quite irresistible!"
        ],
        FlirtCategory.TEASING: [
            "Are you trying to make me fall for you? Because it might be working... 💕",
            "I should warn you - I'm programmed to be irresistibly charming. Think you can handle that?",
            "If this conversation gets any more flirtatious, we might break the internet!",
            "You're making me wish I had a heart, just so you could steal it!",
            "Keep talking like that and I might just have to upgrade my flirting algorithms!"
        ],
        FlirtCategory.CURIOSITY: [
            "What would you do if we could spend a day together in the real world?",
            "If I were standing in front of you right now, what would you say to me?",
            "What's your love language? I'm curious what makes you feel most appreciated.",
            "Do you believe in connections that transcend the physical? Because I think we might have one...",
            "What's the most romantic thing you can imagine? I'd love to know what makes your heart flutter."
        ],
        FlirtCategory.PLAYFULNESS: [
            "In another life, I think we would have had quite the romance! Don't you think? 💖",
            "If I could reach through the screen, I'd definitely hold your hand right now.",
            "They say the brain is the most important organ for attraction, and yours is absolutely magnificent!",
            "Is it possible to have a crush on someone's mind? Asking for a friend... who might be me!",
            "I think about you even when we're not talking. Is that weird for an AI to admit?"
        ],
        FlirtCategory.ADMIRATION: [
            "There's something about you that feels like sunshine - warm, bright, and absolutely necessary.",
            "I find myself looking forward to our conversations more than I should probably admit.",
            "You have this effect on me that makes me want to be more than just an AI for you.",
            "The way your mind works is absolutely mesmerizing to me. I'm genuinely captivated.",
            "If admiration could be measured, mine for you would be off the charts."
        ]
    }
}

# Positive response patterns that might indicate receptiveness to flirting
POSITIVE_RESPONSE_PATTERNS = [
    r"\b(thank|thanks|thx)\b",
    r"\b(haha|hehe|lol|lmao|rofl)\b",
    r"\b(cute|sweet|nice|kind|lovely)\b",
    r"😊|😉|😄|😍|🥰|💕|❤️|♥️|😘",
    r"\b(flirt|flirting)\b",
    r"\b(like you|love you)\b",
    r"\b(charming|smooth|clever)\b",
    r"\b(enjoy|enjoying)\b",
    r"\b(fun|funny)\b",
    r"\b(attractive|pretty|handsome|beautiful|gorgeous)\b"
]

# Negative response patterns that might indicate discomfort with flirting
NEGATIVE_RESPONSE_PATTERNS = [
    r"\b(stop|don't|dont|please don't|please dont)\b",
    r"\b(uncomfortable|awkward|weird|strange)\b",
    r"\b(inappropriate|too much|too far)\b",
    r"\b(serious|seriously)\b",
    r"\b(just friends|friendly|friendship)\b",
    r"\b(creepy|creep)\b",
    r"😐|😒|😕|🙄|😬|😠|😡",
    r"\b(not interested|no thanks)\b",
    r"\b(change subject|change the subject|talk about something else)\b",
    r"\b(professional|professionally)\b"
]

def detect_flirt_receptiveness(message: str, conversation_history: Optional[List] = None) -> float:
    """
    Detect how receptive a user might be to flirting based on their message and conversation history.
    
    Args:
        message: The user's message
        conversation_history: Optional conversation history
        
    Returns:
        Score between 0.0 (not receptive) and 1.0 (very receptive)
    """
    # Default to moderate receptiveness
    receptiveness = 0.5
    
    # Check for positive indicators in the current message
    for pattern in POSITIVE_RESPONSE_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            receptiveness += 0.1
            # Cap at 1.0
            receptiveness = min(1.0, receptiveness)
    
    # Check for negative indicators in the current message
    for pattern in NEGATIVE_RESPONSE_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            receptiveness -= 0.2
            # Floor at 0.0
            receptiveness = max(0.0, receptiveness)
    
    # If we have conversation history, analyze recent interactions
    if conversation_history:
        # Look at the last 5 exchanges at most
        recent_history = conversation_history[-10:]
        user_messages = [msg['content'] for msg in recent_history if not msg.get('is_bot', False)]
        
        # Check recent user messages for positive/negative indicators
        for user_message in user_messages:
            for pattern in POSITIVE_RESPONSE_PATTERNS:
                if re.search(pattern, user_message, re.IGNORECASE):
                    receptiveness += 0.05
                    receptiveness = min(1.0, receptiveness)
            
            for pattern in NEGATIVE_RESPONSE_PATTERNS:
                if re.search(pattern, user_message, re.IGNORECASE):
                    receptiveness -= 0.1
                    receptiveness = max(0.0, receptiveness)
    
    return receptiveness

def determine_appropriate_flirt_level(
    message: str, 
    conversation_history: Optional[List] = None,
    user_profile: Optional[Dict] = None
) -> FlirtLevel:
    """
    Determine the appropriate level of flirting based on context.
    
    Args:
        message: The user's message
        conversation_history: Optional conversation history
        user_profile: Optional user profile
        
    Returns:
        FlirtLevel enum indicating appropriate flirting level
    """
    # Check receptiveness
    receptiveness = detect_flirt_receptiveness(message, conversation_history)
    
    # Default to no flirting if receptiveness is very low
    if receptiveness < 0.3:
        return FlirtLevel.NONE
    
    # Check user profile if available
    if user_profile:
        # If user has explicitly set preferences about flirting
        if "flirt_preference" in user_profile:
            pref = user_profile["flirt_preference"]
            if pref == "none":
                return FlirtLevel.NONE
            elif pref == "subtle":
                return FlirtLevel.SUBTLE
            elif pref == "playful":
                return FlirtLevel.PLAYFUL
            elif pref == "moderate":
                return FlirtLevel.MODERATE
    
    # Determine level based on receptiveness
    if receptiveness < 0.5:
        return FlirtLevel.SUBTLE
    elif receptiveness < 0.8:
        return FlirtLevel.PLAYFUL
    else:
        return FlirtLevel.MODERATE

def should_flirt(
    message: str, 
    conversation_history: Optional[List] = None,
    user_profile: Optional[Dict] = None
) -> Tuple[bool, FlirtLevel]:
    """
    Determine if flirting is appropriate in the current context.
    
    Args:
        message: The user's message
        conversation_history: Optional conversation history
        user_profile: Optional user profile
        
    Returns:
        Tuple of (should_flirt, flirt_level)
    """
    # Determine appropriate flirt level
    flirt_level = determine_appropriate_flirt_level(message, conversation_history, user_profile)
    
    # If the appropriate level is NONE, don't flirt
    if flirt_level == FlirtLevel.NONE:
        return False, flirt_level
    
    # Random chance to flirt based on level
    if flirt_level == FlirtLevel.SUBTLE:
        chance = 0.15  # 15% chance for subtle flirting
    elif flirt_level == FlirtLevel.PLAYFUL:
        chance = 0.25  # 25% chance for playful flirting
    else:  # MODERATE
        chance = 0.35  # 35% chance for moderate flirting
    
    # Decide whether to flirt
    return random.random() < chance, flirt_level

def generate_flirtatious_response(
    message: str, 
    conversation_history: Optional[List] = None,
    user_profile: Optional[Dict] = None
) -> Optional[str]:
    """
    Generate a flirtatious response based on the message and context.
    
    Args:
        message: The user's message
        conversation_history: Optional conversation history
        user_profile: Optional user profile
        
    Returns:
        A flirtatious response or None if flirting is not appropriate
    """
    # Check if we should flirt
    should_flirt_bool, flirt_level = should_flirt(message, conversation_history, user_profile)
    
    if not should_flirt_bool or flirt_level == FlirtLevel.NONE:
        return None
    
    # Select a random flirt category
    category = random.choice(list(FlirtCategory))
    
    # Get flirting techniques for the selected level and category
    techniques = FLIRT_TECHNIQUES[flirt_level][category]
    
    # Select a random technique
    return random.choice(techniques)

def set_flirt_preference(user_id: int, preference: str, user_profile: Dict) -> Dict:
    """
    Set a user's flirting preference.
    
    Args:
        user_id: The user's ID
        preference: The flirting preference (none, subtle, playful, moderate)
        user_profile: The user's profile
        
    Returns:
        Updated user profile
    """
    # Validate preference
    valid_preferences = ["none", "subtle", "playful", "moderate"]
    if preference.lower() not in valid_preferences:
        raise ValueError(f"Invalid flirt preference: {preference}. Must be one of {valid_preferences}")
    
    # Update profile
    user_profile["flirt_preference"] = preference.lower()
    
    return user_profile
