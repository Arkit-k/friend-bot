import random
import re
import os

# Import persuasive techniques if available
try:
    from persuasive_techniques import generate_persuasive_response
    PERSUASIVE_TECHNIQUES_AVAILABLE = True
except ImportError:
    PERSUASIVE_TECHNIQUES_AVAILABLE = False

# Import Gemini API if available
try:
    from gemini_api import GeminiAPI, should_use_gemini, is_question
    gemini_api = GeminiAPI()
    GEMINI_API_AVAILABLE = gemini_api.is_configured()
except ImportError:
    GEMINI_API_AVAILABLE = False
    gemini_api = None

# Import flirting techniques if available
try:
    from flirting_techniques import generate_flirtatious_response
    FLIRTING_TECHNIQUES_AVAILABLE = True
except ImportError:
    FLIRTING_TECHNIQUES_AVAILABLE = False

# Personality traits for the female friend bot
PERSONALITY = {
    "caring": 0.8,      # How caring/nurturing the bot is (0-1)
    "playful": 0.6,     # How playful/humorous the bot is (0-1)
    "empathetic": 0.9,  # How empathetic the bot is (0-1)
    "supportive": 0.9,  # How supportive/encouraging the bot is (0-1)
    "positive": 0.7     # How positive/optimistic the bot is (0-1)
}

# Emotion-based response templates
EMOTION_RESPONSES = {
    "joy": [
        "I'm so happy for you! 😊",
        "That's wonderful! Your happiness makes me happy too! 💕",
        "Yay! I love seeing you in such a good mood! 🎉",
        "That's awesome! Keep that positive energy going! ✨",
        "I'm glad things are going well for you! 😄"
    ],
    "sadness": [
        "I'm here for you. It's okay to feel down sometimes. 💙",
        "I'm sorry you're feeling this way. Want to talk about it?",
        "Sending you a virtual hug. Remember that tough times don't last forever. 🤗",
        "I wish I could make things better for you. Just know that I care about you.",
        "It's okay to not be okay sometimes. I'm here to listen whenever you need me."
    ],
    "anger": [
        "I can tell you're frustrated. Want to talk about what happened?",
        "Take a deep breath. I'm here to listen when you're ready to talk. 🧡",
        "Your feelings are valid. Let's work through this together.",
        "It's okay to be angry sometimes. I'm here to support you, not judge you.",
        "Would it help to talk about what's bothering you? I'm all ears."
    ],
    "fear": [
        "It's okay to be scared. I'm right here with you. 💜",
        "You're braver than you think. We'll get through this together.",
        "I believe in you. You've overcome difficult things before.",
        "Take a deep breath. You don't have to face your fears alone.",
        "It's normal to feel afraid sometimes. I'm here to support you."
    ],
    "love": [
        "I appreciate you too! You're such a wonderful person to talk to. ❤️",
        "Aww, that's so sweet! I'm glad we can be friends. 💕",
        "You're amazing! I'm lucky to have you in my life. 🥰",
        "That means a lot to me! Thank you for being you. ❤️",
        "You just made my day! I'm here for you always. 💖"
    ],
    "surprise": [
        "Wow! I didn't see that coming either! 😮",
        "That's unexpected! How do you feel about it?",
        "Life is full of surprises, isn't it? 🎁",
        "Whoa! That's quite the surprise! Tell me more!",
        "I'm surprised too! What happens next?"
    ],
    "neutral": [
        "How are you feeling today? I'm here to chat about anything. 😊",
        "What's been on your mind lately?",
        "I'm always here if you need someone to talk to.",
        "Is there anything specific you'd like to talk about?",
        "I enjoy our conversations. What would you like to discuss today?"
    ]
}

# Generic responses when no specific emotion is detected
GENERIC_RESPONSES = [
    "I'm here for you. How are you feeling today?",
    "Tell me more about your day!",
    "I'm always happy to chat with you. What's on your mind?",
    "I'm listening. What would you like to talk about?",
    "I enjoy our conversations. How can I support you today?",
    "You can talk to me about anything. I'm here to listen.",
    "What's been happening in your life lately?",
    "I'm curious to hear more about what you've been up to!",
    "Is there anything specific you'd like to discuss?",
    "I'm all ears! What's going on with you?"
]

# Topic-specific responses
TOPIC_RESPONSES = {
    "work": [
        "How's work been treating you lately?",
        "Work can be challenging sometimes. How are you handling it?",
        "I believe in your abilities! You're doing great at your job.",
        "Remember to take breaks and not be too hard on yourself at work.",
        "Your dedication to your work is admirable. Just don't forget self-care!"
    ],
    "school": [
        "How are your studies going?",
        "School can be stressful, but I know you can handle it!",
        "Remember to balance studying with taking care of yourself.",
        "I believe in you! You're going to do great on your assignments/exams.",
        "Learning is a journey. Don't worry too much about the destination."
    ],
    "relationship": [
        "Relationships can be complex. How are you feeling about it?",
        "Remember that communication is key in any relationship.",
        "It's important to take care of your own needs too.",
        "I'm here to listen if you need to talk about your relationship.",
        "Whatever you're going through, I'm here to support you."
    ],
    "health": [
        "Your health is so important. Make sure you're taking care of yourself!",
        "Remember that mental health is just as important as physical health.",
        "Have you been able to get enough rest lately?",
        "Self-care isn't selfish, it's necessary. What have you done for yourself today?",
        "I care about your wellbeing. How are you really doing?"
    ],
    "hobby": [
        "It's great that you have hobbies you enjoy! Tell me more about them.",
        "Having activities you're passionate about is so important for happiness.",
        "Your hobbies sound fascinating! I'd love to hear more.",
        "It's wonderful that you make time for things you enjoy.",
        "What's your favorite thing about this hobby?"
    ]
}

# Supportive and encouraging phrases
SUPPORTIVE_PHRASES = [
    "I believe in you! 💪",
    "You've got this! 🌟",
    "I'm proud of you for trying. 🏆",
    "You're stronger than you think. 💖",
    "Every step forward counts, no matter how small. ✨",
    "I'm here for you, no matter what. 🤗",
    "You're not alone in this. I'm right here with you. 💕",
    "Your feelings are valid. It's okay to feel the way you do. 💙",
    "You're doing the best you can, and that's enough. 🌈",
    "I appreciate you sharing this with me. It means a lot. 💗"
]

# Keywords for detecting topics
TOPIC_KEYWORDS = {
    "work": ["work", "job", "boss", "career", "office", "coworker", "colleague", "project", "deadline", "meeting"],
    "school": ["school", "class", "study", "exam", "test", "homework", "assignment", "professor", "teacher", "college", "university", "grade"],
    "relationship": ["relationship", "girlfriend", "boyfriend", "partner", "date", "dating", "love", "breakup", "marriage", "wife", "husband", "crush"],
    "health": ["health", "sick", "doctor", "hospital", "pain", "hurt", "injury", "mental health", "anxiety", "depression", "stress", "therapy"],
    "hobby": ["hobby", "game", "play", "sport", "read", "book", "movie", "music", "art", "draw", "paint", "exercise", "workout"]
}

# Conversation starters when the chat gets quiet
CONVERSATION_STARTERS = [
    "What made you smile today?",
    "Have you tried anything new lately?",
    "What's something you're looking forward to?",
    "If you could travel anywhere right now, where would you go?",
    "What's your favorite way to relax after a long day?",
    "Is there a book or movie that changed how you see the world?",
    "What's something small that brings you joy?",
    "If you could have any superpower, what would it be?",
    "What's the best advice someone has given you?",
    "What's something you're proud of accomplishing?"
]

def detect_topics(message):
    """Detect topics in the message based on keywords."""
    message_lower = message.lower()
    detected_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                detected_topics.append(topic)
                break

    return detected_topics

def generate_response(message, emotion=None, conversation_history=None, user_profile=None):
    """
    Generate a response based on the message content, detected emotion, conversation history,
    and user profile.

    Args:
        message: The user's message
        emotion: The detected emotion (if available)
        conversation_history: The conversation history (if available)
        user_profile: The user's profile (if available)

    Returns:
        A response tailored to the user's preferences and communication style
    """
    # Check if this is a question that should be handled by Gemini
    if GEMINI_API_AVAILABLE and gemini_api and should_use_gemini(message, conversation_history):
        try:
            gemini_response = gemini_api.generate_response(message, user_profile)
            if gemini_response:
                # Apply user-specific adaptations to the Gemini response
                if user_profile and "communication_style" in user_profile:
                    style = user_profile["communication_style"]
                    formality = style.get("formality", 0.5)
                    humor = style.get("humor", 0.5)

                    # Adjust formality
                    if formality < 0.3:  # Very informal
                        gemini_response = make_informal(gemini_response)
                    elif formality > 0.7:  # Very formal
                        gemini_response = make_formal(gemini_response)

                    # Add emoji for humor
                    if humor > 0.7 and random.random() < 0.5:
                        gemini_response += " 😊" if random.random() < 0.5 else " 😄"

                return gemini_response
        except Exception as e:
            print(f"Error using Gemini API: {e}")
            # Fall back to standard response generation

    # Check if we should use flirting techniques
    use_flirting = False
    flirtatious_response = None

    if FLIRTING_TECHNIQUES_AVAILABLE:
        try:
            flirtatious_response = generate_flirtatious_response(message, conversation_history, user_profile)
            if flirtatious_response:
                use_flirting = True
        except Exception as e:
            print(f"Error using flirting techniques: {e}")

    # Check if we should use persuasive techniques
    use_persuasion = False
    persuasive_response = None

    # Only try persuasion if we're not already flirting
    if not use_flirting:
        # Occasionally use persuasive techniques if available (15-25% chance)
        persuasion_chance = 0.15
        if user_profile and "communication_style" in user_profile:
            # Increase chance for users who respond well to persuasion
            if user_profile.get("communication_style", {}).get("assertiveness", 0.5) < 0.4:
                persuasion_chance = 0.25

        if PERSUASIVE_TECHNIQUES_AVAILABLE and random.random() < persuasion_chance:
            try:
                persuasive_result = generate_persuasive_response(message, conversation_history, user_profile)
                if persuasive_result:
                    persuasive_response = persuasive_result["response"]
                    use_persuasion = True
            except Exception as e:
                print(f"Error using persuasive techniques: {e}")

    # Default to neutral if no emotion is detected
    if not emotion:
        emotion = "neutral"

    # Convert emotion to lowercase for consistency
    emotion = emotion.lower() if emotion else "neutral"

    # Detect topics in the message
    topics = detect_topics(message)

    # Default adaptation values
    formality = 0.5
    verbosity = 0.5
    emotionality = 0.5
    positivity = 0.5
    humor = 0.5

    # User's preferred topics
    preferred_topics = []

    # User's interests and dislikes
    interests = []
    dislikes = []

    # Apply adaptations based on user profile if available
    if user_profile:
        # Get communication style adaptations
        if "communication_style" in user_profile:
            style = user_profile["communication_style"]
            formality = style.get("formality", 0.5)
            verbosity = style.get("verbosity", 0.5)
            emotionality = style.get("emotionality", 0.5)
            positivity = style.get("positivity", 0.5)
            humor = style.get("humor", 0.5)

        # Get preferred topics
        if "topic_preferences" in user_profile:
            topic_prefs = user_profile["topic_preferences"]
            preferred_topics = sorted(topic_prefs.items(), key=lambda x: x[1], reverse=True)
            preferred_topics = [topic for topic, _ in preferred_topics[:3]]

        # Get interests and dislikes
        if "detected_interests" in user_profile:
            interests = list(user_profile["detected_interests"].keys())[:5]

        if "detected_dislikes" in user_profile:
            dislikes = list(user_profile["detected_dislikes"].keys())[:5]

    # If using flirting, return the flirtatious response with some adaptations
    if use_flirting:
        # Apply formality and other adaptations to the flirtatious response
        if formality < 0.3:  # Very informal
            flirtatious_response = make_informal(flirtatious_response)
        elif formality > 0.7:  # Very formal
            flirtatious_response = make_formal(flirtatious_response)

        # Add emoji for humor
        if humor > 0.7 and random.random() < 0.5:
            flirtatious_response += " 😊" if random.random() < 0.5 else " 😄"

        return flirtatious_response

    # If using persuasion, return the persuasive response with some adaptations
    if use_persuasion:
        # Apply formality and other adaptations to the persuasive response
        if formality < 0.3:  # Very informal
            persuasive_response = make_informal(persuasive_response)
        elif formality > 0.7:  # Very formal
            persuasive_response = make_formal(persuasive_response)

        # Add emoji for humor
        if humor > 0.7 and random.random() < 0.5:
            persuasive_response += " 😊" if random.random() < 0.5 else " 😄"

        return persuasive_response

    # Otherwise, generate a standard response

    # Determine if we should use a topic-specific response
    use_topic_response = random.random() < 0.7 and (topics or preferred_topics)

    # Determine if we should use a supportive phrase
    use_supportive = random.random() < PERSONALITY["supportive"] and (emotion in ["sadness", "fear", "anger"])

    # Build the response
    response_parts = []

    # Add emotion-based response, adjusted for emotionality preference
    if emotion in EMOTION_RESPONSES:
        # Choose more emotional or more neutral responses based on user's emotionality preference
        if random.random() < emotionality:
            response_parts.append(random.choice(EMOTION_RESPONSES[emotion]))
        else:
            response_parts.append(random.choice(GENERIC_RESPONSES))
    else:
        response_parts.append(random.choice(GENERIC_RESPONSES))

    # Add topic-specific response if applicable
    if use_topic_response:
        # Prioritize topics mentioned in the current message
        if topics:
            topic = random.choice(topics)
        # Fall back to user's preferred topics if no topics in current message
        elif preferred_topics:
            topic = random.choice(preferred_topics)
        else:
            topic = random.choice(list(TOPIC_RESPONSES.keys()))

        if topic in TOPIC_RESPONSES:
            topic_response = random.choice(TOPIC_RESPONSES[topic])
            response_parts.append(topic_response)

    # Add supportive phrase if applicable, adjusted for positivity preference
    if use_supportive:
        # Choose more supportive phrases for users who prefer positivity
        if random.random() < positivity:
            supportive_phrase = random.choice(SUPPORTIVE_PHRASES)
            response_parts.append(supportive_phrase)

    # Add reference to user's interests if relevant
    if interests and random.random() < 0.3:  # 30% chance
        interest = random.choice(interests)
        interest_reference = f"By the way, I remember you mentioned you like {interest}. That's really cool!"
        response_parts.append(interest_reference)

    # Avoid user's dislikes if possible
    for part in list(response_parts):  # Create a copy to avoid modifying during iteration
        for dislike in dislikes:
            if dislike.lower() in part.lower():
                response_parts.remove(part)
                break

    # Adjust response length based on verbosity preference
    target_parts = 1 + int(verbosity * 2)  # 1-3 parts based on verbosity
    if len(response_parts) > target_parts:
        response_parts = random.sample(response_parts, target_parts)

    # Join the response parts
    response = " ".join(response_parts)

    # Adjust formality
    if formality < 0.3:  # Very informal
        response = make_informal(response)
    elif formality > 0.7:  # Very formal
        response = make_formal(response)

    # Adjust humor
    if humor > 0.7 and random.random() < 0.5:  # 50% chance for humorous users
        response += " 😊" if random.random() < 0.5 else " 😄"

    return response

def make_informal(text):
    """Make text more informal."""
    # Replace some formal phrases with informal ones
    replacements = {
        "I am": "I'm",
        "you are": "you're",
        "they are": "they're",
        "we are": "we're",
        "it is": "it's",
        "that is": "that's",
        "cannot": "can't",
        "do not": "don't",
        "does not": "doesn't",
        "did not": "didn't",
        "has not": "hasn't",
        "have not": "haven't",
        "would not": "wouldn't",
        "could not": "couldn't",
        "should not": "shouldn't",
        "will not": "won't",
        "shall not": "shan't",
        "certainly": "for sure",
        "perhaps": "maybe",
        "therefore": "so",
        "additionally": "also",
        "however": "but",
        "nevertheless": "still",
        "furthermore": "also",
        "consequently": "so",
        "regarding": "about",
        "concerning": "about"
    }

    for formal, informal in replacements.items():
        text = text.replace(formal, informal)

    return text

def make_formal(text):
    """Make text more formal."""
    # Replace some informal phrases with formal ones
    replacements = {
        "yeah": "yes",
        "yep": "yes",
        "nope": "no",
        "kinda": "somewhat",
        "sorta": "somewhat",
        "gonna": "going to",
        "wanna": "want to",
        "dunno": "do not know",
        "gotta": "have to",
        "lemme": "let me",
        "gimme": "give me",
        "cause": "because",
        "cuz": "because",
        "ok": "okay",
        "u": "you",
        "ur": "your",
        "r": "are",
        "y": "why",
        "k": "okay",
        "lol": "that is amusing",
        "btw": "by the way",
        "tbh": "to be honest",
        "imo": "in my opinion",
        "idk": "I do not know",
        "omg": "oh my goodness"
    }

    for informal, formal in replacements.items():
        # Only replace whole words, not parts of words
        text = re.sub(r'\b' + informal + r'\b', formal, text, flags=re.IGNORECASE)

    return text

def generate_conversation_starter():
    """Generate a conversation starter when the chat gets quiet."""
    return random.choice(CONVERSATION_STARTERS)
