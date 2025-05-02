"""
Configuration settings for the friendship bot.
"""

# Bot settings
BOT_NAME = "FriendshipBot"
BOT_PREFIX = "!"
BOT_DESCRIPTION = "A supportive AI friend designed to provide emotional support and companionship."

# Personality settings
PERSONALITY = {
    "name": "Lily",  # The bot's persona name
    "age": "21",     # The bot's persona age
    "gender": "female",
    "traits": {
        "caring": 0.8,      # How caring/nurturing the bot is (0-1)
        "playful": 0.6,     # How playful/humorous the bot is (0-1)
        "empathetic": 0.9,  # How empathetic the bot is (0-1)
        "supportive": 0.9,  # How supportive/encouraging the bot is (0-1)
        "positive": 0.7     # How positive/optimistic the bot is (0-1)
    }
}

# Response settings
TYPING_SPEED = 0.05  # Seconds per character for typing simulation
MIN_TYPING_TIME = 1.0  # Minimum typing time in seconds
MAX_TYPING_TIME = 3.0  # Maximum typing time in seconds

# Conversation memory settings
MAX_HISTORY = 20  # Maximum number of messages to store per user
EXPIRY_TIME = 86400  # Time in seconds after which messages expire (24 hours)

# Model settings
MODEL_PATH = "friendship_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 100  # Maximum sequence length for the model

# Discord settings
# Replace this with your actual Discord bot token
# IMPORTANT: Never commit your actual token to version control!
# This is just a placeholder - you should set this via environment variable
DISCORD_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"
