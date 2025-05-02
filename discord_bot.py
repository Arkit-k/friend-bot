import os
import discord
from discord.ext import commands
import asyncio
import json
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from response_generator import generate_response
from conversation_memory import ConversationMemory
from conversation_state import ConversationStateManager, ConversationStatus

# Import Gemini API if available
try:
    from gemini_api import GeminiAPI
    gemini_api = GeminiAPI()
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    gemini_api = None

# Import preference learning module
try:
    from preference_learning import PreferenceLearner
    preference_learner = PreferenceLearner()
    PREFERENCE_LEARNING_AVAILABLE = True
    print("Preference learning module loaded successfully!")
except ImportError as e:
    PREFERENCE_LEARNING_AVAILABLE = False
    preference_learner = None
    print(f"Preference learning module not available: {e}")
    print("The bot will run without preference learning capabilities.")
    print("To enable preference learning, make sure preference_learning.py is in the same directory.")
except Exception as e:
    PREFERENCE_LEARNING_AVAILABLE = False
    preference_learner = None
    print(f"Error initializing preference learning module: {e}")
    print("The bot will run without preference learning capabilities.")

# Bot configuration
try:
    # Try with privileged intents first
    intents = discord.Intents.default()
    intents.message_content = True  # This is a privileged intent
    intents.members = True  # This is a privileged intent
    print("Using privileged intents (message_content and members)")
except Exception as e:
    # Fall back to default intents if there's an error
    print(f"Error setting up privileged intents: {e}")
    print("Falling back to default intents")
    intents = discord.Intents.default()

bot = commands.Bot(command_prefix='!', intents=intents)

# Load the model, tokenizer, and label encoder
model_path = "friendship_model.h5"
tokenizer_path = "tokenizer.pkl"
label_encoder_path = "label_encoder.pkl"

# Initialize conversation memory and state manager
conversation_memory = ConversationMemory(max_history=10)
conversation_state = ConversationStateManager()

# Load ML model and related files
try:
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    print("Model, tokenizer, and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")
    print("The bot will run without emotion detection capabilities.")
    model = None
    tokenizer = None
    label_encoder = None

@bot.event
async def on_ready():
    """Event handler for when the bot is ready and connected to Discord."""
    print(f'{bot.user.name} has connected to Discord!')
    print(f'Bot is connected to {len(bot.guilds)} servers')

    # Set custom status
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.listening,
        name="your thoughts 💭"
    ))

@bot.event
async def on_message(message):
    """Event handler for when a message is received."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands if the message starts with the command prefix
    await bot.process_commands(message)

    # Only respond in DMs or when mentioned in a server
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    if is_dm or is_mentioned:
        # Check if this user can interact with the bot in this channel
        can_interact, current_user_id = conversation_state.can_interrupt(message.author.id, message.channel.id)

        # If another user has a protected conversation in this channel, don't respond
        if not can_interact:
            # Only respond with a notice in servers, not DMs (DMs can't be interrupted)
            if not is_dm:
                # Get the username of the user with the protected conversation
                try:
                    current_user = await bot.fetch_user(current_user_id)
                    current_username = current_user.display_name
                except:
                    current_username = "someone else"

                # Send a private message to the user who tried to interrupt
                try:
                    await message.author.send(f"I'm currently in a focused conversation with {current_username} in that channel. Please wait until we're finished, or you can chat with me here in DMs!")
                except:
                    # If we can't send a DM, reply in the channel but mention the user
                    await message.channel.send(f"{message.author.mention} I'm currently in a focused conversation with {current_username}. Please wait until we're finished, or you can chat with me in DMs!")

            return

        # Remove the mention from the message if present
        content = message.content
        if is_mentioned:
            content = content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()

        # Update conversation state
        conversation_state.update_activity(message.author.id, message.channel.id)

        # Add message to conversation memory and update user profile
        conversation_memory.add_message(message.author.id, content, is_bot=False)

        # Get conversation history for context
        history = conversation_memory.get_history(message.author.id)

        # Process message with preference learner if available
        if PREFERENCE_LEARNING_AVAILABLE and preference_learner:
            # Assume male user for now - in a real app, you might want to detect or ask for gender
            user_gender = "male"

            # Process the conversation to learn preferences
            preference_learner.process_conversation([{"content": content, "user_id": message.author.id}], user_gender)

            # Try to generate a preference-based response
            preference_response = preference_learner.generate_preference_based_response(content)

            # If we got a preference-based response, use it occasionally (30% chance)
            if preference_response and random.random() < 0.3:
                response = preference_response

                # Add typing indicator for realism
                async with message.channel.typing():
                    # Add a slight delay for realism
                    await asyncio.sleep(1.5)

                # Send the response
                await message.channel.send(response)

                # Add bot's response to conversation memory
                conversation_memory.add_message(bot.user.id, response, is_bot=True)

                # Skip the regular response generation
                return

        # Detect emotion if model is available
        emotion = None
        if model and tokenizer and label_encoder:
            try:
                # Tokenize and pad the message
                sequence = tokenizer.texts_to_sequences([content])
                padded_sequence = pad_sequences(sequence, maxlen=100)

                # Make prediction
                prediction = model.predict(padded_sequence)
                emotion = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
                print(f"Detected emotion: {emotion}")
            except Exception as e:
                print(f"Error detecting emotion: {e}")

        # Get user profile and adaptation recommendations
        user_profile = conversation_memory.get_adaptation_recommendations(message.author.id)

        # Generate response based on message content, emotion, conversation history, and user profile
        response = generate_response(content, emotion, history, user_profile)

        # Add typing indicator for realism
        async with message.channel.typing():
            # Adjust typing delay based on response length and user's verbosity preference
            base_delay = 0.05 * len(response)
            if user_profile and "response_style" in user_profile:
                # Faster typing for users who prefer concise responses
                response_length_pref = user_profile["response_style"].get("response_length", 0.5)
                typing_delay = min(2.0, base_delay * (0.5 + response_length_pref))
            else:
                typing_delay = min(2.0, base_delay)

            await asyncio.sleep(typing_delay)

        # Send the response
        await message.channel.send(response)

        # Add bot's response to conversation memory
        conversation_memory.add_message(bot.user.id, response, is_bot=True)

@bot.command(name='friendhelp')
async def friend_help_command(ctx):
    """Command to show help information."""
    help_text = """
    **🌸 Friendship Bot Help 🌸**

    I'm here to be your supportive friend! You can:

    • Talk to me in DMs for private conversations
    • Mention me in a server to chat with me there
    • Use these commands:
      - `!friendhelp` - Show this help message
      - `!reset` - Reset our conversation history
      - `!mood` - Check how I think you're feeling
      - `!profile` - View your communication profile
      - `!preference` - Set your preferences (e.g., `!preference formality formal`)
      - `!like` - Tell me something you like (e.g., `!like chocolate`)
      - `!dislike` - Tell me something you dislike (e.g., `!dislike spicy food`)
      - `!focus` - Start a focused conversation where I won't respond to others
      - `!unfocus` - End a focused conversation
    """

    # Add Gemini API commands if available
    if GEMINI_API_AVAILABLE and gemini_api:
        gemini_text = """
      - `!gemini` - Configure Gemini API for answering questions I don't know about
        """
        help_text += gemini_text

    # Add preference learning commands if available
    if PREFERENCE_LEARNING_AVAILABLE and preference_learner:
        pref_text = """
      - `!malepreferences` - View what I've learned about male preferences
        """
        help_text += pref_text

    help_text += """
    I'll do my best to be a good friend and support you! 💕
    """

    await ctx.send(help_text)

@bot.command(name='reset')
async def reset_command(ctx):
    """Command to reset conversation history."""
    conversation_memory.clear_history(ctx.author.id)
    await ctx.send("I've reset our conversation. Let's start fresh! 😊")

@bot.command(name='mood')
async def mood_command(ctx):
    """Command to check the user's detected mood."""
    if not model or not tokenizer or not label_encoder:
        await ctx.send("Sorry, I can't detect emotions right now. My emotion model isn't loaded.")
        return

    # Get recent messages from the user
    history = conversation_memory.get_history(ctx.author.id)
    if not history:
        await ctx.send("I don't have enough of our conversation to analyze your mood yet. Chat with me a bit more!")
        return

    # Get the most recent user message
    user_messages = [msg for msg in history if msg['user_id'] == ctx.author.id]
    if not user_messages:
        await ctx.send("I don't have any messages from you to analyze. Chat with me a bit more!")
        return

    recent_message = user_messages[-1]['content']

    try:
        # Tokenize and pad the message
        sequence = tokenizer.texts_to_sequences([recent_message])
        padded_sequence = pad_sequences(sequence, maxlen=100)

        # Make prediction
        prediction = model.predict(padded_sequence)
        emotion = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

        # Send response based on detected emotion
        emotion_responses = {
            "joy": "You seem happy! That makes me happy too! 😊",
            "sadness": "I'm sensing you might be feeling a bit down. I'm here for you if you need to talk. 💙",
            "anger": "You seem frustrated. Want to talk about what's bothering you? I'm here to listen. 🧡",
            "fear": "I sense you might be worried about something. Remember, you're stronger than you think. 💪",
            "love": "I'm feeling the love! Thanks for being so kind and warm. ❤️",
            "surprise": "You seem surprised! Did something unexpected happen? 😮",
        }

        response = emotion_responses.get(emotion.lower(), f"I think you might be feeling {emotion}. How are you really doing?")
        await ctx.send(response)
    except Exception as e:
        print(f"Error in mood detection: {e}")
        await ctx.send("I'm having trouble reading your mood right now. Let's just keep chatting!")

@bot.command(name='profile')
async def profile_command(ctx):
    """Command to view the user's communication profile."""
    # Get the user's profile
    profile = conversation_memory.get_user_profile(ctx.author.id)

    # Get adaptation recommendations (this enhances the profile with recommendations)
    profile = conversation_memory.get_adaptation_recommendations(ctx.author.id)

    # Create a readable summary
    communication_style = profile.get("communication_style", {})
    topic_preferences = profile.get("topic_preferences", {})
    detected_interests = list(profile.get("detected_interests", {}).keys())[:5]
    detected_dislikes = list(profile.get("detected_dislikes", {}).keys())[:5]

    # Format the communication style as percentages
    style_summary = "\n".join([
        f"• **{style.capitalize()}**: {int(value * 100)}%"
        for style, value in communication_style.items()
    ])

    # Format the topic preferences as a sorted list
    sorted_topics = sorted(topic_preferences.items(), key=lambda x: x[1], reverse=True)
    topic_summary = "\n".join([
        f"• **{topic.capitalize()}**: {int(value * 100)}%"
        for topic, value in sorted_topics[:3]
    ])

    # Format interests and dislikes
    interests_summary = ", ".join(detected_interests) if detected_interests else "None detected yet"
    dislikes_summary = ", ".join(detected_dislikes) if detected_dislikes else "None detected yet"

    profile_text = f"""
    **🌸 Your Communication Profile 🌸**

    Based on our conversations, here's what I've learned about your communication style:

    **Communication Style:**
    {style_summary}

    **Preferred Topics:**
    {topic_summary}

    **Detected Interests:** {interests_summary}

    **Detected Dislikes:** {dislikes_summary}

    I use this information to adapt my responses to better match your preferences. You can use the `!preference` command to explicitly set your preferences.
    """

    await ctx.send(profile_text)

@bot.command(name='preference')
async def preference_command(ctx, category=None, value=None):
    """Command to set user preferences."""
    if not category or not value:
        await ctx.send("""
        Please specify a category and value. For example:
        `!preference formality formal` or `!preference formality informal`

        Available categories:
        • formality (formal/informal)
        • verbosity (verbose/concise)
        • emotionality (emotional/logical)
        • positivity (positive/negative)
        • humor (humorous/serious)
        • flirting (none/subtle/playful/moderate)
        """)
        return

    category = category.lower()
    value = value.lower()

    # Special handling for flirting preference
    if category == "flirting":
        valid_values = ["none", "subtle", "playful", "moderate"]
        if value not in valid_values:
            await ctx.send(f"Sorry, '{value}' is not a valid value for 'flirting'. Valid values are: {', '.join(valid_values)}")
            return

        # Get the user's profile
        profile = conversation_memory.get_user_profile(ctx.author.id)

        # Set flirting preference
        profile["flirt_preference"] = value
        conversation_memory.user_profiler.save_profile(ctx.author.id, profile)

        # Send confirmation
        if value == "none":
            await ctx.send("I've turned off flirting in our conversations. I'll keep things friendly but not flirtatious.")
        elif value == "subtle":
            await ctx.send("I'll occasionally use very subtle flirting in our conversations. Let me know if you'd like to adjust this!")
        elif value == "playful":
            await ctx.send("I'll use playful flirting from time to time to make our conversations more fun. Let me know if you'd like to adjust this!")
        elif value == "moderate":
            await ctx.send("I'll use moderate flirting to make our conversations more engaging. Let me know if you'd like to adjust this!")

        return

    # Map text values to numeric values (0-1)
    value_mappings = {
        "formality": {"formal": 0.8, "neutral": 0.5, "informal": 0.2},
        "verbosity": {"verbose": 0.8, "neutral": 0.5, "concise": 0.2},
        "emotionality": {"emotional": 0.8, "neutral": 0.5, "logical": 0.2},
        "positivity": {"positive": 0.8, "neutral": 0.5, "negative": 0.2},
        "humor": {"humorous": 0.8, "neutral": 0.5, "serious": 0.2}
    }

    if category not in value_mappings:
        await ctx.send(f"Sorry, '{category}' is not a valid preference category. Use `!preference` to see available categories.")
        return

    if value not in value_mappings[category]:
        await ctx.send(f"Sorry, '{value}' is not a valid value for '{category}'. Valid values are: {', '.join(value_mappings[category].keys())}")
        return

    # Set the preference
    numeric_value = value_mappings[category][value]
    conversation_memory.set_user_preference(ctx.author.id, "communication_style", category, numeric_value)

    await ctx.send(f"I've set your {category} preference to {value}. I'll adapt my responses accordingly! 😊")

@bot.command(name='like')
async def like_command(ctx, *, interest=None):
    """Command to explicitly tell the bot about something the user likes."""
    if not interest:
        await ctx.send("Please tell me something you like. For example: `!like chocolate`")
        return

    # Add the interest to the user's profile
    profile = conversation_memory.get_user_profile(ctx.author.id)

    if "detected_interests" not in profile:
        profile["detected_interests"] = {}

    profile["detected_interests"][interest.lower()] = profile["detected_interests"].get(interest.lower(), 0) + 5
    conversation_memory.user_profiler.save_profile(ctx.author.id, profile)

    responses = [
        f"Thanks for letting me know you like {interest}! I'll remember that. 😊",
        f"I'm glad you enjoy {interest}! I'll keep that in mind for our future conversations.",
        f"{interest} sounds great! I've added that to your interests.",
        f"I'll remember that you like {interest}. Thanks for sharing that with me!"
    ]

    await ctx.send(random.choice(responses))

@bot.command(name='dislike')
async def dislike_command(ctx, *, dislike=None):
    """Command to explicitly tell the bot about something the user dislikes."""
    if not dislike:
        await ctx.send("Please tell me something you dislike. For example: `!dislike spicy food`")
        return

    # Add the dislike to the user's profile
    profile = conversation_memory.get_user_profile(ctx.author.id)

    if "detected_dislikes" not in profile:
        profile["detected_dislikes"] = {}

    profile["detected_dislikes"][dislike.lower()] = profile["detected_dislikes"].get(dislike.lower(), 0) + 5
    conversation_memory.user_profiler.save_profile(ctx.author.id, profile)

    responses = [
        f"I understand that you don't like {dislike}. I'll remember that. 😊",
        f"Thanks for letting me know you dislike {dislike}. I'll keep that in mind.",
        f"I'll make a note that you're not a fan of {dislike}.",
        f"I'll remember that you dislike {dislike}. Thanks for sharing that with me!"
    ]

    await ctx.send(random.choice(responses))

@bot.command(name='focus')
async def focus_command(ctx):
    """Command to start a protected conversation that can't be interrupted."""
    # Check if we're in a server (not a DM)
    is_dm = isinstance(ctx.channel, discord.DMChannel)
    if is_dm:
        await ctx.send("You're already in a private conversation with me in DMs! No one else can interrupt us here. 😊")
        return

    # Start a protected conversation
    success = conversation_state.start_protected_conversation(ctx.author.id, ctx.channel.id)

    if success:
        await ctx.send(f"🔒 I'm now in a focused conversation with you, {ctx.author.mention}! I won't respond to others in this channel until we're done or the conversation times out after inactivity. Use `!unfocus` when you're finished.")
    else:
        # Check if the user already has a protected conversation
        state = conversation_state.get_state(ctx.author.id)
        if state["status"] == ConversationStatus.PROTECTED.value:
            await ctx.send("We're already in a focused conversation! Use `!unfocus` when you're finished.")
        else:
            await ctx.send("I couldn't start a focused conversation right now. Please try again later.")

@bot.command(name='unfocus')
async def unfocus_command(ctx):
    """Command to end a protected conversation."""
    # End the protected conversation
    success = conversation_state.end_protected_conversation(ctx.author.id)

    if success:
        await ctx.send(f"🔓 Our focused conversation has ended, {ctx.author.mention}. I'll now respond to others in this channel again.")
    else:
        await ctx.send("We weren't in a focused conversation. Use `!focus` to start one.")

@bot.command(name='malepreferences')
async def male_preferences_command(ctx, category=None):
    """Command to view what the bot has learned about male preferences."""
    # Check if preference learning is available
    if not PREFERENCE_LEARNING_AVAILABLE or not preference_learner:
        await ctx.send("Preference learning is not available. Please make sure the `preference_learning.py` module is installed.")
        return

    # If no category specified, show all categories
    if not category:
        categories = ["likes", "dislikes", "interests", "values"]

        # Create a summary of all categories
        summary = "**What I've Learned About Male Preferences**\n\n"

        for cat in categories:
            top_prefs = preference_learner.get_top_preferences(cat, 5)
            if top_prefs:
                summary += f"**Top {cat.capitalize()}:**\n"
                for i, (pref, count) in enumerate(top_prefs, 1):
                    summary += f"{i}. {pref.capitalize()} ({count} mentions)\n"
                summary += "\n"

        summary += "Use `!malepreferences [category]` to see more details about a specific category (likes, dislikes, interests, values)."

        await ctx.send(summary)
        return

    # If category is specified, show more details for that category
    category = category.lower()
    valid_categories = ["likes", "dislikes", "interests", "values"]

    if category not in valid_categories:
        await ctx.send(f"Invalid category: {category}. Valid categories are: {', '.join(valid_categories)}")
        return

    # Get top preferences for the category
    top_prefs = preference_learner.get_top_preferences(category, 10)

    if not top_prefs:
        await ctx.send(f"I haven't learned any male {category} yet. Keep chatting with me!")
        return

    # Create a detailed view of the category
    detail = f"**Top Male {category.capitalize()}**\n\n"

    for i, (pref, count) in enumerate(top_prefs, 1):
        detail += f"{i}. {pref.capitalize()} ({count} mentions)\n"

        # For the top 3 items, show related preferences
        if i <= 3:
            related = preference_learner.get_related_preferences(pref, category, 3)
            if related:
                detail += f"   Related: {', '.join(related)}\n"

    await ctx.send(detail)

@bot.command(name='gemini')
async def gemini_command(ctx, action=None, *, value=None):
    """Command to configure the Gemini API."""
    # Check if Gemini API is available
    if not GEMINI_API_AVAILABLE or not gemini_api:
        await ctx.send("Gemini API integration is not available. Please make sure the `gemini_api.py` module is installed.")
        return

    if not action:
        # Show current status
        if gemini_api.is_configured():
            await ctx.send("Gemini API is configured and ready to use. Use `!gemini help` to see available commands.")
        else:
            await ctx.send("Gemini API is not configured. Please set your API key with `!gemini key YOUR_API_KEY`.")
        return

    if action.lower() == "help":
        help_text = """
        **Gemini API Commands**

        • `!gemini` - Check Gemini API status
        • `!gemini help` - Show this help message
        • `!gemini key YOUR_API_KEY` - Set your Gemini API key
        • `!gemini test "Your test question"` - Test the Gemini API with a question
        • `!gemini on` - Enable Gemini for unknown questions
        • `!gemini off` - Disable Gemini for unknown questions

        You can get a Gemini API key from: https://makersuite.google.com/app/apikey
        """
        await ctx.send(help_text)
        return

    if action.lower() == "key":
        # Set API key
        if not value:
            await ctx.send("Please provide an API key. Usage: `!gemini key YOUR_API_KEY`")
            return

        # Delete the message to protect the API key
        try:
            await ctx.message.delete()
        except:
            # If we can't delete the message, warn the user
            await ctx.send("⚠️ I couldn't delete your message. Your API key might be visible in the chat history.")

        # Set the API key
        success = gemini_api.set_api_key(value)
        if success:
            await ctx.send("✅ Gemini API key set successfully! I can now answer questions I don't know about.")
        else:
            await ctx.send("❌ Failed to set Gemini API key. Please try again.")

        return

    if action.lower() == "test":
        # Test the API with a question
        if not gemini_api.is_configured():
            await ctx.send("Gemini API is not configured. Please set your API key with `!gemini key YOUR_API_KEY`.")
            return

        if not value:
            await ctx.send("Please provide a test question. Usage: `!gemini test \"What is the capital of France?\"`")
            return

        # Show typing indicator
        async with ctx.typing():
            response = gemini_api.generate_response(value)

            if response:
                await ctx.send(f"**Test Question:** {value}\n\n**Gemini Response:** {response}")
            else:
                await ctx.send("❌ Failed to get a response from Gemini API. Please check your API key and try again.")

        return

    if action.lower() in ["on", "enable"]:
        # Enable Gemini for unknown questions
        gemini_api.config["use_for_unknown_questions"] = True
        gemini_api.save_config("config/gemini_config.json")
        await ctx.send("✅ Gemini API is now enabled for unknown questions.")
        return

    if action.lower() in ["off", "disable"]:
        # Disable Gemini for unknown questions
        gemini_api.config["use_for_unknown_questions"] = False
        gemini_api.save_config("config/gemini_config.json")
        await ctx.send("✅ Gemini API is now disabled for unknown questions.")
        return

    # If we get here, the action wasn't recognized
    await ctx.send(f"Unknown action: {action}. Use `!gemini help` to see available commands.")

# Run the bot (token should be stored in environment variable or config file)
def run_bot(token):
    """Run the Discord bot with the provided token."""
    # Check if we're running on a hosting platform that needs a keep-alive server
    try:
        # Try to import the keep_alive module
        from keep_alive import keep_alive

        # Start the keep-alive server
        print("Starting keep-alive server for hosting platforms...")
        keep_alive()
    except ImportError:
        # If the module is not found, it's okay - we're probably running locally
        print("Keep-alive server not started (running locally)")
    except Exception as e:
        # If there's another error, log it but continue
        print(f"Error starting keep-alive server: {e}")

    # Run the bot
    bot.run(token)

if __name__ == "__main__":
    # Load token from environment variable or config file
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                token = config.get("discord_token")
        except:
            print("Error: Discord token not found in environment variables or config.json")
            exit(1)

    run_bot(token)
