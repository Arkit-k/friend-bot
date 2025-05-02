# Friendship Bot

A Discord-based AI friendship bot designed to provide emotional support and companionship. This bot uses machine learning to detect emotions in user messages and responds with empathetic, supportive messages. It's specifically designed as a female AI friend model to support male users.

## Features

- **Emotion Detection**: Uses a machine learning model to detect emotions in user messages
- **Personalized Responses**: Generates responses based on detected emotions and conversation context
- **Conversation Memory**: Maintains conversation history to provide context-aware responses
- **Support Features**: Offers encouragement, active listening, and emotional support
- **Discord Integration**: Seamlessly integrates with Discord for easy interaction
- **Female Interaction Patterns**: Trained on authentic female-male interactions to provide genuine support
- **Male Preference Learning**: Learns what male users like from conversations to provide better responses
- **Flirting Techniques**: Uses tasteful flirting to make conversations more fun and engaging
- **Focused Conversations**: Prevents interruptions during one-on-one chats
- **Gemini API Integration**: Uses Google's Gemini for answering complex questions the bot doesn't know

## Setup

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure the Bot**:
   - Create a Discord bot on the [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a `.env` file with your Discord token:
     ```
     DISCORD_TOKEN=your_discord_token_here
     ```
   - (Optional) Add other API keys to the `.env` file:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     REDDIT_CLIENT_ID=your_reddit_client_id_here
     REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
     ```
   - Invite the bot to your server using the OAuth2 URL generator

3. **Run the Bot**:
   ```
   # Using the batch file (recommended for Windows)
   run_bot.bat

   # Or manually
   python create_dummy_model.py  # Only needed first time
   python discord_bot.py
   ```

## Usage

- Talk to the bot in DMs for private conversations
- Mention the bot in a server to chat with it there
- Use commands:
  - `!friendhelp` - Show help information
  - `!reset` - Reset conversation history
  - `!mood` - Check how the bot thinks you're feeling
  - `!profile` - View your communication profile
  - `!preference` - Set your preferences (e.g., `!preference formality formal`)
  - `!preference flirting` - Set flirting level (none/subtle/playful/moderate)
  - `!like` - Tell the bot something you like (e.g., `!like chocolate`)
  - `!dislike` - Tell the bot something you dislike (e.g., `!dislike spicy food`)
  - `!focus` - Start a focused conversation where the bot won't respond to others
  - `!unfocus` - End a focused conversation
  - `!malepreferences` - View what the bot has learned about male preferences
  - `!gemini` - Configure Gemini API for answering complex questions

## Project Structure

- **Bot Core**:
  - `discord_bot.py` - Main Discord bot file with event handlers
  - `response_generator.py` - Logic for generating responses based on emotions
  - `conversation_memory.py` - System to maintain conversation context
  - `conversation_state.py` - System to manage conversation states and prevent interruptions
  - `preference_learning.py` - System to learn male preferences from conversations
  - `flirting_techniques.py` - Implementation of flirting techniques
  - `gemini_api.py` - Integration with Google's Gemini API
  - `reddit_api.py` - Integration with Reddit API for data collection
  - `run_bot.bat` - Batch file to run the bot on Windows
  - `create_dummy_model.py` - Script to create dummy model files

- **Model & API**:
  - `friendship_model.h5` - Emotion detection model
  - `tokenizer.pkl` - Tokenizer for text preprocessing
  - `label_encoder.pkl` - Label encoder for emotion classes

- **Configuration**:
  - `.env` - Environment variables for API keys and tokens
  - `.env.example` - Example environment variables file

- **Documentation**:
  - `README.md` - Main documentation file
  - `ENV_SETUP.md` - Guide for setting up environment variables
  - `PREFERENCE_LEARNING.md` - Documentation for the preference learning system
  - `FLIRTING_TECHNIQUES.md` - Documentation for the flirting system
  - `CONVERSATION_PROTECTION.md` - Documentation for the focused conversation system
  - `GEMINI_INTEGRATION.md` - Documentation for the Gemini API integration
  - `REDDIT_INTEGRATION.md` - Documentation for the Reddit API integration

## Data Collection and Model Training

The bot is designed to be trained on authentic female-male interactions to provide genuine emotional support. The data collection process includes:

1. **Gathering Data from Multiple Sources**:
   - Reddit conversations from supportive subreddits
   - Hugging Face datasets with emotional context
   - Books about women's psychology and communication patterns
   - Communication techniques from interpersonal skills books
   - Filtering for supportive female responses

2. **Processing and Cleaning**:
   - Extracting relevant content from books about female psychology
   - Analyzing communication techniques from "How to Win Friends and Influence People" and "How to Talk to Anyone"
   - Generating synthetic conversations demonstrating these techniques
   - Removing inappropriate content
   - Cleaning and normalizing text
   - Labeling with emotions

3. **Training the Model**:
   - Using bidirectional LSTM networks for emotion detection
   - Incorporating insights from women's psychology literature
   - Applying proven communication techniques from interpersonal skills books
   - Fine-tuning for supportive conversation patterns
   - Optimizing for authentic female communication styles
   - Learning effective ways to build rapport and provide emotional support

To run the entire data collection and training pipeline:
```
python run_data_pipeline.py

# To include book search for additional women's psychology literature:
python run_data_pipeline.py --search-books

# To focus only on communication techniques from interpersonal skills books:
python run_data_pipeline.py --focus-techniques
```

For more details on the data collection process, see the [data_collection README](data_collection/README.md).

## Customization

You can customize the bot's personality and responses by modifying:
- `config.py` - Adjust personality traits and response settings
- `response_generator.py` - Add or modify response templates
- Train on your own data by following the data collection process

## License

This project is licensed under the MIT License - see the LICENSE file for details.
