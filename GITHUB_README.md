# Friendship Bot

A Discord bot designed to be a supportive female AI friend, with advanced features like emotion detection, preference learning, and flirting capabilities.

## Features

- **Emotion Detection**: Detects user emotions from messages
- **User Profiling**: Learns user communication styles and preferences
- **Male Preference Learning**: Learns what male users like from conversations
- **Flirting Techniques**: Uses tasteful flirting to make conversations more fun
- **Focused Conversations**: Prevents interruptions during one-on-one chats
- **Gemini API Integration**: Uses Google's Gemini for answering complex questions
- **Reddit Integration**: Collects conversation data from Reddit

## Deployment Options

This bot can be deployed on various platforms:

### Render (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed instructions.

### Other Free Options

- **Replit**: Easy setup, good for beginners
- **Railway**: 500 hours of free runtime per month
- **Heroku**: Limited free tier
- **Oracle Cloud**: Most powerful free option

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Discord token:
   ```
   DISCORD_TOKEN=your_discord_token_here
   ```
4. Run the bot: `python discord_bot.py`

## Commands

- `!friendhelp` - Show help information
- `!reset` - Reset conversation history
- `!mood` - Check detected mood
- `!profile` - View communication profile
- `!preference` - Set preferences
- `!like` - Tell the bot something you like
- `!dislike` - Tell the bot something you dislike
- `!focus` - Start a focused conversation
- `!unfocus` - End a focused conversation
- `!malepreferences` - View learned male preferences
- `!gemini` - Configure Gemini API (if available)

## Important Notes

1. **Privileged Intents**: This bot requires privileged intents to be enabled in the Discord Developer Portal:
   - MESSAGE CONTENT INTENT
   - SERVER MEMBERS INTENT

2. **API Keys**: To use all features, you'll need:
   - Discord Bot Token
   - Gemini API Key (optional)
   - Reddit API Credentials (optional)

## Documentation

- [DISCORD_INTEGRATION_GUIDE.md](DISCORD_INTEGRATION_GUIDE.md) - How to set up Discord integration
- [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) - How to deploy to Render
- [ENV_SETUP.md](ENV_SETUP.md) - How to set up environment variables
- [PREFERENCE_LEARNING.md](PREFERENCE_LEARNING.md) - How the preference learning system works
- [REDDIT_INTEGRATION.md](REDDIT_INTEGRATION.md) - How the Reddit API integration works
