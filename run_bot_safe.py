"""
Safe runner for the Discord bot that handles common errors gracefully.
"""

import os
import sys
import discord
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_bot_safely():
    """Run the Discord bot with error handling."""
    # Get the token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        print("Error: DISCORD_TOKEN not found in .env file")
        print("Please add your Discord token to the .env file")
        return 1
    
    try:
        # Import the bot module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from discord_bot import run_bot
        
        # Run the bot
        run_bot(token)
        return 0
    
    except discord.errors.PrivilegedIntentsRequired:
        print("\n" + "="*80)
        print("ERROR: PRIVILEGED INTENTS NOT ENABLED")
        print("="*80)
        print("\nYour bot is trying to use privileged intents that haven't been enabled in the Discord Developer Portal.")
        print("\nTo fix this issue:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Select your bot application")
        print("3. Click on 'Bot' in the left sidebar")
        print("4. Scroll down to 'Privileged Gateway Intents'")
        print("5. Enable 'MESSAGE CONTENT INTENT' and 'SERVER MEMBERS INTENT'")
        print("6. Click 'Save Changes'")
        print("7. Restart your bot")
        print("\nAlternatively, you can modify your bot code to not use these intents.")
        return 1
    
    except discord.errors.LoginFailure:
        print("\n" + "="*80)
        print("ERROR: INVALID DISCORD TOKEN")
        print("="*80)
        print("\nYour Discord token is invalid or has expired.")
        print("\nTo fix this issue:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Select your bot application")
        print("3. Click on 'Bot' in the left sidebar")
        print("4. Under 'TOKEN', click 'Reset Token'")
        print("5. Copy the new token")
        print("6. Update your .env file with the new token")
        print("7. Restart your bot")
        return 1
    
    except Exception as e:
        print(f"\nError running bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_bot_safely())
