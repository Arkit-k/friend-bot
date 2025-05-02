#!/usr/bin/env python
"""
Setup script for the Friendship Discord Bot.
This script helps users set up their bot by creating a config.json file with their Discord token.
"""

import os
import json
import argparse
from getpass import getpass

def main():
    """Main setup function."""
    print("=== Friendship Discord Bot Setup ===")
    print("This script will help you set up your Discord bot.")
    print("You'll need a Discord bot token from the Discord Developer Portal.")
    print("Visit https://discord.com/developers/applications to create a bot and get a token.")
    print("\n")
    
    # Check if config.json already exists
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
        
        print("Existing configuration found:")
        print(f"Bot Name: {config.get('bot_name', 'Lily')}")
        print(f"Command Prefix: {config.get('command_prefix', '!')}")
        
        if config.get("discord_token") != "YOUR_DISCORD_BOT_TOKEN_HERE":
            print("Discord Token: [Already Set]")
        else:
            print("Discord Token: [Not Set]")
        
        update = input("\nDo you want to update this configuration? (y/n): ").lower()
        if update != "y":
            print("Setup cancelled. Existing configuration will be used.")
            return
    else:
        config = {
            "discord_token": "YOUR_DISCORD_BOT_TOKEN_HERE",
            "bot_name": "Lily",
            "command_prefix": "!",
            "personality": {
                "name": "Lily",
                "age": "21",
                "gender": "female",
                "traits": {
                    "caring": 0.8,
                    "playful": 0.6,
                    "empathetic": 0.9,
                    "supportive": 0.9,
                    "positive": 0.7
                }
            }
        }
    
    # Get Discord token
    token = getpass("\nEnter your Discord bot token (input will be hidden): ")
    if token:
        config["discord_token"] = token
    
    # Get bot name
    bot_name = input(f"\nEnter bot name [{config.get('bot_name', 'Lily')}]: ")
    if bot_name:
        config["bot_name"] = bot_name
        config["personality"]["name"] = bot_name
    
    # Get command prefix
    prefix = input(f"\nEnter command prefix [{config.get('command_prefix', '!')}]: ")
    if prefix:
        config["command_prefix"] = prefix
    
    # Save configuration
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("\nConfiguration saved to config.json")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the bot: python run_bot.py")
    print("\nEnjoy your Friendship Bot!")

if __name__ == "__main__":
    main()
