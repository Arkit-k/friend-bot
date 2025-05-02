#!/usr/bin/env python
"""
Launcher script for the Friendship Discord Bot.
"""

import os
import json
import argparse
from discord_bot import run_bot

def main():
    """Main entry point for the bot."""
    parser = argparse.ArgumentParser(description='Run the Friendship Discord Bot')
    parser.add_argument('--token', help='Discord bot token (overrides config file)')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Get token from command line args, environment variable, or config file
    token = args.token or os.getenv("DISCORD_TOKEN")
    
    # If token not provided via args or env var, try to load from config file
    if not token:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                token = config.get('discord_token')
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Please provide a Discord token via --token argument, DISCORD_TOKEN environment variable, or config.json file.")
            return
    
    if not token or token == "YOUR_DISCORD_BOT_TOKEN_HERE":
        print("Error: No valid Discord token found.")
        print("Please set your Discord token in config.json or provide it via command line argument or environment variable.")
        return
    
    print("Starting Friendship Discord Bot...")
    run_bot(token)

if __name__ == "__main__":
    main()
