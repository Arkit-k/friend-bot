"""
Simple script to test if your Discord token is valid.
"""

import os
import discord
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the token
TOKEN = os.getenv('DISCORD_TOKEN')

# Print token (first few characters only for security)
if TOKEN:
    print(f"Found token: {TOKEN[:10]}...")
else:
    print("No token found in .env file")
    exit(1)

# Create a simple client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Successfully logged in as {client.user} (ID: {client.user.id})')
    print(f'Bot is in {len(client.guilds)} servers')
    for guild in client.guilds:
        print(f'- {guild.name} (ID: {guild.id})')
    
    # Exit after successful login
    await client.close()

@client.event
async def on_error(event, *args, **kwargs):
    print(f'Error in event {event}: {args} {kwargs}')

# Run the client
try:
    print("Attempting to connect to Discord...")
    client.run(TOKEN)
    print("Connection successful!")
except discord.errors.LoginFailure as e:
    print(f"Error: Invalid token. Please check your Discord token. Error details: {e}")
except Exception as e:
    print(f"Error: {e}")
