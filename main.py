import discord
from discord.ext import commands
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Event: Bot is ready
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}!")

# Command to interact with the Groq chat model
@bot.command(name="chat")
async def chat(ctx, *, question: str):
    """Send a user prompt to the Groq chat model and return the response."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            temperature=1.0,
            max_tokens=800,
            top_p=1.0,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:  # Ensure content is not empty
                response += delta.content + " "

        await ctx.send(response.strip())

    except Exception as e:
        await ctx.send("Something went wrong. Please try again later.")
        print(f"Error: {e}")

# Run the bot
bot.run(DISCORD_TOKEN)
