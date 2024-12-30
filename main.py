import discord
from discord.ext import commands
import os
from groq import Groq
from dotenv import load_dotenv
import asyncio
from database import initialize_db, add_question_answer, get_answer, store_interaction, get_all_interactions
from ml_model_custom import train_model, predict_response, update_feedback
from ml_model_mood import train_sentiment_model, predict_mood
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# Initialize the database
initialize_db()

# Store user names
user_names = {}

# Train the sentiment model
sentiment_model, sentiment_vectorizer = train_sentiment_model()
if sentiment_model is None or sentiment_vectorizer is None:
    sentiment_model, sentiment_vectorizer = LogisticRegression(), TfidfVectorizer()

# Train the custom response model
response_model, response_vectorizer = train_model(get_all_interactions())

# Event: Bot is ready
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}!")

# Command to interact with the Groq chat model
@bot.command(name="blify")
async def blify(ctx, *, question: str = None):
    """Send a user prompt to the Groq chat model and return the response."""
    if question is None:
        await ctx.send("Please provide a question.")
        return

    # Check for predefined answers in the database
    answer = get_answer(question)
    if answer:
        await ctx.send(answer)
        return

    user_id = str(ctx.author.id)
    if user_id not in user_names:
        await ctx.send("Hi there! What's your name?")
        try:
            msg = await bot.wait_for('message', check=lambda message: message.author == ctx.author, timeout=30.0)
            user_names[user_id] = msg.content
            await ctx.send(f"Nice to meet you, {msg.content}!")
            return
        except asyncio.TimeoutError:
            # Do not send a message to the channel if the user takes too long to respond
            return

    user_name = user_names[user_id]

    # Predict the mood of the user
    mood = predict_mood(sentiment_model, sentiment_vectorizer, question)

    try:
        # Add a timer for the request
        async with ctx.typing():
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": f"You are Blify, a friendly and supportive best friend. You know the user's name is {user_name}. The user is in a {mood} mood and is interested in {question}. Focus on this topic."},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.9,  # Adjust temperature for more engaging responses
                    max_tokens=800,
                    top_p=1.0,
                    stream=False,  # Disable streaming
                    stop=None,
                ),
                timeout=30.0  # Set a timeout for the request
            )

        response = completion.choices[0].message.content

        # Store the question and response in the database
        add_question_answer(question, response.strip())
        store_interaction(user_id, question, response.strip())

        await ctx.send(response.strip())

        # Update the sentiment model with new interactions
        sentiment_model, sentiment_vectorizer = train_sentiment_model()
        if sentiment_model is None or sentiment_vectorizer is None:
            sentiment_model, sentiment_vectorizer = LogisticRegression(), TfidfVectorizer()

        # Update the custom response model with new interactions
        response_model, response_vectorizer = train_model(get_all_interactions())

    except asyncio.TimeoutError:
        await ctx.send("The request took too long to process. Please try again later.")
    except Exception as e:
        await ctx.send("Something went wrong. Please try again later.")
        print(f"Error: {e}")

# Run the bot
bot.run(DISCORD_TOKEN)
