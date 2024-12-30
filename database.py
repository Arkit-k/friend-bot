import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

def initialize_db():
    interactions = pd.DataFrame(columns=['user_id', 'question', 'response', 'feedback'])
    interactions.to_csv('interactions.csv', index=False)

def add_question_answer(question, answer):
    interactions = pd.read_csv('interactions.csv')
    new_interaction = pd.DataFrame([[None, question, answer, None]], columns=['user_id', 'question', 'response', 'feedback'])
    interactions = pd.concat([interactions, new_interaction], ignore_index=True)
    interactions.to_csv('interactions.csv', index=False)

def get_answer(question):
    interactions = pd.read_csv('interactions.csv')
    answer = interactions.loc[interactions['question'] == question, 'response']
    if not answer.empty:
        return answer.iloc[0]
    return None

def store_interaction(user_id, question, response):
    interactions = pd.read_csv('interactions.csv')
    new_interaction = pd.DataFrame([[user_id, question, response, None]], columns=['user_id', 'question', 'response', 'feedback'])
    interactions = pd.concat([interactions, new_interaction], ignore_index=True)
    interactions.to_csv('interactions.csv', index=False)

def get_all_interactions():
    return pd.read_csv('interactions.csv')