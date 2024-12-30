import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

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

def get_all_interactions():
    return pd.read_csv('interactions.csv')

def store_interaction(user_id, question, response):
    interactions = pd.read_csv('interactions.csv')
    new_interaction = pd.DataFrame([[user_id, question, response, None]], columns=['user_id', 'question', 'response', 'feedback'])
    interactions = pd.concat([interactions, new_interaction], ignore_index=True)
    interactions.to_csv('interactions.csv', index=False)

def train_sentiment_model():
    try:
        interactions = pd.read_csv('interactions.csv')
    except FileNotFoundError:
        interactions = pd.DataFrame(columns=['user_id', 'question', 'response', 'feedback'])

    vectorizer = TfidfVectorizer()
    if interactions.empty:
        # No data available to train the model, return None
        return None, None

    # Prepare the data
    interactions['sentiment'] = interactions['question'].apply(lambda x: TextBlob(x).sentiment.polarity)
    interactions['mood'] = interactions['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

    X = vectorizer.fit_transform(interactions['question'])
    y = interactions['mood']

    # Train the model
    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

def predict_mood(model, vectorizer, question):
    if model is None or vectorizer is None:
        return 'neutral'  # Default mood if model or vectorizer is None
    X = vectorizer.transform([question])
    prediction = model.predict(X)
    return prediction[0]

def update_feedback(user_id, question, feedback):
    interactions = pd.read_csv('interactions.csv')
    interactions.loc[(interactions['user_id'] == user_id) & (interactions['question'] == question), 'feedback'] = feedback
    interactions.to_csv('interactions.csv', index=False)