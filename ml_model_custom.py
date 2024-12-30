import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize an empty DataFrame to store interactions
interactions = pd.DataFrame(columns=['user_id', 'question', 'response', 'feedback'])

def store_interaction(user_id, question, response):
    global interactions
    new_interaction = pd.DataFrame([[user_id, question, response, None]], columns=['user_id', 'question', 'response', 'feedback'])
    interactions = pd.concat([interactions, new_interaction], ignore_index=True)

def train_model(interactions):
    vectorizer = TfidfVectorizer()
    if interactions.empty or interactions['feedback'].notnull().sum() == 0:
        # No feedback available to train the model, return a default model and vectorizer
        model = LogisticRegression()
        return model, vectorizer

    # Prepare the data
    X = vectorizer.fit_transform(interactions['question'])
    y = interactions['feedback']

    # Train the model
    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

def predict_response(model, vectorizer, question):
    X = vectorizer.transform([question])
    prediction = model.predict(X)
    return prediction[0]

def update_feedback(user_id, question, feedback):
    interactions = pd.read_csv('interactions.csv')
    interactions.loc[(interactions['user_id'] == user_id) & (interactions['question'] == question), 'feedback'] = feedback
    interactions.to_csv('interactions.csv', index=False)