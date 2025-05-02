"""
Create dummy emotion detection model files for testing.

This script creates simple placeholder files for the emotion detection model,
tokenizer, and label encoder to allow the bot to run without errors.
"""

import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

def create_dummy_model():
    """Create a dummy emotion detection model."""
    print("Creating dummy emotion detection model files...")
    
    # Create a simple model
    model = Sequential()
    model.add(Dense(10, input_shape=(100,), activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Save the model
    model.save('friendship_model.h5')
    print("Created dummy model: friendship_model.h5")
    
    # Create a dummy tokenizer
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(["happy", "sad", "angry", "fear", "love", "surprise"])
    
    # Save the tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Created dummy tokenizer: tokenizer.pkl")
    
    # Create a dummy label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(["joy", "sadness", "anger", "fear", "love", "surprise"])
    
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Created dummy label encoder: label_encoder.pkl")
    
    print("Dummy model files created successfully!")

if __name__ == "__main__":
    create_dummy_model()
