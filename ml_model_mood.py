import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

# Load the merged dataset
data = pd.read_csv("final_merged_dataset.csv")

# Preprocessing: Handle missing data
data['utterance'] = data['utterance'].fillna("No utterance provided.")
data['response'] = data['response'].fillna("No response provided.")

# Select input features and labels (Emotion Label for classification)
X = data['utterance']
y = data['emotion_label']  # or 'sentiment_label' for sentiment classification

# Encode the labels (emotion_label or sentiment_label)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize the utterances (text)
tokenizer = Tokenizer(num_words=5000)  # Use top 5000 words
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform input length
max_sequence_length = 100  # Adjust based on your data
X = pad_sequences(X, maxlen=max_sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture
model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length))

# LSTM Layer to capture sequential context
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))  # To prevent overfitting
model.add(LSTM(128))
model.add(Dropout(0.2))

# Global average pooling
model.add(Dense(64, activation='relu'))  # Directly using Dense instead of pooling
model.add(Dropout(0.2))

# Output Layer - Softmax for multi-class classification (Emotion classification)
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model
model.save("friendship_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the LabelEncoder for later use
with open("label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

# Print dataset shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")









