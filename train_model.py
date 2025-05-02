"""
Model Training Script for Friendship Bot

This script trains a machine learning model on the collected conversation data
to improve the bot's ability to generate appropriate responses.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """Load the processed data."""
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validation.csv'
    test_path = 'data/processed/test.csv'
    
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        print("Processed data files not found. Please run data_processor.py first.")
        return None, None, None
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def preprocess_data(train_df, val_df, test_df, max_words=10000, max_sequence_length=100):
    """Preprocess the data for model training."""
    # Extract features and labels
    X_train = train_df['utterance'].values
    y_train = train_df['emotion_label'].values
    
    X_val = val_df['utterance'].values
    y_val = val_df['emotion_label'].values
    
    X_test = test_df['utterance'].values
    y_test = test_df['emotion_label'].values
    
    # Tokenize the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_sequence_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Save tokenizer and label encoder
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return (X_train_pad, y_train_encoded, X_val_pad, y_val_encoded, 
            X_test_pad, y_test_encoded, tokenizer, label_encoder)

def build_model(max_words, max_sequence_length, embedding_dim=128, num_classes=6):
    """Build the model architecture."""
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length))
    
    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Train the model."""
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model and generate performance metrics."""
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    
    return loss, accuracy

def plot_training_history(history):
    """Plot training history."""
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')

def main():
    """Main function to train the model."""
    print("Loading data...")
    train_df, val_df, test_df = load_data()
    
    if train_df is None:
        return
    
    print("Preprocessing data...")
    max_words = 10000
    max_sequence_length = 100
    
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     tokenizer, label_encoder) = preprocess_data(train_df, val_df, test_df, 
                                                max_words, max_sequence_length)
    
    print("Building model...")
    num_classes = len(label_encoder.classes_)
    model = build_model(max_words, max_sequence_length, embedding_dim=128, num_classes=num_classes)
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)
    
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Saving model...")
    model.save('models/friendship_model.h5')
    
    print("Training complete!")

if __name__ == "__main__":
    main()
