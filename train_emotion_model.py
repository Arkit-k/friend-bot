"""
Train the emotion detection model for the friendship bot.
This script trains a model to detect emotions in text messages.
"""

import os
import json
import pickle
import argparse
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_emotion_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_emotion_model")

def load_data(data_file):
    """
    Load emotion-labeled data from a JSON file.
    
    Args:
        data_file: Path to the JSON file
        
    Returns:
        List of data items
    """
    logger.info(f"Loading data from {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} data items")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

def prepare_data(data, max_words=10000, max_sequence_length=100):
    """
    Prepare data for training.
    
    Args:
        data: List of data items
        max_words: Maximum number of words in the vocabulary
        max_sequence_length: Maximum sequence length
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, tokenizer, label_encoder)
    """
    logger.info("Preparing data for training...")
    
    # Extract texts and labels
    texts = [item["text"] for item in data]
    labels = [item["emotion_id"] for item in data]
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, encoded_labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Prepared {len(X_train)} training samples and {len(X_val)} validation samples")
    logger.info(f"Vocabulary size: {len(tokenizer.word_index)}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, X_val, y_train, y_val, tokenizer, label_encoder

def build_model(max_words, max_sequence_length, embedding_dim, num_classes):
    """
    Build the emotion detection model.
    
    Args:
        max_words: Maximum number of words in the vocabulary
        max_sequence_length: Maximum sequence length
        embedding_dim: Embedding dimension
        num_classes: Number of emotion classes
        
    Returns:
        Compiled model
    """
    logger.info("Building model...")
    
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    logger.info(model.summary())
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the model.
    
    Args:
        model: Compiled model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    logger.info(f"Training model for {epochs} epochs with batch size {batch_size}...")
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/emotion_model_checkpoint.h5',
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

def save_model(model, tokenizer, label_encoder):
    """
    Save the model, tokenizer, and label encoder.
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        label_encoder: Fitted label encoder
    """
    logger.info("Saving model, tokenizer, and label encoder...")
    
    # Save model
    model.save('friendship_model.h5')
    logger.info("Model saved to friendship_model.h5")
    
    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info("Tokenizer saved to tokenizer.pkl")
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info("Label encoder saved to label_encoder.pkl")

def evaluate_model(model, X_val, y_val, label_encoder):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: Trained model
        X_val: Validation data
        y_val: Validation labels
        label_encoder: Fitted label encoder
    """
    logger.info("Evaluating model on validation set...")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    logger.info(f"Validation loss: {loss:.4f}")
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    from sklearn.metrics import classification_report
    report = classification_report(
        y_val,
        y_pred_classes,
        target_names=label_encoder.classes_
    )
    logger.info(f"Classification report:\n{report}")

def main():
    """Main function to train the emotion detection model."""
    parser = argparse.ArgumentParser(description="Train the emotion detection model")
    parser.add_argument("--data-file", type=str, default="data/processed/emotion_labeled_data.json",
                        help="Path to the emotion-labeled data file")
    parser.add_argument("--max-words", type=int, default=10000,
                        help="Maximum number of words in the vocabulary")
    parser.add_argument("--max-sequence-length", type=int, default=100,
                        help="Maximum sequence length")
    parser.add_argument("--embedding-dim", type=int, default=200,
                        help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    data = load_data(args.data_file)
    if not data:
        logger.error("No data loaded. Exiting.")
        return 1
    
    # Prepare data
    X_train, X_val, y_train, y_val, tokenizer, label_encoder = prepare_data(
        data, args.max_words, args.max_sequence_length
    )
    
    # Build model
    num_classes = len(label_encoder.classes_)
    model = build_model(args.max_words, args.max_sequence_length, args.embedding_dim, num_classes)
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate model
    evaluate_model(model, X_val, y_val, label_encoder)
    
    # Save model
    save_model(model, tokenizer, label_encoder)
    
    logger.info("Model training completed successfully!")
    return 0

if __name__ == "__main__":
    main()
