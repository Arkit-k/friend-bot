import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Path to the saved model and tokenizer files
model_path = "friendship_model.h5"
tokenizer_path = "tokenizer.pkl"
label_encoder_path = "label_encoder.pkl"

# Load the model
model = load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Load the tokenizer
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
else:
    # Recreate the tokenizer if it doesn't exist
    data = pd.read_csv("csv/final_merged_dataset.csv")
    X_train_texts = data['utterance']  # Use your training data column
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train_texts)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

# Load or recreate the label encoder
if os.path.exists(label_encoder_path):
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    y = data['emotion_label']  # Replace with the correct label column
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

@app.route("/predict", methods=["POST"])
def predict():
    print("Received a POST request at /predict")  # Add this line for debugging
    try:
        # Get the texts from the POST request
        data = request.get_json()
        texts = data.get("texts", [])
        print(f"Received texts: {texts}")  # Add this line for debugging

        if not texts:
            return jsonify({"error": "No texts provided in the request."})

        # Tokenizing the texts
        sequences = tokenizer.texts_to_sequences(texts)
        print(f"Tokenized sequences: {sequences}")  # Add this line for debugging

        # Padding the sequences to ensure they are of the same length
        padded_sequences = pad_sequences(sequences, maxlen=100)
        print(f"Padded sequences: {padded_sequences}")  # Add this line for debugging

        # Make predictions
        predictions = model.predict(padded_sequences)
        print(f"Raw predictions: {predictions}")  # Add this line for debugging

        # Decode predictions back to the labels
        predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        print(f"Decoded predictions: {predicted_labels}")  # Add this line for debugging

        return jsonify({"predictions": predicted_labels.tolist()})
        

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)






