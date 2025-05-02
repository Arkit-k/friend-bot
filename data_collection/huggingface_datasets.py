"""
Hugging Face Dataset Downloader

This script downloads and processes conversation datasets from Hugging Face
that can be useful for training a female friendship bot.
"""

import os
import pandas as pd
from datasets import load_dataset
import re

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# List of potentially useful datasets on Hugging Face
DATASETS = [
    {
        "name": "daily_dialog",
        "config": None,
        "description": "Daily conversations on various topics",
        "processing_func": "process_daily_dialog"
    },
    {
        "name": "empathetic_dialogues",
        "config": None,
        "description": "Empathetic conversations with emotional contexts",
        "processing_func": "process_empathetic_dialogues"
    },
    {
        "name": "blended_skill_talk",
        "config": None,
        "description": "Conversations blending empathy, personality, and knowledge",
        "processing_func": "process_blended_skill_talk"
    },
    {
        "name": "conv_ai_2",
        "config": None,
        "description": "Persona-based conversations",
        "processing_func": "process_conv_ai_2"
    }
]

def process_daily_dialog(dataset):
    """Process the Daily Dialog dataset."""
    data = []
    
    for item in dataset["train"]:
        dialog = item["dialog"]
        emotion = item["emotion"]
        
        for i in range(1, len(dialog)):
            # Assuming even indices might be male and odd indices might be female
            # This is a simplification and not always accurate
            if i % 2 == 1:  # Potential female response
                data.append({
                    "utterance": dialog[i-1],
                    "response": dialog[i],
                    "emotion_label": emotion[i] if i < len(emotion) else "unknown",
                    "dataset": "daily_dialog",
                    "is_female_response": True  # Assumption
                })
    
    return pd.DataFrame(data)

def process_empathetic_dialogues(dataset):
    """Process the Empathetic Dialogues dataset."""
    data = []
    
    for item in dataset["train"]:
        utterance = item.get("utterance", "")
        response = item.get("response", "")
        emotion = item.get("emotion", "unknown")
        
        if utterance and response:
            # Check if the speaker is likely female based on context
            is_female = None
            female_indicators = ["as a woman", "as a female", "i'm a woman", "i'm female"]
            if any(indicator in response.lower() for indicator in female_indicators):
                is_female = True
            
            data.append({
                "utterance": utterance,
                "response": response,
                "emotion_label": emotion,
                "dataset": "empathetic_dialogues",
                "is_female_response": is_female
            })
    
    return pd.DataFrame(data)

def process_blended_skill_talk(dataset):
    """Process the Blended Skill Talk dataset."""
    data = []
    
    for item in dataset["train"]:
        dialog = item.get("dialog", [])
        personas = item.get("personas", [])
        
        # Extract potential gender information from personas
        is_female = None
        for persona in personas:
            if "i am a woman" in persona.lower() or "i am female" in persona.lower():
                is_female = True
                break
        
        for i in range(1, len(dialog), 2):
            if i < len(dialog):
                data.append({
                    "utterance": dialog[i-1],
                    "response": dialog[i],
                    "emotion_label": "unknown",  # This dataset doesn't have emotion labels
                    "dataset": "blended_skill_talk",
                    "is_female_response": is_female,
                    "persona": "; ".join(personas)
                })
    
    return pd.DataFrame(data)

def process_conv_ai_2(dataset):
    """Process the ConvAI2 dataset."""
    data = []
    
    for item in dataset["train"]:
        dialog = item.get("dialog", [])
        personas = item.get("personas", [])
        
        # Extract potential gender information from personas
        is_female = None
        for persona in personas:
            if "i am a woman" in persona.lower() or "i am female" in persona.lower():
                is_female = True
                break
        
        for i in range(1, len(dialog), 2):
            if i < len(dialog):
                data.append({
                    "utterance": dialog[i-1],
                    "response": dialog[i],
                    "emotion_label": "unknown",  # This dataset doesn't have emotion labels
                    "dataset": "conv_ai_2",
                    "is_female_response": is_female,
                    "persona": "; ".join(personas)
                })
    
    return pd.DataFrame(data)

def download_and_process_dataset(dataset_info):
    """Download and process a dataset from Hugging Face."""
    name = dataset_info["name"]
    config = dataset_info["config"]
    processing_func_name = dataset_info["processing_func"]
    
    print(f"Downloading dataset: {name}")
    
    try:
        # Load the dataset
        dataset = load_dataset(name, config)
        
        # Process the dataset
        processing_func = globals()[processing_func_name]
        df = processing_func(dataset)
        
        # Save the processed dataset
        output_file = f"data/{name}_processed.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Processed {len(df)} examples from {name}")
        print(f"Saved to {output_file}")
        
        return df
    
    except Exception as e:
        print(f"Error processing dataset {name}: {e}")
        return None

def main():
    """Main function to download and process all datasets."""
    all_data = []
    
    for dataset_info in DATASETS:
        df = download_and_process_dataset(dataset_info)
        if df is not None:
            all_data.append(df)
    
    # Combine all datasets
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv("data/all_huggingface_datasets.csv", index=False)
        print(f"Combined dataset saved with {len(combined_df)} examples")
    else:
        print("No datasets were successfully processed")

if __name__ == "__main__":
    main()
