"""
Data Processor for Friendship Bot

This script processes and cleans the collected conversation data,
preparing it for training the friendship bot model.
"""

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
from sklearn.model_selection import train_test_split

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace URLs with a token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Replace user mentions with a token
    text = re.sub(r'@\w+', '[USER]', text)

    # Replace subreddit mentions with a token
    text = re.sub(r'r/\w+', '[SUBREDDIT]', text)

    # Convert emojis to text
    text = emoji.demojize(text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep emoticons
    text = re.sub(r'[^\w\s\:\;\(\)\[\]\{\}\<\>\-\_\.\,\!\?\'\"\@\#\$\%\^\&\*\+\=\/\\]', '', text)

    return text.strip()

def filter_inappropriate_content(df):
    """Filter out inappropriate content."""
    # List of inappropriate keywords to filter out
    inappropriate_keywords = [
        # Add inappropriate keywords here
        "nsfw", "explicit", "porn", "sex", "nude", "xxx"
    ]

    # Create a regex pattern for filtering
    pattern = '|'.join(inappropriate_keywords)

    # Filter out rows containing inappropriate content
    mask = ~(df['utterance'].str.contains(pattern, case=False, na=False) |
             df['response'].str.contains(pattern, case=False, na=False))

    return df[mask]

def add_emotion_labels(df):
    """Add or refine emotion labels based on text content."""
    # Simple emotion keyword mapping
    emotion_keywords = {
        'joy': ['happy', 'joy', 'excited', 'glad', 'wonderful', 'love', 'amazing', 'great'],
        'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'down', 'upset', 'hurt'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
        'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
        'neutral': []  # Default
    }

    def detect_emotion(text):
        if not isinstance(text, str):
            return 'neutral'

        text = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                return emotion
        return 'neutral'

    # Apply emotion detection to responses without labels
    mask = df['emotion_label'].isin(['unknown', ''])
    df.loc[mask, 'emotion_label'] = df.loc[mask, 'response'].apply(detect_emotion)

    return df

def process_reddit_data():
    """Process Reddit data."""
    # Find all Reddit data files
    reddit_files = [f for f in os.listdir('data') if f.startswith('reddit_female_responses_') and f.endswith('.csv')]

    if not reddit_files:
        print("No Reddit data files found")
        return None

    # Load and combine all Reddit data
    dfs = []
    for file in reddit_files:
        df = pd.read_csv(os.path.join('data', file))
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Rename columns to match our format
    combined_df = combined_df.rename(columns={
        'parent_text': 'utterance',
        'comment_text': 'response'
    })

    # Add dataset source
    combined_df['dataset'] = 'reddit'

    # Add emotion label (will be refined later)
    combined_df['emotion_label'] = 'unknown'

    # Select and reorder columns
    columns = ['utterance', 'response', 'emotion_label', 'dataset', 'is_female']
    combined_df = combined_df[columns]

    # Clean text
    combined_df['utterance'] = combined_df['utterance'].apply(clean_text)
    combined_df['response'] = combined_df['response'].apply(clean_text)

    # Filter out rows with empty utterances or responses
    combined_df = combined_df[combined_df['utterance'].str.len() > 0]
    combined_df = combined_df[combined_df['response'].str.len() > 0]

    # Filter inappropriate content
    combined_df = filter_inappropriate_content(combined_df)

    # Add emotion labels
    combined_df = add_emotion_labels(combined_df)

    return combined_df

def process_huggingface_data():
    """Process Hugging Face data."""
    # Check if the combined file exists
    hf_file = 'data/all_huggingface_datasets.csv'
    if not os.path.exists(hf_file):
        print("No Hugging Face data file found")
        return None

    # Load the data
    df = pd.read_csv(hf_file)

    # Clean text
    df['utterance'] = df['utterance'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

    # Filter out rows with empty utterances or responses
    df = df[df['utterance'].str.len() > 0]
    df = df[df['response'].str.len() > 0]

    # Filter inappropriate content
    df = filter_inappropriate_content(df)

    # Add emotion labels
    df = add_emotion_labels(df)

    return df

def process_book_data():
    """Process book data."""
    # Check if the book data file exists
    book_file = 'data/books/book_conversations.csv'
    if not os.path.exists(book_file):
        print("No book data file found")
        return None

    # Load the data
    df = pd.read_csv(book_file)

    # Clean text
    df['utterance'] = df['utterance'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

    # Filter out rows with empty utterances or responses
    df = df[df['utterance'].str.len() > 0]
    df = df[df['response'].str.len() > 0]

    # Filter inappropriate content
    df = filter_inappropriate_content(df)

    # Add emotion labels
    df = add_emotion_labels(df)

    return df

def process_communication_techniques_data():
    """Process communication techniques data from specific books."""
    # Check if the communication techniques data file exists
    techniques_file = 'data/books/communication_techniques.csv'
    if not os.path.exists(techniques_file):
        print("No communication techniques data file found")
        return None

    # Load the data
    df = pd.read_csv(techniques_file)

    # Clean text
    df['utterance'] = df['utterance'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

    # Filter out rows with empty utterances or responses
    df = df[df['utterance'].str.len() > 0]
    df = df[df['response'].str.len() > 0]

    # Filter inappropriate content
    df = filter_inappropriate_content(df)

    # Add emotion labels
    df = add_emotion_labels(df)

    return df

def process_synthetic_conversations_data():
    """Process synthetic conversations data demonstrating communication techniques."""
    # Check if the synthetic conversations data file exists
    conversations_file = 'data/books/conversations/synthetic_conversations.csv'
    if not os.path.exists(conversations_file):
        print("No synthetic conversations data file found")
        return None

    # Load the data
    df = pd.read_csv(conversations_file)

    # Clean text
    df['utterance'] = df['utterance'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

    # Filter out rows with empty utterances or responses
    df = df[df['utterance'].str.len() > 0]
    df = df[df['response'].str.len() > 0]

    # Filter inappropriate content
    df = filter_inappropriate_content(df)

    # Add emotion labels
    df = add_emotion_labels(df)

    return df

def main():
    """Main function to process all data."""
    # Process Reddit data
    reddit_df = process_reddit_data()
    if reddit_df is not None:
        print(f"Processed {len(reddit_df)} Reddit conversations")
        reddit_df.to_csv('data/processed/reddit_processed.csv', index=False)

    # Process Hugging Face data
    hf_df = process_huggingface_data()
    if hf_df is not None:
        print(f"Processed {len(hf_df)} Hugging Face conversations")
        hf_df.to_csv('data/processed/huggingface_processed.csv', index=False)

    # Process Book data
    book_df = process_book_data()
    if book_df is not None:
        print(f"Processed {len(book_df)} Book conversations")
        book_df.to_csv('data/processed/book_processed.csv', index=False)

    # Process Communication Techniques data
    techniques_df = process_communication_techniques_data()
    if techniques_df is not None:
        print(f"Processed {len(techniques_df)} Communication Techniques examples")
        techniques_df.to_csv('data/processed/techniques_processed.csv', index=False)

    # Process Synthetic Conversations data
    synthetic_df = process_synthetic_conversations_data()
    if synthetic_df is not None:
        print(f"Processed {len(synthetic_df)} Synthetic Conversations")
        synthetic_df.to_csv('data/processed/synthetic_conversations_processed.csv', index=False)

    # Combine all data
    dfs = []
    if reddit_df is not None:
        dfs.append(reddit_df)
    if hf_df is not None:
        dfs.append(hf_df)
    if book_df is not None:
        dfs.append(book_df)
    if techniques_df is not None:
        dfs.append(techniques_df)
    if synthetic_df is not None:
        dfs.append(synthetic_df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['utterance', 'response'])

        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Save the datasets
        train_df.to_csv('data/processed/train.csv', index=False)
        val_df.to_csv('data/processed/validation.csv', index=False)
        test_df.to_csv('data/processed/test.csv', index=False)
        combined_df.to_csv('data/processed/final_merged_dataset.csv', index=False)

        print(f"Final dataset: {len(combined_df)} conversations")
        print(f"Train set: {len(train_df)} conversations")
        print(f"Validation set: {len(val_df)} conversations")
        print(f"Test set: {len(test_df)} conversations")

        # Print dataset source distribution
        print("\nDataset source distribution:")
        print(combined_df['dataset'].value_counts())

        # Print emotion distribution
        print("\nEmotion distribution:")
        print(combined_df['emotion_label'].value_counts())

        # If techniques data is available, print technique distribution
        if 'technique' in combined_df.columns and combined_df['technique'].notna().any():
            print("\nCommunication technique distribution:")
            print(combined_df['technique'].value_counts().head(10))  # Show top 10 techniques
    else:
        print("No data to process")

if __name__ == "__main__":
    main()
