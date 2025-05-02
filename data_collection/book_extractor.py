"""
Book Data Extractor for Friendship Bot

This script extracts and processes information from books about female psychology,
communication patterns, and relationship dynamics to enhance the training data.
"""

import os
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs('data/books', exist_ok=True)

# Download NLTK resources
nltk.download('punkt')

def extract_sentences_from_text(text_file, min_length=20, max_length=200):
    """
    Extract sentences from a text file.
    
    Args:
        text_file: Path to the text file
        min_length: Minimum sentence length to include
        max_length: Maximum sentence length to include
        
    Returns:
        List of extracted sentences
    """
    print(f"Extracting sentences from {text_file}...")
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(text_file, 'r', encoding='latin-1') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file {text_file}: {e}")
            return []
    
    # Clean the text
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Filter sentences by length
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if min_length <= len(sentence) <= max_length:
            filtered_sentences.append(sentence)
    
    print(f"Extracted {len(filtered_sentences)} sentences from {text_file}")
    return filtered_sentences

def extract_relevant_sentences(sentences, keywords, context_size=2):
    """
    Extract sentences containing relevant keywords and their surrounding context.
    
    Args:
        sentences: List of sentences
        keywords: List of keywords to search for
        context_size: Number of sentences before and after to include
        
    Returns:
        List of relevant sentence blocks
    """
    relevant_indices = []
    
    # Find sentences containing keywords
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if any(keyword.lower() in sentence_lower for keyword in keywords):
            relevant_indices.append(i)
    
    # Add context sentences
    context_indices = set()
    for idx in relevant_indices:
        for j in range(max(0, idx - context_size), min(len(sentences), idx + context_size + 1)):
            context_indices.add(j)
    
    # Extract the relevant blocks
    relevant_blocks = []
    current_block = []
    
    for i in range(len(sentences)):
        if i in context_indices:
            current_block.append(sentences[i])
        elif current_block:
            relevant_blocks.append(' '.join(current_block))
            current_block = []
    
    # Add the last block if it exists
    if current_block:
        relevant_blocks.append(' '.join(current_block))
    
    return relevant_blocks

def create_conversation_pairs(text_blocks, max_length=100):
    """
    Create conversation pairs from text blocks.
    
    Args:
        text_blocks: List of text blocks
        max_length: Maximum length of each part of the pair
        
    Returns:
        DataFrame with utterance-response pairs
    """
    pairs = []
    
    for block in text_blocks:
        sentences = sent_tokenize(block)
        
        if len(sentences) < 2:
            continue
        
        # Create pairs from consecutive sentences
        for i in range(len(sentences) - 1):
            utterance = sentences[i][:max_length]
            response = sentences[i + 1][:max_length]
            
            # Skip if either part is too short
            if len(utterance) < 10 or len(response) < 10:
                continue
                
            pairs.append({
                'utterance': utterance,
                'response': response,
                'emotion_label': 'neutral',  # Default label
                'dataset': 'book',
                'is_female_response': True  # Assuming these are female perspectives
            })
    
    return pd.DataFrame(pairs)

def process_book_file(file_path, keywords):
    """
    Process a single book file.
    
    Args:
        file_path: Path to the book file
        keywords: List of keywords to search for
        
    Returns:
        DataFrame with conversation pairs
    """
    # Extract sentences
    sentences = extract_sentences_from_text(file_path)
    
    if not sentences:
        return None
    
    # Extract relevant blocks
    relevant_blocks = extract_relevant_sentences(sentences, keywords)
    
    print(f"Extracted {len(relevant_blocks)} relevant blocks from {file_path}")
    
    # Create conversation pairs
    pairs_df = create_conversation_pairs(relevant_blocks)
    
    print(f"Created {len(pairs_df)} conversation pairs from {file_path}")
    
    return pairs_df

def main():
    """Main function to process book files."""
    parser = argparse.ArgumentParser(description='Extract data from books for training the friendship bot')
    parser.add_argument('--input_dir', default='data/books/raw', help='Directory containing book text files')
    parser.add_argument('--output_file', default='data/books/book_conversations.csv', help='Output CSV file')
    args = parser.parse_args()
    
    # Create input directory if it doesn't exist
    os.makedirs(args.input_dir, exist_ok=True)
    
    # Keywords related to female communication, psychology, and relationships
    keywords = [
        "women", "woman", "female", "girl", "friendship", "relationship", "communication",
        "emotional support", "empathy", "listening", "understanding", "caring",
        "nurturing", "compassion", "validation", "feelings", "emotion", "support",
        "connection", "bonding", "trust", "intimacy", "vulnerability", "sharing",
        "psychology", "feminine", "perspective", "viewpoint", "experience"
    ]
    
    # Check if there are any files in the input directory
    book_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                 if os.path.isfile(os.path.join(args.input_dir, f)) and f.endswith(('.txt', '.text'))]
    
    if not book_files:
        print(f"No book files found in {args.input_dir}")
        print("Please add .txt files of books about female psychology, communication, etc.")
        return
    
    # Process each book file
    all_pairs = []
    
    for file_path in tqdm(book_files, desc="Processing books"):
        pairs_df = process_book_file(file_path, keywords)
        if pairs_df is not None and not pairs_df.empty:
            all_pairs.append(pairs_df)
    
    if not all_pairs:
        print("No conversation pairs were extracted from the books")
        return
    
    # Combine all pairs
    combined_df = pd.concat(all_pairs, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['utterance', 'response'])
    
    # Save to CSV
    combined_df.to_csv(args.output_file, index=False)
    
    print(f"Extracted {len(combined_df)} unique conversation pairs from {len(book_files)} books")
    print(f"Data saved to {args.output_file}")

if __name__ == "__main__":
    main()
