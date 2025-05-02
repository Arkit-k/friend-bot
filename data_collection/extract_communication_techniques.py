"""
Communication Techniques Extractor

This script extracts communication techniques and principles from specific books
on interpersonal communication and formats them for training the friendship bot.

Books targeted:
1. "How to Win Friends and Influence People" by Dale Carnegie
2. "How to Talk to Anyone" by Leil Lowndes
"""

import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from tqdm import tqdm
import json

# Create necessary directories
os.makedirs('data/books/techniques', exist_ok=True)

# Download NLTK resources
nltk.download('punkt', quiet=True)

# Define the techniques we want to extract from each book
CARNEGIE_TECHNIQUES = {
    "fundamental_techniques": [
        "Don't criticize, condemn or complain",
        "Give honest and sincere appreciation",
        "Arouse in the other person an eager want",
        "criticism", "appreciation", "interest"
    ],
    "make_people_like_you": [
        "Become genuinely interested in other people",
        "Smile",
        "Remember that a person's name is to that person the sweetest and most important sound",
        "Be a good listener",
        "Talk in terms of the other person's interests",
        "Make the other person feel important",
        "interest", "smile", "name", "listen", "importance"
    ],
    "win_people_to_your_way": [
        "The only way to get the best of an argument is to avoid it",
        "Show respect for the other person's opinions",
        "If you are wrong, admit it quickly and emphatically",
        "Begin in a friendly way",
        "Get the other person saying 'yes, yes' immediately",
        "Let the other person do a great deal of the talking",
        "Let the other person feel that the idea is his or hers",
        "Try honestly to see things from the other person's point of view",
        "Be sympathetic with the other person's ideas and desires",
        "Appeal to the nobler motives",
        "Dramatize your ideas",
        "Throw down a challenge",
        "argument", "respect", "admit", "friendly", "agreement", "talking", "ownership", "perspective", "sympathy", "motives"
    ],
    "be_a_leader": [
        "Begin with praise and honest appreciation",
        "Call attention to people's mistakes indirectly",
        "Talk about your own mistakes before criticizing the other person",
        "Ask questions instead of giving direct orders",
        "Let the other person save face",
        "Praise the slightest improvement",
        "Give the other person a fine reputation to live up to",
        "Use encouragement",
        "Make the fault easy to correct",
        "Make the other person happy about doing the thing you suggest",
        "praise", "indirect", "humility", "questions", "face", "improvement", "reputation", "encouragement", "correction", "happiness"
    ]
}

LOWNDES_TECHNIQUES = {
    "first_impressions": [
        "The Flooding Smile",
        "Sticky Eyes",
        "Epoxy Eyes",
        "Hang by Your Teeth",
        "The Big-Baby Pivot",
        "Hello Old Friend",
        "Limit the Fidget",
        "Hans's Horse Sense",
        "Watch the Scene Before You Make the Scene",
        "Always Wear a Whatzit",
        "smile", "eye contact", "posture", "fidgeting", "observation", "accessory"
    ],
    "small_talk": [
        "Be the Chooser, Not the Choosee",
        "The Swiveling Spotlight",
        "Parroting",
        "Encore!",
        "Never the Naked Thank You",
        "Scramble Therapy",
        "Ac-cen-tu-ate the Positive",
        "The Latest News . . . Don't Leave Home Without It",
        "The Exclusive Smile",
        "Don't Touch a Cliché with a Ten-Foot Pole",
        "initiative", "spotlight", "repeat", "gratitude", "positive", "news", "smile", "originality"
    ],
    "conversation": [
        "The Attractor Factor",
        "Comm-YOU-nication",
        "The Business Card Doodle",
        "Instant History",
        "Prosaic, Not Poetic",
        "Baring Their Hot Button",
        "Bare the Buried WIIFM (and WIIFY)",
        "Read Their Pulse",
        "Listening Between the Lines",
        "Volley the Conversation",
        "The Emotional Prediction",
        "interest", "you-focus", "personalization", "history", "simplicity", "hot button", "benefit", "pulse", "listening", "volley", "emotion"
    ],
    "rapport": [
        "Echoing",
        "Potent Imaging",
        "Employ Empathizers",
        "Anatomically Correct Empathizers",
        "The Premature We",
        "Instant History",
        "Accidental Touch",
        "How to Knife a Tomato",
        "Tracking",
        "The Mood Match",
        "Pacing",
        "mirroring", "imagery", "empathy", "we", "history", "touch", "expertise", "tracking", "mood", "pace"
    ]
}

def extract_techniques_from_text(text, techniques_dict, book_title):
    """
    Extract passages related to communication techniques from text.
    
    Args:
        text: The book text
        techniques_dict: Dictionary of technique categories and keywords
        book_title: Title of the book
        
    Returns:
        DataFrame with technique examples
    """
    print(f"Extracting techniques from {book_title}...")
    
    # Clean the text
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    print(f"Total sentences: {len(sentences)}")
    
    # Extract relevant passages
    technique_examples = []
    
    for category, keywords in techniques_dict.items():
        print(f"Processing category: {category}")
        
        for i, sentence in enumerate(tqdm(sentences)):
            sentence_lower = sentence.lower()
            
            # Check if the sentence contains any of the keywords
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                # Get context (sentences before and after)
                start_idx = max(0, i - 2)
                end_idx = min(len(sentences), i + 3)
                context = " ".join(sentences[start_idx:end_idx])
                
                # Determine which specific technique this relates to
                matching_techniques = [
                    technique for technique in keywords 
                    if len(technique) > 10 and technique.lower() in context.lower()
                ]
                
                technique_name = matching_techniques[0] if matching_techniques else category
                
                technique_examples.append({
                    'book': book_title,
                    'category': category,
                    'technique': technique_name,
                    'context': context,
                    'utterance': sentences[i],
                    'response': sentences[i+1] if i+1 < len(sentences) else "",
                    'is_female_response': True,  # Assuming the bot will use these as female responses
                    'emotion_label': 'neutral'
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(technique_examples)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['context'])
    
    print(f"Extracted {len(df)} technique examples from {book_title}")
    return df

def create_conversation_pairs(df):
    """
    Create conversation pairs from technique examples.
    
    Args:
        df: DataFrame with technique examples
        
    Returns:
        DataFrame with conversation pairs
    """
    conversation_pairs = []
    
    for _, row in df.iterrows():
        # Create a conversation pair from the utterance and response
        if row['utterance'] and row['response']:
            conversation_pairs.append({
                'utterance': row['utterance'],
                'response': row['response'],
                'emotion_label': 'neutral',
                'dataset': f"book_{row['book']}",
                'is_female_response': True,
                'technique': row['technique'],
                'category': row['category']
            })
        
        # Create technique explanation pairs
        technique_explanation = f"How do I {row['technique'].lower()}?"
        technique_response = row['context']
        
        conversation_pairs.append({
            'utterance': technique_explanation,
            'response': technique_response,
            'emotion_label': 'neutral',
            'dataset': f"book_{row['book']}",
            'is_female_response': True,
            'technique': row['technique'],
            'category': row['category']
        })
        
        # Create application examples
        application_question = f"Can you give me an example of {row['technique'].lower()}?"
        application_response = row['context']
        
        conversation_pairs.append({
            'utterance': application_question,
            'response': application_response,
            'emotion_label': 'neutral',
            'dataset': f"book_{row['book']}",
            'is_female_response': True,
            'technique': row['technique'],
            'category': row['category']
        })
    
    return pd.DataFrame(conversation_pairs)

def create_technique_summaries(techniques_dict, book_title):
    """
    Create summaries of techniques for training.
    
    Args:
        techniques_dict: Dictionary of technique categories and keywords
        book_title: Title of the book
        
    Returns:
        DataFrame with technique summaries
    """
    summaries = []
    
    for category, techniques in techniques_dict.items():
        # Only use the actual techniques (longer strings), not the keywords
        actual_techniques = [t for t in techniques if len(t) > 10]
        
        for technique in actual_techniques:
            # Create different types of questions about this technique
            questions = [
                f"What is {technique}?",
                f"How do I {technique.lower()}?",
                f"Can you explain {technique}?",
                f"What does {technique} mean?",
                f"How can I use {technique} in a conversation?"
            ]
            
            # Create a generic response about the technique
            response = f"{technique} is an important communication principle from {book_title}. " \
                      f"It involves focusing on {category.replace('_', ' ')} to build better relationships. " \
                      f"When you practice this technique, you'll connect better with others and create more meaningful interactions."
            
            for question in questions:
                summaries.append({
                    'utterance': question,
                    'response': response,
                    'emotion_label': 'neutral',
                    'dataset': f"book_{book_title}",
                    'is_female_response': True,
                    'technique': technique,
                    'category': category
                })
    
    return pd.DataFrame(summaries)

def process_book_file(file_path, book_title):
    """
    Process a book file to extract communication techniques.
    
    Args:
        file_path: Path to the book file
        book_title: Title of the book
        
    Returns:
        DataFrame with conversation pairs based on techniques
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    # Determine which techniques dictionary to use based on the book title
    if "win friends" in book_title.lower():
        techniques_dict = CARNEGIE_TECHNIQUES
    elif "talk to anyone" in book_title.lower():
        techniques_dict = LOWNDES_TECHNIQUES
    else:
        print(f"Unknown book: {book_title}. Using generic technique extraction.")
        techniques_dict = {**CARNEGIE_TECHNIQUES, **LOWNDES_TECHNIQUES}
    
    # Extract technique examples
    examples_df = extract_techniques_from_text(text, techniques_dict, book_title)
    
    # Create conversation pairs
    conversation_pairs = create_conversation_pairs(examples_df)
    
    # Create technique summaries
    technique_summaries = create_technique_summaries(techniques_dict, book_title)
    
    # Combine conversation pairs and technique summaries
    combined_df = pd.concat([conversation_pairs, technique_summaries], ignore_index=True)
    
    # Save the raw examples for reference
    examples_df.to_csv(f"data/books/techniques/{book_title.replace(' ', '_').lower()}_examples.csv", index=False)
    
    return combined_df

def create_technique_application_examples():
    """
    Create examples of how to apply communication techniques in various scenarios.
    
    Returns:
        DataFrame with application examples
    """
    application_examples = []
    
    # Scenarios where communication techniques can be applied
    scenarios = [
        "making friends",
        "resolving a conflict",
        "networking at an event",
        "starting a conversation with a stranger",
        "deepening a friendship",
        "showing empathy to someone who is sad",
        "giving constructive feedback",
        "asking for help",
        "expressing disagreement respectfully",
        "showing appreciation",
        "active listening",
        "building rapport",
        "making small talk",
        "having a difficult conversation",
        "connecting with someone new"
    ]
    
    # Combine all techniques from both books
    all_techniques = {}
    for category, techniques in {**CARNEGIE_TECHNIQUES, **LOWNDES_TECHNIQUES}.items():
        all_techniques[category] = [t for t in techniques if len(t) > 10]
    
    # Create application examples for each scenario and technique
    for scenario in scenarios:
        for category, techniques in all_techniques.items():
            for technique in techniques:
                # Create a question about applying this technique in this scenario
                question = f"How can I use {technique} when {scenario}?"
                
                # Create a response with advice
                response = f"When {scenario}, you can apply {technique} by focusing on the other person's perspective. " \
                          f"This technique helps build connection by emphasizing {category.replace('_', ' ')}. " \
                          f"For example, you might want to ask open-ended questions, show genuine interest, and respond thoughtfully. " \
                          f"Remember that authentic communication is about creating a safe space for sharing and understanding."
                
                application_examples.append({
                    'utterance': question,
                    'response': response,
                    'emotion_label': 'neutral',
                    'dataset': "book_application_examples",
                    'is_female_response': True,
                    'technique': technique,
                    'category': category,
                    'scenario': scenario
                })
    
    return pd.DataFrame(application_examples)

def main():
    """Main function to extract communication techniques from books."""
    parser = argparse.ArgumentParser(description='Extract communication techniques from books')
    parser.add_argument('--input_dir', default='data/books/raw', help='Directory containing book text files')
    parser.add_argument('--output_file', default='data/books/communication_techniques.csv', help='Output CSV file')
    args = parser.parse_args()
    
    # Check if the input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist. Creating it...")
        os.makedirs(args.input_dir, exist_ok=True)
        print(f"Please add the book files to {args.input_dir} and run this script again.")
        
        # Create a README file with instructions
        readme_path = os.path.join(args.input_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("Please add the following book files to this directory:\n\n")
            f.write("1. 'How to Win Friends and Influence People' by Dale Carnegie\n")
            f.write("2. 'How to Talk to Anyone' by Leil Lowndes\n\n")
            f.write("The files should be in plain text (.txt) format.\n")
        
        return
    
    # Look for the specific books we want to process
    book_files = []
    
    for filename in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            lower_filename = filename.lower()
            
            if "win" in lower_filename and "friend" in lower_filename:
                book_files.append((file_path, "How to Win Friends and Influence People"))
            elif "talk" in lower_filename and "anyone" in lower_filename:
                book_files.append((file_path, "How to Talk to Anyone"))
    
    if not book_files:
        print(f"No relevant book files found in {args.input_dir}")
        print("Please add the book files with names containing 'win friends' or 'talk to anyone'")
        return
    
    # Process each book file
    all_pairs = []
    
    for file_path, book_title in book_files:
        pairs_df = process_book_file(file_path, book_title)
        if pairs_df is not None:
            all_pairs.append(pairs_df)
    
    # Create application examples
    application_df = create_technique_application_examples()
    all_pairs.append(application_df)
    
    # Combine all pairs
    if all_pairs:
        combined_df = pd.concat(all_pairs, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['utterance', 'response'])
        
        # Save to CSV
        combined_df.to_csv(args.output_file, index=False)
        
        # Also save as JSON for easier inspection
        json_output = args.output_file.replace('.csv', '.json')
        combined_df.to_json(json_output, orient='records', indent=2)
        
        print(f"Extracted {len(combined_df)} conversation pairs from {len(book_files)} books")
        print(f"Data saved to {args.output_file} and {json_output}")
        
        # Create a techniques catalog
        create_techniques_catalog(combined_df)
    else:
        print("No conversation pairs were extracted from the books")

def create_techniques_catalog(df):
    """
    Create a catalog of all communication techniques for reference.
    
    Args:
        df: DataFrame with technique examples
    """
    techniques = {}
    
    for _, row in df.iterrows():
        category = row['category']
        technique = row['technique']
        
        if category not in techniques:
            techniques[category] = {}
        
        if technique not in techniques[category]:
            techniques[category][technique] = []
        
        if len(techniques[category][technique]) < 3:  # Limit to 3 examples per technique
            example = {
                'utterance': row['utterance'],
                'response': row['response']
            }
            techniques[category][technique].append(example)
    
    # Save the catalog as JSON
    with open('data/books/techniques/techniques_catalog.json', 'w') as f:
        json.dump(techniques, f, indent=2)
    
    print(f"Created techniques catalog with {len(techniques)} categories")

if __name__ == "__main__":
    main()
