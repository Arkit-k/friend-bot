"""
Conversation Generator for Communication Techniques

This script generates synthetic conversations that demonstrate the application
of communication techniques from "How to Win Friends and Influence People" and
"How to Talk to Anyone" in realistic scenarios.
"""

import os
import pandas as pd
import json
import random
from tqdm import tqdm
import argparse

# Create necessary directories
os.makedirs('data/books/conversations', exist_ok=True)

# Scenarios for generating conversations
SCENARIOS = [
    {
        "name": "making_friends",
        "description": "Starting a new friendship",
        "male_openers": [
            "I'm new to this area and don't know many people yet.",
            "I've been feeling a bit lonely lately and want to make new friends.",
            "How do you usually meet new people?",
            "I find it hard to connect with new people sometimes.",
            "What's the best way to make friends as an adult?"
        ]
    },
    {
        "name": "feeling_down",
        "description": "Providing emotional support",
        "male_openers": [
            "I've been feeling really down lately.",
            "Work has been really stressful and I'm not coping well.",
            "I don't know why I feel so sad all the time.",
            "I'm going through a tough time right now.",
            "Everything feels overwhelming these days."
        ]
    },
    {
        "name": "relationship_advice",
        "description": "Seeking advice about relationships",
        "male_openers": [
            "I think I messed up with someone I care about.",
            "How do you know if someone is right for you?",
            "I'm having trouble communicating with my partner.",
            "What's the best way to resolve conflicts in a relationship?",
            "I'm not sure if I should continue this relationship."
        ]
    },
    {
        "name": "career_guidance",
        "description": "Seeking career advice",
        "male_openers": [
            "I'm thinking about changing careers but I'm scared.",
            "How do you know when it's time to look for a new job?",
            "I feel stuck in my current position with no growth.",
            "I'm not sure what I want to do with my life career-wise.",
            "How do you balance work ambitions with personal happiness?"
        ]
    },
    {
        "name": "social_anxiety",
        "description": "Dealing with social anxiety",
        "male_openers": [
            "I get really nervous in social situations.",
            "Large groups of people make me anxious.",
            "I worry too much about what others think of me.",
            "How can I be more comfortable in social settings?",
            "I always feel like I'm saying the wrong thing."
        ]
    },
    {
        "name": "confidence_building",
        "description": "Building self-confidence",
        "male_openers": [
            "I wish I could be more confident.",
            "How do you build self-confidence?",
            "I always doubt myself and my abilities.",
            "What's the secret to believing in yourself?",
            "How do you overcome self-doubt?"
        ]
    },
    {
        "name": "small_talk",
        "description": "Making better small talk",
        "male_openers": [
            "I'm terrible at small talk.",
            "How do you keep a conversation going?",
            "I never know what to say to people I just met.",
            "What are good topics for starting conversations?",
            "How do you avoid awkward silences?"
        ]
    },
    {
        "name": "active_listening",
        "description": "Being a better listener",
        "male_openers": [
            "I think I need to be a better listener.",
            "How can I show people I'm really listening to them?",
            "I get distracted easily when others are talking.",
            "What makes someone a good listener?",
            "How do you remember what people tell you?"
        ]
    }
]

# Follow-up questions from the male user
FOLLOW_UP_QUESTIONS = [
    "Can you give me an example of how to do that?",
    "That's interesting. How would that work in practice?",
    "I've never thought about it that way. Can you explain more?",
    "What if that doesn't work for me?",
    "That sounds difficult. How do I start?",
    "What's the most important thing to remember about this?",
    "Have you tried this approach yourself?",
    "How do I know if I'm doing it right?",
    "What's a common mistake people make with this?",
    "How long does it take to get good at this?"
]

# Gratitude and closing statements from the male user
GRATITUDE_STATEMENTS = [
    "Thank you, that's really helpful advice.",
    "I appreciate you taking the time to explain this to me.",
    "That makes a lot of sense. Thanks for your help.",
    "I'll definitely try that approach. Thanks!",
    "You've given me a lot to think about. Thank you.",
    "That's exactly what I needed to hear.",
    "I feel much better about this now. Thank you.",
    "Your advice is always so practical. Thanks!",
    "I'm grateful for your perspective on this.",
    "Thanks for being such a supportive friend."
]

def load_techniques(file_path):
    """
    Load communication techniques from JSON file.
    
    Args:
        file_path: Path to the techniques JSON file
        
    Returns:
        Dictionary of techniques
    """
    if not os.path.exists(file_path):
        print(f"Techniques file {file_path} not found")
        return {}
    
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_conversation(scenario, techniques, num_exchanges=3):
    """
    Generate a synthetic conversation for a scenario using communication techniques.
    
    Args:
        scenario: Dictionary with scenario information
        techniques: Dictionary of communication techniques
        num_exchanges: Number of exchanges in the conversation
        
    Returns:
        List of dictionaries with conversation turns
    """
    conversation = []
    
    # Start with a male opener
    male_opener = random.choice(scenario["male_openers"])
    conversation.append({
        "speaker": "male",
        "text": male_opener,
        "is_female_response": False
    })
    
    # Select random techniques to demonstrate
    all_techniques = []
    for category, technique_dict in techniques.items():
        for technique, examples in technique_dict.items():
            all_techniques.append({
                "category": category,
                "technique": technique,
                "examples": examples
            })
    
    selected_techniques = random.sample(all_techniques, min(num_exchanges, len(all_techniques)))
    
    # Generate the conversation exchanges
    for i, technique_info in enumerate(selected_techniques):
        category = technique_info["category"]
        technique = technique_info["technique"]
        examples = technique_info["examples"]
        
        # Female response demonstrating the technique
        if examples:
            # Use an example response or create a generic one
            example = random.choice(examples)
            female_response = example.get("response", "")
        else:
            female_response = f"I understand how you feel. {technique} is really helpful in this situation. It's about focusing on the other person and building connection through {category.replace('_', ' ')}."
        
        conversation.append({
            "speaker": "female",
            "text": female_response,
            "is_female_response": True,
            "technique": technique,
            "category": category
        })
        
        # Male follow-up, except for the last exchange
        if i < len(selected_techniques) - 1:
            male_follow_up = random.choice(FOLLOW_UP_QUESTIONS)
            conversation.append({
                "speaker": "male",
                "text": male_follow_up,
                "is_female_response": False
            })
    
    # End with a gratitude statement from the male user
    conversation.append({
        "speaker": "male",
        "text": random.choice(GRATITUDE_STATEMENTS),
        "is_female_response": False
    })
    
    return conversation

def create_conversation_pairs(conversations):
    """
    Create conversation pairs for training from generated conversations.
    
    Args:
        conversations: List of generated conversations
        
    Returns:
        DataFrame with conversation pairs
    """
    pairs = []
    
    for conversation in conversations:
        scenario = conversation["scenario"]
        exchanges = conversation["exchanges"]
        
        for i in range(len(exchanges) - 1):
            current = exchanges[i]
            next_turn = exchanges[i + 1]
            
            # Only create pairs where the response is from the female bot
            if not current["is_female_response"] and next_turn["is_female_response"]:
                pair = {
                    "utterance": current["text"],
                    "response": next_turn["text"],
                    "emotion_label": "neutral",  # Default emotion
                    "dataset": "synthetic_conversation",
                    "is_female_response": True,
                    "scenario": scenario,
                    "technique": next_turn.get("technique", ""),
                    "category": next_turn.get("category", "")
                }
                pairs.append(pair)
    
    return pd.DataFrame(pairs)

def main():
    """Main function to generate synthetic conversations."""
    parser = argparse.ArgumentParser(description='Generate synthetic conversations demonstrating communication techniques')
    parser.add_argument('--techniques_file', default='data/books/techniques/techniques_catalog.json', help='JSON file with communication techniques')
    parser.add_argument('--output_file', default='data/books/conversations/synthetic_conversations.csv', help='Output CSV file')
    parser.add_argument('--num_conversations', type=int, default=50, help='Number of conversations to generate per scenario')
    parser.add_argument('--exchanges_per_conversation', type=int, default=3, help='Number of exchanges per conversation')
    args = parser.parse_args()
    
    # Load communication techniques
    techniques = load_techniques(args.techniques_file)
    
    if not techniques:
        print("No techniques found. Please run extract_communication_techniques.py first.")
        return
    
    # Generate conversations for each scenario
    all_conversations = []
    
    for scenario in tqdm(SCENARIOS, desc="Generating conversations"):
        for _ in range(args.num_conversations):
            conversation = {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "exchanges": generate_conversation(scenario, techniques, args.exchanges_per_conversation)
            }
            all_conversations.append(conversation)
    
    print(f"Generated {len(all_conversations)} conversations")
    
    # Create conversation pairs for training
    pairs_df = create_conversation_pairs(all_conversations)
    
    # Save the conversation pairs
    pairs_df.to_csv(args.output_file, index=False)
    
    # Also save as JSON for easier inspection
    json_output = args.output_file.replace('.csv', '.json')
    
    # Save the full conversations as JSON
    with open(json_output, 'w') as f:
        json.dump(all_conversations, f, indent=2)
    
    print(f"Created {len(pairs_df)} conversation pairs")
    print(f"Data saved to {args.output_file} and {json_output}")

if __name__ == "__main__":
    main()
