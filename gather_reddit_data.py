"""
Script to gather conversation data from Reddit.

This script uses the Reddit API to gather conversation data from various subreddits
that can be used to train or fine-tune the friendship bot.
"""

import os
import argparse
import json
from reddit_api import RedditAPI

def main():
    """Main function to gather Reddit data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gather conversation data from Reddit")
    parser.add_argument("--subreddits", nargs="+", default=["AskWomen", "TwoXChromosomes", "relationships", "dating_advice"],
                        help="List of subreddits to gather data from")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum number of posts to retrieve per subreddit")
    parser.add_argument("--output", type=str, default="data/reddit_conversations.json",
                        help="Path to the output file for raw conversations")
    parser.add_argument("--training-output", type=str, default="data/reddit_training_data.json",
                        help="Path to the output file for training data")
    parser.add_argument("--female-only", action="store_true",
                        help="Only gather data from female-oriented subreddits")
    
    args = parser.parse_args()
    
    # Initialize Reddit API
    reddit_api = RedditAPI()
    
    if not reddit_api.is_configured:
        print("Reddit API is not configured. Please check your .env file.")
        return
    
    # Gather conversations
    all_conversations = []
    
    if args.female_only:
        print("Gathering conversations from female-oriented subreddits...")
        all_conversations = reddit_api.get_female_conversations(limit=args.limit)
    else:
        for subreddit in args.subreddits:
            print(f"Gathering conversations from r/{subreddit}...")
            conversations = reddit_api.get_conversations(subreddit, limit=args.limit)
            
            # Add subreddit name to each conversation
            for conversation in conversations:
                conversation["subreddit"] = subreddit
            
            all_conversations.extend(conversations)
    
    # Save raw conversations
    if all_conversations:
        reddit_api.save_conversations(all_conversations, output_file=args.output)
        
        # Extract and save training data
        reddit_api.extract_training_data(all_conversations, output_file=args.training_output)
        
        print(f"Gathered {len(all_conversations)} conversations from Reddit.")
    else:
        print("No conversations were gathered. Please check your Reddit API credentials and subreddit names.")

if __name__ == "__main__":
    main()
