"""
Reddit API Integration for Friendship Bot

This module provides integration with Reddit API to gather conversation data
and other information that can be used to enhance the bot's capabilities.
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import PRAW (Python Reddit API Wrapper)
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("PRAW (Python Reddit API Wrapper) is not installed. Reddit functionality will be disabled.")
    print("To enable Reddit functionality, install PRAW: pip install praw")

class RedditAPI:
    """
    A class to interact with the Reddit API for gathering conversation data
    and other information.
    """
    
    def __init__(self):
        """Initialize the Reddit API client."""
        # Check if PRAW is available
        if not PRAW_AVAILABLE:
            self.reddit = None
            self.is_configured = False
            return
        
        # Get Reddit API credentials from environment variables
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT')
        self.username = os.getenv('REDDIT_USERNAME')
        self.password = os.getenv('REDDIT_PASSWORD')
        
        # Check if required credentials are available
        if not self.client_id or not self.client_secret or not self.user_agent:
            self.reddit = None
            self.is_configured = False
            print("Reddit API credentials are missing. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your .env file.")
            return
        
        # Initialize Reddit API client
        try:
            # If username and password are provided, use them for authentication
            if self.username and self.password:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    username=self.username,
                    password=self.password
                )
                print(f"Reddit API initialized with user authentication for u/{self.username}")
            else:
                # Otherwise, use read-only mode
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                print("Reddit API initialized in read-only mode")
            
            self.is_configured = True
        except Exception as e:
            self.reddit = None
            self.is_configured = False
            print(f"Error initializing Reddit API: {e}")
    
    def get_conversations(self, subreddit_name: str, limit: int = 10) -> List[Dict]:
        """
        Get conversations from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit to get conversations from
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of conversations (post + comments)
        """
        if not self.is_configured or not self.reddit:
            print("Reddit API is not configured. Cannot get conversations.")
            return []
        
        conversations = []
        
        try:
            # Get the subreddit
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            for post in subreddit.hot(limit=limit):
                # Skip posts with no comments
                if post.num_comments == 0:
                    continue
                
                # Get post details
                post_data = {
                    "id": post.id,
                    "title": post.title,
                    "content": post.selftext,
                    "author": str(post.author),
                    "score": post.score,
                    "created_utc": post.created_utc,
                    "comments": []
                }
                
                # Get comments (replace more comments to get all comments)
                post.comments.replace_more(limit=0)
                
                # Add comments to post data
                for comment in post.comments.list():
                    comment_data = {
                        "id": comment.id,
                        "content": comment.body,
                        "author": str(comment.author),
                        "score": comment.score,
                        "created_utc": comment.created_utc,
                        "parent_id": comment.parent_id
                    }
                    post_data["comments"].append(comment_data)
                
                conversations.append(post_data)
            
            return conversations
        
        except Exception as e:
            print(f"Error getting conversations from r/{subreddit_name}: {e}")
            return []
    
    def get_female_conversations(self, limit: int = 50) -> List[Dict]:
        """
        Get conversations from female-oriented subreddits.
        
        Args:
            limit: Maximum number of posts to retrieve per subreddit
            
        Returns:
            List of conversations (post + comments)
        """
        if not self.is_configured or not self.reddit:
            print("Reddit API is not configured. Cannot get conversations.")
            return []
        
        # List of female-oriented subreddits
        female_subreddits = [
            "TwoXChromosomes",
            "AskWomen",
            "TheGirlSurvivalGuide",
            "femalefashionadvice",
            "relationships"
        ]
        
        all_conversations = []
        
        for subreddit_name in female_subreddits:
            print(f"Getting conversations from r/{subreddit_name}...")
            conversations = self.get_conversations(subreddit_name, limit=limit)
            all_conversations.extend(conversations)
            
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        return all_conversations
    
    def save_conversations(self, conversations: List[Dict], output_file: str = "data/reddit_conversations.json"):
        """
        Save conversations to a JSON file.
        
        Args:
            conversations: List of conversations to save
            output_file: Path to the output file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save conversations to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        print(f"Saved {len(conversations)} conversations to {output_file}")
    
    def extract_training_data(self, conversations: List[Dict], output_file: str = "data/reddit_training_data.json"):
        """
        Extract training data from conversations.
        
        Args:
            conversations: List of conversations to extract training data from
            output_file: Path to the output file
        """
        training_data = []
        
        for conversation in conversations:
            # Skip posts with no comments
            if not conversation["comments"]:
                continue
            
            # Add post as context
            context = conversation["title"]
            if conversation["content"]:
                context += "\n" + conversation["content"]
            
            # Process comments
            for comment in conversation["comments"]:
                # Skip deleted or removed comments
                if comment["content"] in ["[deleted]", "[removed]"]:
                    continue
                
                # Add comment to training data
                training_example = {
                    "context": context,
                    "response": comment["content"],
                    "metadata": {
                        "subreddit": conversation.get("subreddit", "unknown"),
                        "author": comment["author"],
                        "score": comment["score"]
                    }
                }
                
                training_data.append(training_example)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save training data to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Extracted {len(training_data)} training examples to {output_file}")

# Example usage
if __name__ == "__main__":
    # Initialize Reddit API
    reddit_api = RedditAPI()
    
    if reddit_api.is_configured:
        # Get conversations from a subreddit
        conversations = reddit_api.get_conversations("AskWomen", limit=5)
        
        # Save conversations to file
        reddit_api.save_conversations(conversations)
        
        # Extract training data
        reddit_api.extract_training_data(conversations)
    else:
        print("Reddit API is not configured. Please check your .env file.")
