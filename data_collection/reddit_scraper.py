"""
Reddit Scraper for Female-Male Interaction Data

This script uses PRAW (Python Reddit API Wrapper) to collect conversation data
from relevant subreddits that showcase supportive female-male interactions.
"""

import praw
import pandas as pd
import time
import os
from datetime import datetime

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize Reddit API client
# You'll need to create a Reddit app at https://www.reddit.com/prefs/apps
# and get your client_id and client_secret
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="FriendshipBotDataCollection/1.0 by YourUsername",
    username="YOUR_REDDIT_USERNAME",  # Optional
    password="YOUR_REDDIT_PASSWORD"   # Optional
)

# List of subreddits to scrape
SUBREDDITS = [
    "relationship_advice",
    "askwomen",
    "TwoXChromosomes",
    "dating_advice",
    "CasualConversation",
    "offmychest",
    "MomForAMinute",
    "AskWomenAdvice",
    "supportiveconversations"
]

# Keywords to filter relevant posts
KEYWORDS = [
    "friend", "support", "advice", "help", "talk", "listen", 
    "emotional support", "care", "friendship", "relationship",
    "feeling", "emotion", "comfort", "encourage", "cheer up"
]

def is_relevant_post(post_title, post_text):
    """Check if a post is relevant based on keywords."""
    combined_text = (post_title + " " + post_text).lower()
    return any(keyword.lower() in combined_text for keyword in KEYWORDS)

def is_likely_female_response(comment, author_history=None):
    """
    Attempt to determine if a comment is likely from a female user.
    This is an imperfect heuristic and should be used carefully.
    """
    # Check comment for female indicators
    female_indicators = [
        "as a woman", "as a female", "i'm a woman", "i'm female",
        "i am a woman", "i am female", "f/", "female here"
    ]
    
    if any(indicator in comment.body.lower() for indicator in female_indicators):
        return True
    
    # Check user flair if available
    if hasattr(comment, 'author_flair_text') and comment.author_flair_text:
        if 'female' in comment.author_flair_text.lower() or 'f/' in comment.author_flair_text.lower():
            return True
    
    # If we have author history, check their recent comments
    if author_history:
        female_history_indicators = sum(1 for c in author_history 
                                      if any(indicator in c.body.lower() for indicator in female_indicators))
        if female_history_indicators > 0:
            return True
    
    # If no clear indicators, return None (unknown)
    return None

def collect_data_from_subreddit(subreddit_name, post_limit=100, comment_limit=100):
    """Collect data from a specific subreddit."""
    print(f"Collecting data from r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    
    data = []
    
    # Get top posts from the subreddit
    for post in subreddit.top(time_filter="year", limit=post_limit):
        if not is_relevant_post(post.title, post.selftext):
            continue
            
        print(f"Processing post: {post.title[:50]}...")
        
        # Expand all comments
        post.comments.replace_more(limit=0)
        
        # Process comments
        for comment in post.comments.list()[:comment_limit]:
            if not comment.author:
                continue
                
            # Try to determine if the commenter is female
            is_female = is_likely_female_response(comment)
            
            if is_female or is_female is None:  # Include if female or unknown
                # Get the parent comment/post text
                parent_text = ""
                if comment.parent_id.startswith('t1_'):  # Parent is a comment
                    parent = reddit.comment(id=comment.parent_id[3:])
                    if parent.author:
                        parent_text = parent.body
                else:  # Parent is the post
                    parent_text = post.selftext
                
                # Add to dataset
                data.append({
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'post_title': post.title,
                    'parent_text': parent_text,
                    'comment_text': comment.body,
                    'comment_score': comment.score,
                    'is_female': is_female,
                    'timestamp': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return data

def main():
    """Main function to collect data from all subreddits."""
    all_data = []
    
    for subreddit in SUBREDDITS:
        try:
            subreddit_data = collect_data_from_subreddit(subreddit)
            all_data.extend(subreddit_data)
            print(f"Collected {len(subreddit_data)} comments from r/{subreddit}")
            
            # Be nice to Reddit's servers
            time.sleep(2)
            
        except Exception as e:
            print(f"Error collecting data from r/{subreddit}: {e}")
    
    # Save the data
    df = pd.DataFrame(all_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data/reddit_female_responses_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Data collection complete. Saved {len(df)} records to {filename}")
    print(f"Female responses: {df['is_female'].value_counts().get(True, 0)}")
    print(f"Unknown gender: {df['is_female'].value_counts().get(None, 0)}")

if __name__ == "__main__":
    main()
