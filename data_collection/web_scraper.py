"""
Web scraper for collecting conversation data from various online sources.
This module provides functions to scrape data from forums, Q&A sites, and other
conversation-rich websites to enhance the bot's training data.
"""

import os
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection/scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_scraper")

# User agents to rotate through to avoid being blocked
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

class WebScraper:
    """Class for scraping conversation data from websites."""
    
    def __init__(self, output_dir="data/scraped"):
        """Initialize the scraper with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session = requests.Session()
    
    def _get_random_user_agent(self):
        """Get a random user agent to avoid detection."""
        return random.choice(USER_AGENTS)
    
    def _make_request(self, url, max_retries=3):
        """Make an HTTP request with retries and random user agent."""
        headers = {'User-Agent': self._get_random_user_agent()}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to retrieve {url} after {max_retries} attempts")
                    return None
                time.sleep(2 * (attempt + 1))  # Exponential backoff
    
    def scrape_quora(self, topics=None, max_questions=50):
        """
        Scrape conversation data from Quora.
        
        Args:
            topics: List of topics to scrape
            max_questions: Maximum number of questions to scrape per topic
            
        Returns:
            List of conversation dictionaries
        """
        if topics is None:
            topics = ["relationships", "dating", "friendship", "communication", "psychology"]
        
        all_conversations = []
        
        for topic in topics:
            logger.info(f"Scraping Quora topic: {topic}")
            topic_url = f"https://www.quora.com/topic/{topic}"
            
            response = self._make_request(topic_url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find question links
            question_links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/') and not href.startswith('/topic/') and '?' not in href:
                    question_links.append(urljoin("https://www.quora.com", href))
            
            # Limit the number of questions
            question_links = question_links[:max_questions]
            
            # Process each question
            for question_url in question_links:
                logger.info(f"Scraping question: {question_url}")
                
                # Add a delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
                response = self._make_request(question_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract question title
                question_title = ""
                title_elem = soup.find('title')
                if title_elem:
                    question_title = title_elem.text.replace(" - Quora", "").strip()
                
                # Extract answers
                answers = []
                answer_elements = soup.select('.q-box.spacing_log_answer_content')
                
                for answer_elem in answer_elements:
                    answer_text = answer_elem.get_text(strip=True)
                    if answer_text:
                        answers.append(answer_text)
                
                # Create conversation object
                if question_title and answers:
                    conversation = {
                        "source": "quora",
                        "topic": topic,
                        "question": question_title,
                        "answers": answers,
                        "url": question_url
                    }
                    all_conversations.append(conversation)
        
        # Save the scraped data
        output_file = os.path.join(self.output_dir, "quora_conversations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)
        
        logger.info(f"Saved {len(all_conversations)} Quora conversations to {output_file}")
        return all_conversations
    
    def scrape_relationship_advice(self, max_pages=5, posts_per_page=25):
        """
        Scrape conversation data from relationship_advice subreddit.
        
        Args:
            max_pages: Maximum number of pages to scrape
            posts_per_page: Number of posts per page
            
        Returns:
            List of conversation dictionaries
        """
        all_conversations = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"Scraping relationship_advice page {page}")
            
            # Use old.reddit.com which is easier to scrape
            url = f"https://old.reddit.com/r/relationship_advice/?count={(page-1)*posts_per_page}&after=t3_{page-1}"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find post links
            post_links = []
            for a_tag in soup.select('.title.may-blank'):
                if 'href' in a_tag.attrs:
                    href = a_tag['href']
                    if href.startswith('/r/relationship_advice/comments/'):
                        post_links.append(urljoin("https://old.reddit.com", href))
            
            # Process each post
            for post_url in post_links:
                logger.info(f"Scraping post: {post_url}")
                
                # Add a delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
                response = self._make_request(post_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract post title
                post_title = ""
                title_elem = soup.select_one('.title')
                if title_elem:
                    post_title = title_elem.text.strip()
                
                # Extract post content
                post_content = ""
                selftext_elem = soup.select_one('.usertext-body')
                if selftext_elem:
                    post_content = selftext_elem.get_text(strip=True)
                
                # Extract comments
                comments = []
                comment_elements = soup.select('.usertext-body')
                
                # Skip the first one (post content)
                for comment_elem in comment_elements[1:]:
                    comment_text = comment_elem.get_text(strip=True)
                    if comment_text and len(comment_text) > 20:  # Filter out very short comments
                        comments.append(comment_text)
                
                # Create conversation object
                if post_title and comments:
                    conversation = {
                        "source": "reddit",
                        "subreddit": "relationship_advice",
                        "title": post_title,
                        "content": post_content,
                        "comments": comments,
                        "url": post_url
                    }
                    all_conversations.append(conversation)
        
        # Save the scraped data
        output_file = os.path.join(self.output_dir, "relationship_advice_conversations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)
        
        logger.info(f"Saved {len(all_conversations)} relationship_advice conversations to {output_file}")
        return all_conversations
    
    def scrape_ask_women(self, max_pages=5, posts_per_page=25):
        """
        Scrape conversation data from AskWomen subreddit.
        
        Args:
            max_pages: Maximum number of pages to scrape
            posts_per_page: Number of posts per page
            
        Returns:
            List of conversation dictionaries
        """
        all_conversations = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"Scraping AskWomen page {page}")
            
            # Use old.reddit.com which is easier to scrape
            url = f"https://old.reddit.com/r/AskWomen/?count={(page-1)*posts_per_page}&after=t3_{page-1}"
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find post links
            post_links = []
            for a_tag in soup.select('.title.may-blank'):
                if 'href' in a_tag.attrs:
                    href = a_tag['href']
                    if href.startswith('/r/AskWomen/comments/'):
                        post_links.append(urljoin("https://old.reddit.com", href))
            
            # Process each post
            for post_url in post_links:
                logger.info(f"Scraping post: {post_url}")
                
                # Add a delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
                response = self._make_request(post_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract post title
                post_title = ""
                title_elem = soup.select_one('.title')
                if title_elem:
                    post_title = title_elem.text.strip()
                
                # Extract comments
                comments = []
                comment_elements = soup.select('.usertext-body')
                
                # Skip the first one if it's the post content
                for comment_elem in comment_elements[1:]:
                    comment_text = comment_elem.get_text(strip=True)
                    if comment_text and len(comment_text) > 20:  # Filter out very short comments
                        comments.append(comment_text)
                
                # Create conversation object
                if post_title and comments:
                    conversation = {
                        "source": "reddit",
                        "subreddit": "AskWomen",
                        "title": post_title,
                        "comments": comments,
                        "url": post_url
                    }
                    all_conversations.append(conversation)
        
        # Save the scraped data
        output_file = os.path.join(self.output_dir, "askwomen_conversations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)
        
        logger.info(f"Saved {len(all_conversations)} AskWomen conversations to {output_file}")
        return all_conversations
    
    def scrape_all(self):
        """Scrape data from all supported sources."""
        quora_data = self.scrape_quora()
        relationship_advice_data = self.scrape_relationship_advice()
        askwomen_data = self.scrape_ask_women()
        
        # Combine all data
        all_data = quora_data + relationship_advice_data + askwomen_data
        
        # Save combined data
        output_file = os.path.join(self.output_dir, "all_scraped_conversations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2)
        
        logger.info(f"Saved {len(all_data)} total conversations to {output_file}")
        return all_data

# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrape_all()
