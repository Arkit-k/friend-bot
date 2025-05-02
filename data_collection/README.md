# Data Collection for Friendship Bot

This directory contains scripts for collecting, processing, and preparing data to train the Friendship Bot model. The goal is to gather authentic female-male interactions to make the bot's responses more natural and supportive.

## Overview

The data collection process involves:

1. **Gathering data from Reddit** - Collecting supportive female responses from relevant subreddits
2. **Downloading datasets from Hugging Face** - Using existing conversation datasets with emotional context
3. **Extracting data from books** - Gathering insights from books about women's psychology and communication
4. **Extracting communication techniques** - Analyzing specific books on interpersonal skills
5. **Generating synthetic conversations** - Creating examples that demonstrate communication techniques
6. **Processing and cleaning the data** - Preparing the data for model training
7. **Training the emotion detection model** - Building a model that can detect emotions in text

## Prerequisites

Before running the scripts, install the required dependencies:

```bash
pip install praw pandas datasets nltk emoji scikit-learn tensorflow matplotlib seaborn tqdm beautifulsoup4 requests
```

## Scripts

### 1. Reddit Scraper (`reddit_scraper.py`)

Collects supportive female responses from Reddit using the PRAW API.

**Setup:**
1. Create a Reddit app at https://www.reddit.com/prefs/apps
2. Get your `client_id` and `client_secret`
3. Update the script with your credentials

**Usage:**
```bash
python reddit_scraper.py
```

### 2. Hugging Face Dataset Downloader (`huggingface_datasets.py`)

Downloads and processes conversation datasets from Hugging Face.

**Usage:**
```bash
python huggingface_datasets.py
```

### 3. Book Downloader (`download_books.py`)

Downloads public domain books related to women's psychology and communication from Project Gutenberg.

**Usage:**
```bash
python download_books.py
# To search for additional books:
python download_books.py --search
```

### 4. Book Data Extractor (`book_extractor.py`)

Extracts relevant content from downloaded books and creates conversation pairs.

**Usage:**
```bash
python book_extractor.py
```

### 5. Communication Techniques Extractor (`extract_communication_techniques.py`)

Extracts communication techniques and principles from "How to Win Friends and Influence People" and "How to Talk to Anyone".

**Usage:**
```bash
python extract_communication_techniques.py
```

### 6. Synthetic Conversations Generator (`generate_technique_conversations.py`)

Generates synthetic conversations that demonstrate the application of communication techniques in various scenarios.

**Usage:**
```bash
python generate_technique_conversations.py
```

### 7. Data Processor (`data_processor.py`)

Cleans and prepares the collected data for training.

**Usage:**
```bash
python data_processor.py
```

## Running the Full Pipeline

You can run the entire data collection and model training pipeline using the script in the parent directory:

```bash
python run_data_pipeline.py
```

This will execute all the steps in sequence. You can skip specific steps using command-line arguments:

```bash
python run_data_pipeline.py --skip-reddit --skip-huggingface
# To include book search:
python run_data_pipeline.py --search-books
# To focus only on communication techniques:
python run_data_pipeline.py --focus-techniques
```

## Data Sources

The scripts collect data from:

1. **Reddit Subreddits:**
   - relationship_advice
   - askwomen
   - TwoXChromosomes
   - dating_advice
   - CasualConversation
   - offmychest
   - MomForAMinute
   - AskWomenAdvice
   - supportiveconversations

2. **Hugging Face Datasets:**
   - daily_dialog
   - empathetic_dialogues
   - blended_skill_talk
   - conv_ai_2

3. **Books from Project Gutenberg:**
   - Woman in Modern Society by Earl Barnes
   - The Psychology of Female Violence by Anna Motz
   - Woman and the New Race by Margaret Sanger
   - The Psychology of Beauty by Ethel D. Puffer
   - Woman as Decoration by Emily Burbank
   - Additional books found through search

4. **Communication Skills Books:**
   - How to Win Friends and Influence People by Dale Carnegie
   - How to Talk to Anyone by Leil Lowndes

## Output

The scripts will create:

- `data/` - Raw data from various sources
- `data/books/raw/` - Downloaded book text files
- `data/books/book_conversations.csv` - Extracted conversation pairs from books
- `data/books/techniques/` - Extracted communication techniques
- `data/books/techniques/techniques_catalog.json` - Catalog of communication techniques
- `data/books/conversations/` - Synthetic conversations demonstrating techniques
- `data/processed/` - Cleaned and processed data
- `data/processed/final_merged_dataset.csv` - The final dataset used for training
- `models/` - Trained models and tokenizers
- `plots/` - Visualizations of model performance

## Notes

- The Reddit scraper attempts to identify female responses based on context clues, but this is not always accurate
- The book extractor focuses on sections containing keywords related to female psychology and communication
- The communication techniques extractor analyzes specific interpersonal skills books to extract actionable techniques
- The synthetic conversations generator creates realistic examples of applying these techniques in various scenarios
- The data processing includes filtering for inappropriate content
- The emotion detection is based on keywords and may not capture all nuances
- Books provide deeper insights into female psychology and communication patterns that may not be present in casual conversations
- The communication skills books provide specific techniques that can be directly applied in conversations
