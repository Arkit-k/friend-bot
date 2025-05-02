#!/usr/bin/env python
"""
Data Pipeline Runner

This script runs the entire data collection and model training pipeline:
1. Collect data from Reddit
2. Download datasets from Hugging Face
3. Download and extract data from books about women's psychology
4. Extract communication techniques from specific books
5. Generate synthetic conversations demonstrating these techniques
6. Process and clean the data
7. Train the model
"""

import os
import subprocess
import argparse
import time

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")

    result = subprocess.run(['python', script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"SUCCESS: {description}")
        print(result.stdout)
    else:
        print(f"ERROR: {description} failed with code {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        if input("Continue with pipeline? (y/n): ").lower() != 'y':
            print("Pipeline aborted.")
            return False

    return True

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Run the data collection and model training pipeline')
    parser.add_argument('--skip-reddit', action='store_true', help='Skip Reddit data collection')
    parser.add_argument('--skip-huggingface', action='store_true', help='Skip Hugging Face data collection')
    parser.add_argument('--skip-books', action='store_true', help='Skip book data collection')
    parser.add_argument('--skip-techniques', action='store_true', help='Skip communication techniques extraction')
    parser.add_argument('--skip-synthetic', action='store_true', help='Skip synthetic conversations generation')
    parser.add_argument('--skip-processing', action='store_true', help='Skip data processing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--search-books', action='store_true', help='Search for additional books')
    parser.add_argument('--focus-techniques', action='store_true', help='Focus only on communication techniques')
    args = parser.parse_args()

    # If focus-techniques is set, skip other data sources
    if args.focus_techniques:
        args.skip_reddit = True
        args.skip_huggingface = True
        args.skip_books = True
        args.skip_techniques = False
        args.skip_synthetic = False

    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/books/raw', exist_ok=True)
    os.makedirs('data/books/techniques', exist_ok=True)
    os.makedirs('data/books/conversations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    start_time = time.time()

    # Step 1: Collect data from Reddit
    if not args.skip_reddit:
        if not run_script('data_collection/reddit_scraper.py', 'Reddit Data Collection'):
            return
    else:
        print("Skipping Reddit data collection")

    # Step 2: Download datasets from Hugging Face
    if not args.skip_huggingface:
        if not run_script('data_collection/huggingface_datasets.py', 'Hugging Face Data Collection'):
            return
    else:
        print("Skipping Hugging Face data collection")

    # Step 3: Download and process book data
    if not args.skip_books:
        # Download books
        book_cmd = 'data_collection/download_books.py'
        if args.search_books:
            book_cmd += ' --search'

        if not run_script(book_cmd, 'Book Download'):
            return

        # Extract data from books
        if not run_script('data_collection/book_extractor.py', 'Book Data Extraction'):
            return
    else:
        print("Skipping book data collection")

    # Step 4: Extract communication techniques from specific books
    if not args.skip_techniques:
        # Check if the communication books exist
        comm_books_exist = False
        for filename in os.listdir('data/books/raw'):
            lower_filename = filename.lower()
            if ("win" in lower_filename and "friend" in lower_filename) or ("talk" in lower_filename and "anyone" in lower_filename):
                comm_books_exist = True
                break

        if not comm_books_exist:
            print("Communication books not found. Creating placeholder files...")

            # Create a directory for the communication books
            os.makedirs('data/books/raw', exist_ok=True)

            # Create placeholder files with instructions
            with open('data/books/raw/how_to_win_friends.txt', 'w') as f:
                f.write("This is a placeholder file. Please replace with the actual content of 'How to Win Friends and Influence People' by Dale Carnegie.")

            with open('data/books/raw/how_to_talk_to_anyone.txt', 'w') as f:
                f.write("This is a placeholder file. Please replace with the actual content of 'How to Talk to Anyone' by Leil Lowndes.")

        # Extract communication techniques
        if not run_script('data_collection/extract_communication_techniques.py', 'Communication Techniques Extraction'):
            return
    else:
        print("Skipping communication techniques extraction")

    # Step 5: Generate synthetic conversations demonstrating techniques
    if not args.skip_synthetic:
        if not run_script('data_collection/generate_technique_conversations.py', 'Synthetic Conversations Generation'):
            return
    else:
        print("Skipping synthetic conversations generation")

    # Step 6: Process and clean the data
    if not args.skip_processing:
        if not run_script('data_collection/data_processor.py', 'Data Processing'):
            return
    else:
        print("Skipping data processing")

    # Step 7: Train the model
    if not args.skip_training:
        if not run_script('train_model.py', 'Model Training'):
            return
    else:
        print("Skipping model training")

    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*80}")
    print(f"Pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'='*80}\n")

    # Check if all necessary files exist
    model_file = 'models/friendship_model.h5'
    tokenizer_file = 'models/tokenizer.pkl'
    label_encoder_file = 'models/label_encoder.pkl'

    if all(os.path.exists(f) for f in [model_file, tokenizer_file, label_encoder_file]):
        print("All model files were created successfully!")
        print("\nNext steps:")
        print("1. Copy the model files to the main directory:")
        print("   - models/friendship_model.h5 → friendship_model.h5")
        print("   - models/tokenizer.pkl → tokenizer.pkl")
        print("   - models/label_encoder.pkl → label_encoder.pkl")
        print("2. Run the Discord bot: python run_bot.py")
    else:
        print("Some model files are missing. Check the logs for errors.")

if __name__ == "__main__":
    main()
