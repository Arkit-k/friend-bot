"""
Main script to collect data from the internet and train the model.
This script orchestrates the entire process of data collection, processing, and model training.
"""

import os
import argparse
import logging
import subprocess
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("collect_and_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("collect_and_train")

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/scraped",
        "data/huggingface",
        "data/processed",
        "data/reddit",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for the process to complete
        process.wait()
        
        # Check if the process was successful
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def collect_data(args):
    """Collect data from various sources."""
    success = True
    
    # Web scraping
    if args.web or args.all:
        logger.info("Starting web scraping...")
        cmd = "python data_collection/web_scraper.py"
        success = success and run_command(cmd, "Web scraping")
    
    # Hugging Face datasets
    if args.huggingface or args.all:
        logger.info("Starting Hugging Face data collection...")
        cmd = "python data_collection/huggingface_downloader.py"
        success = success and run_command(cmd, "Hugging Face data collection")
    
    # Reddit API
    if args.reddit or args.all:
        logger.info("Starting Reddit data collection...")
        cmd = "python gather_reddit_data.py --female-only --limit 100"
        success = success and run_command(cmd, "Reddit data collection")
    
    return success

def process_data():
    """Process the collected data."""
    logger.info("Processing collected data...")
    cmd = "python data_collection/data_processor.py"
    return run_command(cmd, "Data processing")

def train_model(args):
    """Train the model using the processed data."""
    logger.info("Training the model...")
    
    # Create a dummy model if requested
    if args.dummy:
        logger.info("Creating dummy model files...")
        cmd = "python create_dummy_model.py"
        success = run_command(cmd, "Creating dummy model")
        if not success:
            return False
    
    # Train the emotion detection model
    logger.info("Training emotion detection model...")
    cmd = f"python train_emotion_model.py --epochs {args.epochs}"
    success = run_command(cmd, "Training emotion detection model")
    
    return success

def main():
    """Main function to orchestrate the entire process."""
    parser = argparse.ArgumentParser(description="Collect data and train the model")
    
    # Data collection arguments
    parser.add_argument("--web", action="store_true", help="Collect data from web scraping")
    parser.add_argument("--huggingface", action="store_true", help="Collect data from Hugging Face datasets")
    parser.add_argument("--reddit", action="store_true", help="Collect data from Reddit API")
    parser.add_argument("--all", action="store_true", help="Collect data from all sources")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dummy", action="store_true", help="Create dummy model files")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # If no data collection method is specified, use all
    if not (args.web or args.huggingface or args.reddit or args.all):
        args.all = True
    
    # Create necessary directories
    create_directories()
    
    # Start the process
    start_time = time.time()
    logger.info("Starting data collection and model training process...")
    
    # Collect data
    if not args.skip_collection:
        logger.info("Step 1: Collecting data...")
        if not collect_data(args):
            logger.error("Data collection failed. Exiting.")
            return 1
    else:
        logger.info("Skipping data collection as requested.")
    
    # Process data
    if not args.skip_processing:
        logger.info("Step 2: Processing data...")
        if not process_data():
            logger.error("Data processing failed. Exiting.")
            return 1
    else:
        logger.info("Skipping data processing as requested.")
    
    # Train model
    if not args.skip_training:
        logger.info("Step 3: Training model...")
        if not train_model(args):
            logger.error("Model training failed. Exiting.")
            return 1
    else:
        logger.info("Skipping model training as requested.")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Process completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
