"""
Hugging Face dataset downloader for collecting conversation data.
This module provides functions to download and process conversation datasets
from Hugging Face to enhance the bot's training data.
"""

import os
import json
import logging
from datasets import load_dataset, list_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection/huggingface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("huggingface_downloader")

class HuggingFaceDownloader:
    """Class for downloading and processing conversation datasets from Hugging Face."""
    
    def __init__(self, output_dir="data/huggingface"):
        """Initialize the downloader with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_dataset(self, dataset_name, subset=None, split="train", max_samples=10000):
        """
        Download and process a dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset (if applicable)
            split: Split of the dataset (train, validation, test)
            max_samples: Maximum number of samples to process
            
        Returns:
            Processed conversations
        """
        logger.info(f"Downloading dataset: {dataset_name}" + (f" (subset: {subset})" if subset else ""))
        
        try:
            # Load the dataset
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Limit the number of samples
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
            
            # Process the dataset based on its format
            if dataset_name == "empathetic_dialogues":
                return self._process_empathetic_dialogues(dataset)
            elif dataset_name == "daily_dialog":
                return self._process_daily_dialog(dataset)
            elif dataset_name == "conv_ai_2":
                return self._process_conv_ai_2(dataset)
            elif dataset_name == "blended_skill_talk":
                return self._process_blended_skill_talk(dataset)
            else:
                # Generic processing for unknown datasets
                return self._process_generic_dataset(dataset, dataset_name)
        
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {e}")
            return []
    
    def _process_empathetic_dialogues(self, dataset):
        """Process the Empathetic Dialogues dataset."""
        conversations = []
        
        for item in dataset:
            try:
                conversation = {
                    "source": "huggingface",
                    "dataset": "empathetic_dialogues",
                    "context": item["context"],
                    "message": item["utterance"],
                    "emotion": item["emotion"],
                    "conversation_id": item["conv_id"]
                }
                conversations.append(conversation)
            except Exception as e:
                logger.warning(f"Error processing item in empathetic_dialogues: {e}")
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, "empathetic_dialogues.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} conversations from empathetic_dialogues to {output_file}")
        return conversations
    
    def _process_daily_dialog(self, dataset):
        """Process the Daily Dialog dataset."""
        conversations = []
        
        for item in dataset:
            try:
                # Each item contains a list of utterances
                utterances = item["dialog"]
                emotions = item["emotion"]
                
                # Create conversation pairs
                for i in range(len(utterances) - 1):
                    conversation = {
                        "source": "huggingface",
                        "dataset": "daily_dialog",
                        "message": utterances[i],
                        "response": utterances[i + 1],
                        "emotion": emotions[i] if i < len(emotions) else None
                    }
                    conversations.append(conversation)
            except Exception as e:
                logger.warning(f"Error processing item in daily_dialog: {e}")
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, "daily_dialog.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} conversations from daily_dialog to {output_file}")
        return conversations
    
    def _process_conv_ai_2(self, dataset):
        """Process the ConvAI2 dataset."""
        conversations = []
        
        for item in dataset:
            try:
                # Extract personality
                personality = item["personality"]
                
                # Extract utterances
                utterances = item["utterances"][-1]
                history = utterances["history"]
                
                # Create conversation pairs
                for i in range(0, len(history) - 1, 2):
                    conversation = {
                        "source": "huggingface",
                        "dataset": "conv_ai_2",
                        "personality": personality,
                        "message": history[i],
                        "response": history[i + 1] if i + 1 < len(history) else None
                    }
                    if conversation["response"]:
                        conversations.append(conversation)
            except Exception as e:
                logger.warning(f"Error processing item in conv_ai_2: {e}")
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, "conv_ai_2.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} conversations from conv_ai_2 to {output_file}")
        return conversations
    
    def _process_blended_skill_talk(self, dataset):
        """Process the Blended Skill Talk dataset."""
        conversations = []
        
        for item in dataset:
            try:
                # Extract data
                context = item.get("context", "")
                dialog = item["dialog"]
                
                # Create conversation pairs
                for i in range(0, len(dialog) - 1, 2):
                    conversation = {
                        "source": "huggingface",
                        "dataset": "blended_skill_talk",
                        "context": context,
                        "message": dialog[i],
                        "response": dialog[i + 1] if i + 1 < len(dialog) else None
                    }
                    if conversation["response"]:
                        conversations.append(conversation)
            except Exception as e:
                logger.warning(f"Error processing item in blended_skill_talk: {e}")
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, "blended_skill_talk.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} conversations from blended_skill_talk to {output_file}")
        return conversations
    
    def _process_generic_dataset(self, dataset, dataset_name):
        """Process a generic dataset."""
        conversations = []
        
        # Try to identify conversation pairs based on common field names
        message_fields = ["input", "question", "prompt", "source", "message", "context"]
        response_fields = ["output", "answer", "response", "target", "reply"]
        
        # Get the field names in the dataset
        field_names = dataset.column_names
        
        # Find message and response fields
        message_field = None
        for field in message_fields:
            if field in field_names:
                message_field = field
                break
        
        response_field = None
        for field in response_fields:
            if field in field_names:
                response_field = field
                break
        
        if not message_field or not response_field:
            logger.warning(f"Could not identify message and response fields in {dataset_name}")
            return []
        
        # Process the dataset
        for item in dataset:
            try:
                conversation = {
                    "source": "huggingface",
                    "dataset": dataset_name,
                    "message": item[message_field],
                    "response": item[response_field]
                }
                
                # Add any other relevant fields
                for field in field_names:
                    if field not in [message_field, response_field]:
                        conversation[field] = item[field]
                
                conversations.append(conversation)
            except Exception as e:
                logger.warning(f"Error processing item in {dataset_name}: {e}")
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, f"{dataset_name.replace('/', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} conversations from {dataset_name} to {output_file}")
        return conversations
    
    def download_conversation_datasets(self):
        """Download and process multiple conversation datasets."""
        datasets = [
            ("empathetic_dialogues", None),
            ("daily_dialog", None),
            ("conv_ai_2", None),
            ("blended_skill_talk", None),
            ("allenai/prosocial-dialog", None),
            ("facebook/msc", None)
        ]
        
        all_conversations = []
        
        for dataset_name, subset in datasets:
            conversations = self.download_dataset(dataset_name, subset)
            all_conversations.extend(conversations)
        
        # Save combined data
        output_file = os.path.join(self.output_dir, "all_huggingface_conversations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2)
        
        logger.info(f"Saved {len(all_conversations)} total conversations to {output_file}")
        return all_conversations

# Example usage
if __name__ == "__main__":
    downloader = HuggingFaceDownloader()
    downloader.download_conversation_datasets()
