"""
Data loading utilities for the Contrastive EEG-to-Text model
"""

import os
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ZuCoDatasetEnhanced(Dataset):
    """
    Enhanced dataset class for ZuCo EEG data with improved error handling
    """
    
    def __init__(self, data_path, vocab_size=10000, min_freq=5, max_seq_len=None):
        """
        Initialize the ZuCo dataset
        
        Args:
            data_path: Path to preprocessed ZuCo data (pickle file or directory)
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for words to be included in vocabulary
            max_seq_len: Maximum sequence length
        """
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        
        # Initialize tokenizer for text encoding
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        # Load and preprocess data
        self.data, self.eeg_features_dim = self._load_and_process_data()
    
    def _load_from_pickle(self):
        """
        Load data from pickle file with improved error handling
        
        Returns:
            Loaded data or None if error occurred
        """
        try:
            data_path = self.data_path
            # Check if data_path is a directory or file
            path = Path(data_path)
            
            if path.is_dir():
                # If directory, look for preprocessed.pkl file
                potential_files = list(path.glob("*.pkl"))
                if not potential_files:
                    raise FileNotFoundError(f"No pickle files found in directory {data_path}")
                
                # Use the largest pickle file in the directory
                data_path = max(potential_files, key=lambda x: x.stat().st_size)
                logger.info(f"Using pickle file: {data_path}")
            
            logger.info(f"Loading data from {data_path}")
            with open(data_path, 'rb') as handle:
                raw_data = pickle.load(handle)
            return raw_data
        
        except Exception as e:
            logger.error(f"Error loading pickle file: {str(e)}")
            logger.error(f"Traceback (most recent call last):\n{e.__traceback__}")
            return None
    
    def _load_and_process_data(self):
        """
        Load and preprocess data
        
        Returns:
            Tuple of (processed_data, eeg_feature_dimension)
        """
        # Load raw data
        raw_data = self._load_from_pickle()
        
        processed_data = []
        eeg_dim = None
        
        if raw_data is None:
            logger.warning("Failed to load data. Returning empty dataset.")
            return processed_data, eeg_dim
        
        # Process each sample
        try:
            # Logic for processing ZuCo data
            # This is a placeholder - actual implementation depends on the data format
            
            # Example processing logic:
            if isinstance(raw_data, list):
                for item in raw_data:
                    if 'eeg' in item and 'sentence' in item:
                        eeg_features = item['eeg']
                        sentence = item['sentence']
                        
                        # Tokenize sentence
                        tokenized = self.tokenizer(
                            sentence,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_seq_len or 128,
                            return_tensors="pt"
                        )
                        
                        # Extract tokenizer outputs
                        input_ids = tokenized['input_ids'].squeeze(0)
                        attention_mask = tokenized['attention_mask'].squeeze(0)
                        
                        # Record EEG feature dimension
                        if eeg_dim is None and eeg_features is not None:
                            eeg_dim = eeg_features.shape[-1]
                        
                        # Add to processed data
                        processed_data.append({
                            'eeg': torch.tensor(eeg_features, dtype=torch.float),
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'sentence': sentence
                        })
            elif isinstance(raw_data, dict):
                # Alternative format - dictionary with keys for EEG and text
                if 'eeg_data' in raw_data and 'sentences' in raw_data:
                    eeg_data = raw_data['eeg_data']
                    sentences = raw_data['sentences']
                    
                    for i, (eeg, sentence) in enumerate(zip(eeg_data, sentences)):
                        # Tokenize sentence
                        tokenized = self.tokenizer(
                            sentence,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_seq_len or 128,
                            return_tensors="pt"
                        )
                        
                        # Extract tokenizer outputs
                        input_ids = tokenized['input_ids'].squeeze(0)
                        attention_mask = tokenized['attention_mask'].squeeze(0)
                        
                        # Record EEG feature dimension
                        if eeg_dim is None and eeg is not None:
                            eeg_dim = eeg.shape[-1]
                        
                        # Add to processed data
                        processed_data.append({
                            'eeg': torch.tensor(eeg, dtype=torch.float),
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'sentence': sentence
                        })
                        
            logger.info(f"Processed {len(processed_data)} sentences")
            logger.info(f"EEG feature dimension: {eeg_dim}")
            
            return processed_data, eeg_dim
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return [], None
    
    def __len__(self):
        """Get dataset length"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (eeg_features, input_ids, attention_mask)
        """
        item = self.data[idx]
        return item['eeg'], item['input_ids'], item['attention_mask']


def create_data_loaders(dataset, batch_size=32, train_split=0.8, shuffle=True):
    """
    Create data loaders for training and validation
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        train_split: Training data proportion
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot create data loaders.")
        
    # Split dataset into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader