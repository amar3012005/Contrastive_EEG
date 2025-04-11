import os
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer
import scipy.io as sio  # Added for MATLAB file support
import h5py  # Added for HDF5/MATLAB v7.3+ support
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ZuCoDatasetEnhanced(Dataset):
    """
    Enhanced ZuCo dataset handler for contrastive learning
    Loads EEG data and corresponding sentences from ZuCo dataset
    """
    def __init__(self, data_path, vocab_size=10000, min_freq=5, 
                 max_seq_len=2000, special_tokens=None):
        """
        Initialize the ZuCo dataset
        
        Args:
            data_path: Path to the ZuCo dataset (pickle file or directory with MATLAB files)
            vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include in vocabulary
            max_seq_len: Maximum sequence length for EEG data
            special_tokens: Dictionary of special tokens
        """
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        
        # Add special tokens
        self.special_tokens = special_tokens or {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'CLS': '[CLS]',
            'SEP': '[SEP]',
            'MASK': '[MASK]'
        }
        
        # Initialize BART tokenizer for text processing
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        # Initialize dataset variables
        self.eeg_data = []
        self.sentences = []
        self.eeg_features_dim = None
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load data from ZuCo dataset - either pickle or MATLAB files"""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Check if data_path is a pickle file
            if self.data_path.endswith('.pickle'):
                self._load_from_pickle()
            # Check if data_path is a directory that might contain MATLAB files
            elif os.path.isdir(self.data_path):
                self._load_from_matlab_directory()
            # Try loading the pickle file anyway as a fallback
            else:
                self._load_from_pickle()
            
            # Set feature dimension if we have data
            if self.eeg_data:
                self.eeg_features_dim = self.eeg_data[0].shape[1]
                
            logger.info(f"Processed {len(self.sentences)} sentences")
            logger.info(f"EEG feature dimension: {self.eeg_features_dim}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _load_from_pickle(self):
        """Load data from pickle file (original implementation)"""
        try:
            with open(self.data_path, 'rb') as handle:
                dataset = pickle.load(handle)
                
            logger.info(f"Dataset loaded successfully from pickle")
            
            # Debug dataset structure
            logger.info(f"Dataset type: {type(dataset)}")
            if isinstance(dataset, dict):
                logger.info(f"Dictionary keys: {list(dataset.keys())[:5]} (showing first 5)")
                
                # Inspect the first item to understand structure
                first_key = next(iter(dataset))
                first_item = dataset[first_key]
                logger.info(f"First item type: {type(first_item)}")
                
                if isinstance(first_item, dict):
                    logger.info(f"First item keys: {list(first_item.keys())[:5]} (showing first 5)")
            
                # Process dictionary structure
                for subject_id, subject_data in dataset.items():
                    logger.info(f"Processing subject: {subject_id}")
                    
                    # Handle different data structures
                    if isinstance(subject_data, dict):
                        # Structure: {subject_id: {sentence_id: {data}}}
                        for sentence_id, sentence_data in subject_data.items():
                            if isinstance(sentence_data, dict):
                                self._process_sentence_dict(sentence_data)
                    elif isinstance(subject_data, list):
                        # Structure: {subject_id: [{sentence_data1}, {sentence_data2}, ...]}
                        for sentence_data in subject_data:
                            if isinstance(sentence_data, dict):
                                self._process_sentence_dict(sentence_data)
            elif isinstance(dataset, list):
                logger.info("Dataset is a list structure")
                # Process list structure
                for item in dataset:
                    if isinstance(item, dict):
                        self._process_sentence_dict(item)
        except Exception as e:
            logger.error(f"Error loading pickle file: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_sentence_dict(self, sentence_data):
        """Process a dictionary containing sentence data and EEG features"""
        # Extract sentence text
        sentence = None
        for key in ['content', 'sentence', 'text', 'word']:
            if key in sentence_data:
                if isinstance(sentence_data[key], str):
                    sentence = sentence_data[key]
                    break
                elif isinstance(sentence_data[key], list) and all(isinstance(w, str) for w in sentence_data[key]):
                    sentence = ' '.join(sentence_data[key])
                    break
        
        if not sentence:
            return
        
        # Look for EEG data - it could be under different keys
        eeg_data = None
        for key in ['eeg', 'EEG', 'eeg_features', 'EEG_features', 'features']:
            if key in sentence_data:
                eeg_data = sentence_data[key]
                break
        
        if eeg_data is None:
            return
                
        # Process EEG data
        processed_eeg = None
        if isinstance(eeg_data, dict):
            processed_eeg = self._process_eeg_features(eeg_data)
        elif isinstance(eeg_data, np.ndarray):
            # Direct numpy array EEG data
            processed_eeg = eeg_data[:self.max_seq_len] if eeg_data.shape[0] > self.max_seq_len else eeg_data
        
        if processed_eeg is not None and processed_eeg.shape[0] > 0:
            self.sentences.append(sentence)
            self.eeg_data.append(processed_eeg)
            logger.info(f"Added sentence: {sentence[:50]}...")
    
    def _load_from_matlab_directory(self):
        """Load data from a directory containing MATLAB files"""
        logger.info(f"Loading data from MATLAB files in directory: {self.data_path}")
        
        # Step 1: Look for the sentences file
        sentence_file = self._find_sentence_file(self.data_path)
        if not sentence_file:
            logger.error("Could not find sentence file in the directory")
            # Try going up one directory level in case we're in a subdirectory
            parent_dir = os.path.dirname(self.data_path)
            sentence_file = self._find_sentence_file(parent_dir)
            if not sentence_file:
                logger.error("Could not find sentence file in parent directory either")
                return
        
        # Step 2: Load sentences
        logger.info(f"Found sentence file: {sentence_file}")
        sentences = self._load_sentences_from_mat(sentence_file)
        if not sentences:
            logger.error("Failed to load sentences from mat file")
            return
            
        logger.info(f"Loaded {len(sentences)} sentences from MATLAB file")
        
        # Step 3: Find EEG files - they typically include "EEG" in filename
        eeg_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                # Check for typical EEG file patterns
                if any(pattern in file for pattern in ["_EEG.mat", "EEG_", "eeg_data", "EEGData"]):
                    eeg_files.append(os.path.join(root, file))
        
        if not eeg_files:
            logger.error(f"No EEG files found in {self.data_path}")
            return
            
        logger.info(f"Found {len(eeg_files)} EEG files")
        
        # Step 4: Process each EEG file and pair with sentences
        for eeg_file in tqdm(eeg_files, desc="Processing EEG files"):
            try:
                # Load EEG data from mat file
                eeg_data = self._load_eeg_from_mat(eeg_file)
                if eeg_data is None:
                    continue
                
                # Try to extract subject and sentence info from filename
                filename = os.path.basename(eeg_file)
                sentence_idx = self._extract_sentence_index_from_filename(filename)
                
                # If we can identify a specific sentence
                if sentence_idx is not None and 0 <= sentence_idx < len(sentences):
                    self.sentences.append(sentences[sentence_idx])
                    self.eeg_data.append(eeg_data)
                    logger.info(f"Paired EEG data with sentence: {sentences[sentence_idx][:50]}...")
                # Otherwise, if there's only one sentence file, use it for all EEG files
                elif len(sentences) == 1:
                    self.sentences.append(sentences[0])
                    self.eeg_data.append(eeg_data)
                    logger.info(f"Paired EEG data with the only available sentence")
                else:
                    # If we can't match, just store EEG data with a placeholder sentence
                    placeholder = f"Sentence for EEG data from {os.path.basename(eeg_file)}"
                    self.sentences.append(placeholder)
                    self.eeg_data.append(eeg_data)
                    logger.info(f"Using placeholder for EEG file: {filename}")
            except Exception as e:
                logger.error(f"Error processing EEG file {eeg_file}: {str(e)}")
                continue
    
    def _find_sentence_file(self, directory):
        """Find the sentence file in the given directory"""
        possible_names = ["sentencesSR", "sentences", "sent", "text", "SR_sentences"]
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".mat") and any(name.lower() in file.lower() for name in possible_names):
                    return os.path.join(root, file)
        return None
        
    def _load_sentences_from_mat(self, file_path):
        """Load sentences from MAT file"""
        sentences = []
        try:
            # First attempt with scipy.io for regular MAT files
            try:
                mat_data = sio.loadmat(file_path)
                # Look for keys containing 'sentence' or 'text'
                relevant_keys = [k for k in mat_data.keys() 
                              if any(word in k.lower() for word in ['sentence', 'text', 'content']) 
                              and not k.startswith('__')]
                
                if relevant_keys:
                    key = relevant_keys[0]
                    data = mat_data[key]
                    
                    # Handle different array formats
                    if isinstance(data, np.ndarray):
                        if data.dtype.type is np.str_:
                            sentences = data.tolist()
                        else:
                            # Try to extract strings
                            for item in data.flatten():
                                if isinstance(item, np.ndarray) and item.size > 0:
                                    try:
                                        sentences.append(str(item[0]))
                                    except:
                                        pass
                    
                    logger.info(f"Loaded {len(sentences)} sentences with scipy.io")
                    return sentences
            except Exception as e:
                logger.warning(f"Error with scipy.io: {e}, trying h5py...")
                
            # Second attempt with h5py for HDF5/MATLAB v7.3+ files
            try:
                with h5py.File(file_path, 'r') as f:
                    # Find the key for sentences
                    sentence_keys = [k for k in f.keys() 
                                  if any(word in k.lower() for word in ['sentence', 'text', 'content'])]
                    
                    if sentence_keys:
                        key = sentence_keys[0]
                        data = f[key]
                        
                        # Get references to string data
                        for ref in data:
                            try:
                                # Handle different ways strings might be stored
                                if isinstance(ref, h5py.Reference):
                                    sentence = ''.join(chr(c[0]) for c in f[ref])
                                    sentences.append(sentence)
                                elif len(ref) > 0:
                                    sentence = ''.join(chr(c[0]) for c in f[ref[0]])
                                    sentences.append(sentence)
                            except Exception as e:
                                logger.warning(f"Error extracting sentence: {e}")
                                continue
                        
                        logger.info(f"Loaded {len(sentences)} sentences with h5py")
                        return sentences
            except Exception as e:
                logger.warning(f"Error with h5py: {e}")
                
            # If we got here, we couldn't load sentences properly
            logger.error(f"Failed to extract sentences from {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error in _load_sentences_from_mat: {e}")
            return []
    
    def _load_eeg_from_mat(self, file_path):
        """Load EEG data from MAT file"""
        try:
            # Attempt with scipy.io first
            try:
                mat_data = sio.loadmat(file_path)
                
                # Look for EEG data
                eeg_vars = [k for k in mat_data.keys() 
                          if any(name in k.lower() for name in ['eeg', 'data', 'signal']) 
                          and not k.startswith('__')]
                
                if eeg_vars:
                    key = eeg_vars[0]
                    eeg_data = mat_data[key]
                    
                    # If it's a struct with fields
                    if isinstance(eeg_data, np.ndarray) and eeg_data.dtype.names is not None:
                        for field in ['data', 'signal', 'eeg']:
                            if field in eeg_data.dtype.names:
                                return self._process_eeg_numpy(eeg_data[field][0,0])
                    
                    # If it's a direct numpy array
                    return self._process_eeg_numpy(eeg_data)
            except Exception as e:
                logger.warning(f"Error loading with scipy.io: {e}, trying h5py...")
            
            # Try with h5py
            try:
                with h5py.File(file_path, 'r') as f:
                    # Look for EEG data
                    for key in f.keys():
                        if any(name in key.lower() for name in ['eeg', 'data', 'signal']):
                            # If the key points to a dataset
                            if isinstance(f[key], h5py.Dataset):
                                eeg_data = f[key][:]
                                return self._process_eeg_numpy(eeg_data)
                            # If it's a group with 'data' subfield
                            elif isinstance(f[key], h5py.Group):
                                for subkey in f[key].keys():
                                    if any(name in subkey.lower() for name in ['data', 'signal']):
                                        eeg_data = f[key][subkey][:]
                                        return self._process_eeg_numpy(eeg_data)
            except Exception as e:
                logger.warning(f"Error loading with h5py: {e}")
                
            logger.error(f"Could not find EEG data in {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error in _load_eeg_from_mat: {e}")
            return None
    
    def _process_eeg_numpy(self, eeg_data):
        """Process numpy EEG data"""
        if eeg_data is None:
            return None
            
        # Ensure it's a numpy array
        eeg_data = np.array(eeg_data)
        
        # Check dimensions and reshape if needed
        if len(eeg_data.shape) == 1:
            # Single channel data
            eeg_data = eeg_data.reshape(-1, 1)
        elif len(eeg_data.shape) > 2:
            # Multi-dimensional data, flatten to 2D
            # Assume first dimension is time, reshape the rest
            time_points = eeg_data.shape[0]
            features = np.prod(eeg_data.shape[1:])
            eeg_data = eeg_data.reshape(time_points, features)
        
        # Check if sequence is too long
        if eeg_data.shape[0] > self.max_seq_len:
            eeg_data = eeg_data[:self.max_seq_len, :]
        
        return eeg_data
    
    def _extract_sentence_index_from_filename(self, filename):
        """Extract sentence index from filename"""
        # This implementation depends on your filename convention
        # Common pattern: subject_taskSR1_EEG.mat for sentence 1
        try:
            # Look for patterns like 'SR1', 'sr2', etc.
            import re
            matches = re.findall(r'[sS][rR](\d+)', filename)
            if matches:
                # Convert to 0-indexed
                return int(matches[0]) - 1
            
            # Look for numeric part in filename
            matches = re.findall(r'_(\d+)_', filename)
            if matches:
                return int(matches[0]) - 1
                
            return None
        except:
            return None
    
    def _process_eeg_features(self, eeg_features):
        """
        Process EEG features from dictionary of electrodes
        
        Args:
            eeg_features: Dictionary of EEG features per electrode
            
        Returns:
            Processed EEG features as numpy array
        """
        try:
            # Get all electrode data
            all_electrode_data = []
            
            for electrode, data in eeg_features.items():
                # Skip non-data keys
                if electrode in ['word_indexes', 'word_string']:
                    continue
                
                # Append electrode data
                electrode_data = np.array(data)
                all_electrode_data.append(electrode_data)
            
            if not all_electrode_data:
                return None
            
            # Stack all electrode data
            stacked_data = np.column_stack(all_electrode_data)
            
            # Check if sequence is too long
            if stacked_data.shape[0] > self.max_seq_len:
                stacked_data = stacked_data[:self.max_seq_len, :]
            
            return stacked_data
            
        except Exception as e:
            logger.error(f"Error processing EEG features: {str(e)}")
            return None
    
    def __len__(self):
        """Return the number of sentences in the dataset"""
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """Get a single item (EEG data and corresponding sentence)"""
        eeg_data = torch.FloatTensor(self.eeg_data[idx])
        sentence = self.sentences[idx]
        
        return eeg_data, sentence


def create_data_loaders(dataset, batch_size, train_split=0.8, shuffle=True):
    """
    Create training and validation data loaders
    
    Args:
        dataset: The dataset to split into train/val
        batch_size: Batch size for loaders
        train_split: Ratio of training data
        shuffle: Whether to shuffle the data
        
    Returns:
        train_loader, val_loader
    """
    # Define dataset sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # Custom collate function for batching
    def collate_fn(batch):
        # Separate EEG data and sentences
        eeg_data = [item[0] for item in batch]
        sentences = [item[1] for item in batch]
        
        # Tokenize sentences
        encoded = dataset.tokenizer(
            sentences, 
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Find max EEG sequence length in batch
        max_len = max(data.size(0) for data in eeg_data)
        
        # Pad EEG data
        padded_eeg = []
        for data in eeg_data:
            # If data is shorter than max_len, pad with zeros
            if data.size(0) < max_len:
                padding = torch.zeros((max_len - data.size(0), data.size(1)))
                padded_data = torch.cat([data, padding], dim=0)
                padded_eeg.append(padded_data)
            else:
                padded_eeg.append(data)
        
        # Stack all padded EEG data
        eeg_tensor = torch.stack(padded_eeg)
        
        return eeg_tensor, encoded['input_ids'], encoded['attention_mask']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader