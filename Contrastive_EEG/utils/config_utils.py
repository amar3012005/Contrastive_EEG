"""
Configuration utilities for the model
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Validate data path
        if 'dataset' in config and 'data_path' in config['dataset']:
            data_path = Path(config['dataset']['data_path'])
            if not data_path.exists():
                logger.warning(f"Data path {data_path} does not exist. Check the path before running.")
        
        # Create output directory if it doesn't exist
        if 'training' in config and 'save_dir' in config['training']:
            save_dir = Path(config['training']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.error("Using default configuration")
        return get_default_config()


def get_default_config():
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'dataset': {
            'data_path': './data/zuco_preprocessed.pkl',
            'vocab_size': 10000,
            'min_freq': 5,
            'max_seq_len': 128,
            'train_split': 0.8,
            'batch_size': 32
        },
        'eeg_encoder': {
            'input_size': 105,
            'hidden_size': 768,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
            'dim_feedforward': 2048
        },
        'text_encoder': {
            'model': 'facebook/bart-base',
            'freeze_first_step': True,
            'unfreeze_embeddings': True,
            'unfreeze_first_layer': True
        },
        'projection': {
            'hidden_dim': 256,
            'output_dim': 128,
            'dropout': 0.1
        },
        'training': {
            'step1': {
                'epochs': 10,
                'lr': 0.01,
                'lr_step': 5,
                'lr_gamma': 0.1
            },
            'step2': {
                'epochs': 5,
                'lr': 0.001,
                'lr_step': 2,
                'lr_gamma': 0.1
            },
            'optimizer': 'sgd',
            'momentum': 0.9,
            'temperature': 0.07,
            'save_dir': './output'
        }
    }