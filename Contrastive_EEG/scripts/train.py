#!/usr/bin/env python
"""
Main training script for Contrastive EEG-to-Text model
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import yaml
import datetime

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data.data_loader import ZuCoDatasetEnhanced, create_data_loaders
from src.models.contrastive_model import ContrastiveEEG2Text
from src.training.trainer import TwoStepTrainer
from utils.config_utils import get_config
from utils.visualization import plot_training_history, visualize_embeddings, plot_similarity_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('contrastive_eeg_training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train Contrastive EEG-to-Text Model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config file)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to ZuCo dataset (overrides config file)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config file)'
    )
    
    parser.add_argument(
        '--step1-epochs',
        type=int,
        default=None,
        help='Epochs for step 1 training (overrides config file)'
    )
    
    parser.add_argument(
        '--step2-epochs',
        type=int,
        default=None,
        help='Epochs for step 2 training (overrides config file)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir:
        config['training']['save_dir'] = args.output_dir
    if args.data_path:
        config['dataset']['data_path'] = args.data_path
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    if args.step1_epochs:
        config['training']['step1']['epochs'] = args.step1_epochs
    if args.step2_epochs:
        config['training']['step2']['epochs'] = args.step2_epochs
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config['training']['save_dir']) / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    config['training']['save_dir'] = str(save_dir)
    
    # Save configuration to output directory
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Starting training with configuration: {config}")
    logger.info(f"Output will be saved to {save_dir}")
    
    # Load dataset
    logger.info(f"Loading ZuCo dataset from {config['dataset']['data_path']}")
    dataset = ZuCoDatasetEnhanced(
        data_path=config['dataset']['data_path'],
        vocab_size=config['dataset']['vocab_size'],
        min_freq=config['dataset']['min_freq'],
        max_seq_len=config['dataset']['max_seq_len']
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Update EEG input size in config based on dataset
    config['eeg_encoder']['input_size'] = dataset.eeg_features_dim
    logger.info(f"EEG feature dimension: {dataset.eeg_features_dim}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config['dataset']['batch_size'],
        train_split=config['dataset'].get('train_split', 0.8),
        shuffle=True
    )
    
    logger.info(f"Created data loaders with batch size {config['dataset']['batch_size']}")
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(val_loader)} batches")
    
    # Create model
    logger.info("Initializing model")
    model = ContrastiveEEG2Text(
        eeg_input_size=config['eeg_encoder']['input_size'],
        hidden_size=config['eeg_encoder']['hidden_size'],
        proj_hidden=config['projection']['hidden_dim'],
        proj_output=config['projection']['output_dim'],
        temperature=config['training']['temperature'],
        num_layers=config['eeg_encoder']['num_layers'],
        num_heads=config['eeg_encoder']['num_heads'],
        dropout=config['eeg_encoder']['dropout']
    )
    
    # Initialize trainer
    trainer = TwoStepTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    logger.info("Starting training")
    history = trainer.train()
    
    # Save training history plot
    logger.info("Saving training history plot")
    plot_training_history(
        history=history,
        save_dir=save_dir
    )
    
    # Evaluate and visualize
    logger.info("Creating embedding visualizations")
    evaluate_and_visualize(
        model=model,
        val_loader=val_loader,
        save_dir=save_dir
    )
    
    logger.info(f"Training completed. Results saved to {save_dir}")


def evaluate_and_visualize(model, val_loader, save_dir):
    """
    Evaluate model and create visualizations
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        save_dir: Directory to save visualizations
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get a batch of data
    eeg_batch, input_ids_batch, attention_mask_batch = next(iter(val_loader))
    
    # Move data to device
    eeg_batch = eeg_batch.to(device)
    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(eeg_batch, input_ids_batch, attention_mask_batch)
        eeg_embeddings = outputs['eeg_embeddings']
        text_embeddings = outputs['text_embeddings']
        logits = outputs['logits']
    
    # Get sentences from tokenized input
    sentences = val_loader.dataset.dataset.tokenizer.batch_decode(
        input_ids_batch, skip_special_tokens=True
    )
    
    # Create visualizations
    # 1. Embedding visualization
    visualize_embeddings(
        eeg_embeddings=eeg_embeddings,
        text_embeddings=text_embeddings,
        sentences=sentences,
        save_path=os.path.join(save_dir, 'embedding_visualization.png')
    )
    
    # 2. Similarity matrix
    plot_similarity_matrix(
        similarity_matrix=logits,
        sentences=sentences,
        save_path=os.path.join(save_dir, 'similarity_matrix.png')
    )


if __name__ == "__main__":
    main()