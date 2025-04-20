#!/usr/bin/env python
"""
Evaluation and inference script for Contrastive EEG-to-Text model
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
import yaml
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data.data_loader import ZuCoDatasetEnhanced
from src.models.contrastive_model import ContrastiveEEG2Text
from utils.config_utils import get_config
from utils.visualization import visualize_embeddings, plot_similarity_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('contrastive_eeg_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Contrastive EEG-to-Text Model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to ZuCo dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/evaluation',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_trained_model(checkpoint_path):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        model, config
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Extract config from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model
        eeg_input_size = config['eeg_encoder']['input_size']
        hidden_size = config['eeg_encoder']['hidden_size']
        proj_hidden = config['projection']['hidden_dim']
        proj_output = config['projection']['output_dim']
        num_layers = config['eeg_encoder']['num_layers']
        num_heads = config['eeg_encoder']['num_heads']
        dropout = config['eeg_encoder']['dropout']
        temperature = config['training']['temperature']
        
        model = ContrastiveEEG2Text(
            eeg_input_size=eeg_input_size,
            hidden_size=hidden_size,
            proj_hidden=proj_hidden,
            proj_output=proj_output,
            temperature=temperature,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return model, config
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def create_data_loader(dataset, batch_size):
    """
    Create data loader for evaluation
    
    Args:
        dataset: Dataset to create loader from
        batch_size: Batch size
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader, SequentialSampler
    
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
        
        return eeg_tensor, encoded['input_ids'], encoded['attention_mask'], sentences
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        sampler=SequentialSampler(dataset)
    )
    
    return loader


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on data loader
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)
    
    # Store results
    results = {
        'accuracy': [],
        'eeg_to_text_acc': [],
        'text_to_eeg_acc': [],
        'eeg_embeddings': [],
        'text_embeddings': [],
        'sentences': [],
        'similarity_matrices': []
    }
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, (eeg_data, input_ids, attention_mask, sentences) in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Move data to device
            eeg_data = eeg_data.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get model outputs
            outputs = model(eeg_data, input_ids, attention_mask)
            
            # Save embeddings and sentences for visualization
            results['eeg_embeddings'].append(outputs['eeg_embeddings'].cpu())
            results['text_embeddings'].append(outputs['text_embeddings'].cpu())
            results['sentences'].extend(sentences)
            
            # Save metrics
            results['accuracy'].append(outputs['accuracy'].item())
            results['eeg_to_text_acc'].append(outputs['eeg_to_text_acc'].item())
            results['text_to_eeg_acc'].append(outputs['text_to_eeg_acc'].item())
            
            # Save similarity matrix
            results['similarity_matrices'].append(outputs['logits'].cpu())
    
    # Combine results
    results['eeg_embeddings'] = torch.cat(results['eeg_embeddings'], dim=0)
    results['text_embeddings'] = torch.cat(results['text_embeddings'], dim=0)
    results['similarity_matrices'] = torch.cat(results['similarity_matrices'], dim=0)
    
    # Calculate average metrics
    results['avg_accuracy'] = np.mean(results['accuracy'])
    results['avg_eeg_to_text_acc'] = np.mean(results['eeg_to_text_acc'])
    results['avg_text_to_eeg_acc'] = np.mean(results['text_to_eeg_acc'])
    
    return results


def generate_report(results, output_dir):
    """
    Generate evaluation report
    
    Args:
        results: Evaluation results
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save report
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("Contrastive EEG-to-Text Model Evaluation\n")
        f.write("=======================================\n\n")
        
        f.write(f"Average Accuracy: {results['avg_accuracy']:.4f}\n")
        f.write(f"Average EEG→Text Accuracy: {results['avg_eeg_to_text_acc']:.4f}\n")
        f.write(f"Average Text→EEG Accuracy: {results['avg_text_to_eeg_acc']:.4f}\n\n")
        
        f.write(f"Number of samples evaluated: {len(results['sentences'])}\n")
    
    # Generate visualizations
    
    # 1. Embedding visualization (with random subset if too large)
    max_viz_samples = 200
    if len(results['sentences']) > max_viz_samples:
        indices = np.random.choice(len(results['sentences']), max_viz_samples, replace=False)
        eeg_embs_subset = results['eeg_embeddings'][indices]
        text_embs_subset = results['text_embeddings'][indices]
        sentences_subset = [results['sentences'][i] for i in indices]
    else:
        eeg_embs_subset = results['eeg_embeddings']
        text_embs_subset = results['text_embeddings']
        sentences_subset = results['sentences']
    
    visualize_embeddings(
        eeg_embeddings=eeg_embs_subset,
        text_embeddings=text_embs_subset,
        sentences=sentences_subset,
        save_path=os.path.join(output_dir, 'embedding_visualization.png')
    )
    
    # 2. Sample similarity matrix
    max_sim_samples = 20
    if len(results['sentences']) > max_sim_samples:
        indices = np.random.choice(len(results['sentences']), max_sim_samples, replace=False)
        sim_matrix = results['similarity_matrices'][indices][:, indices]
        sentences_subset = [results['sentences'][i] for i in indices]
    else:
        sim_matrix = results['similarity_matrices']
        sentences_subset = results['sentences']
    
    plot_similarity_matrix(
        similarity_matrix=sim_matrix,
        sentences=sentences_subset,
        save_path=os.path.join(output_dir, 'similarity_matrix.png')
    )
    
    logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model
    model, config = load_trained_model(args.model_path)
    
    # Load dataset
    logger.info(f"Loading ZuCo dataset from {args.data_path}")
    dataset = ZuCoDatasetEnhanced(
        data_path=args.data_path,
        vocab_size=config['dataset']['vocab_size'],
        min_freq=config['dataset']['min_freq'],
        max_seq_len=config['dataset']['max_seq_len']
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Create data loader
    data_loader = create_data_loader(dataset, args.batch_size)
    
    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(model, data_loader, device)
    
    # Generate report
    logger.info("Generating evaluation report")
    generate_report(results, args.output_dir)
    
    logger.info(f"Model accuracy: {results['avg_accuracy']:.4f}")
    logger.info(f"EEG→Text accuracy: {results['avg_eeg_to_text_acc']:.4f}")
    logger.info(f"Text→EEG accuracy: {results['avg_text_to_eeg_acc']:.4f}")


if __name__ == "__main__":
    main()