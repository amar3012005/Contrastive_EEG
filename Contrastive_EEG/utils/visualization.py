import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.lines import Line2D

def plot_training_history(history, save_dir=None):
    """
    Plot training history metrics
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    # Plot direction-specific accuracies
    plt.subplot(2, 2, 3)
    plt.plot(history['eeg_to_text_acc'], label='EEG→Text Accuracy')
    plt.plot(history['text_to_eeg_acc'], label='Text→EEG Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Direction-Specific Accuracy vs. Epoch')
    plt.legend()
    
    # Plot temperature
    plt.subplot(2, 2, 4)
    plt.plot(history['temperature'])
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.title('Temperature vs. Epoch')
    
    plt.tight_layout()
    
    # Save plots if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
    
    plt.close()

def visualize_embeddings(eeg_embeddings, text_embeddings, sentences, save_path=None):
    """
    Visualize EEG and text embeddings using t-SNE
    
    Args:
        eeg_embeddings: EEG embeddings tensor
        text_embeddings: Text embeddings tensor
        sentences: List of sentences corresponding to embeddings
        save_path: Path to save visualization
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(eeg_embeddings, torch.Tensor):
        eeg_embeddings = eeg_embeddings.cpu().numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.cpu().numpy()
    
    # Concatenate embeddings for t-SNE
    all_embeddings = np.vstack([eeg_embeddings, text_embeddings])
    
    # Create labels for points (0 for EEG, 1 for text)
    labels = np.concatenate([
        np.zeros(len(eeg_embeddings)), 
        np.ones(len(text_embeddings))
    ])
    
    # Create t-SNE model
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split 2D embeddings back into EEG and text
    eeg_embeddings_2d = embeddings_2d[:len(eeg_embeddings)]
    text_embeddings_2d = embeddings_2d[len(eeg_embeddings):]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot EEG and text embeddings with different colors
    plt.scatter(eeg_embeddings_2d[:, 0], eeg_embeddings_2d[:, 1], 
                c='blue', alpha=0.7, label='EEG')
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                c='red', alpha=0.7, label='Text')
    
    # Plot corresponding points connected by lines
    for i in range(min(len(eeg_embeddings_2d), len(text_embeddings_2d))):
        plt.plot([eeg_embeddings_2d[i, 0], text_embeddings_2d[i, 0]],
                 [eeg_embeddings_2d[i, 1], text_embeddings_2d[i, 1]],
                 'k-', alpha=0.3)
    
    plt.title('t-SNE visualization of EEG and text embeddings')
    plt.legend()
    
    # Add sentence labels for a few random points
    if sentences:
        np.random.seed(42)
        indices = np.random.choice(
            min(len(eeg_embeddings_2d), len(text_embeddings_2d)),
            min(5, len(sentences)),
            replace=False
        )
        
        for idx in indices:
            if idx < len(sentences):
                # Truncate long sentences
                sentence = sentences[idx]
                if len(sentence) > 50:
                    sentence = sentence[:47] + "..."
                    
                # Add label at text embedding point
                plt.annotate(
                    sentence,
                    (text_embeddings_2d[idx, 0], text_embeddings_2d[idx, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round", alpha=0.2)
                )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def plot_similarity_matrix(similarity_matrix, sentences, save_path=None):
    """
    Plot similarity matrix between EEG and text embeddings
    
    Args:
        similarity_matrix: Matrix of similarity scores
        sentences: List of sentences
        save_path: Path to save visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    
    # Limit the number of sentences to display
    max_display = min(20, len(sentences))
    if len(sentences) > max_display:
        similarity_matrix = similarity_matrix[:max_display, :max_display]
        sentences = sentences[:max_display]
    
    # Create labels (truncate long sentences)
    labels = []
    for sent in sentences:
        if len(sent) > 20:
            labels.append(sent[:17] + "...")
        else:
            labels.append(sent)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap='viridis',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('EEG-Text Similarity Matrix')
    plt.xlabel('Text Embeddings')
    plt.ylabel('EEG Embeddings')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()