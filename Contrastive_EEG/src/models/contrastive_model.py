import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer

from .encoders import EnhancedEEGEncoder, ProjectionHead


class ContrastiveEEG2Text(nn.Module):
    """
    Contrastive EEG-to-Text Model
    
    This model encodes EEG signals and text into a shared embedding space
    and applies contrastive learning to align the representations.
    """
    def __init__(self, eeg_input_size, hidden_size=768, proj_hidden=256, proj_output=128,
                 temperature=0.07, num_layers=6, num_heads=8, dropout=0.1):
        """
        Initialize the contrastive model
        
        Args:
            eeg_input_size: Dimension of input EEG features
            hidden_size: Hidden dimension of transformers
            proj_hidden: Hidden dimension of projection heads
            proj_output: Output dimension of projection heads
            temperature: Temperature parameter for contrastive loss
            num_layers: Number of transformer layers for EEG encoder
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(ContrastiveEEG2Text, self).__init__()
        
        # Text encoder (BART)
        self.bart = BartModel.from_pretrained('facebook/bart-base')
        
        # EEG encoder
        self.eeg_encoder = EnhancedEEGEncoder(
            input_size=eeg_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Projection heads
        self.eeg_projection = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=proj_hidden,
            output_dim=proj_output,
            dropout=dropout
        )
        
        self.text_projection = ProjectionHead(
            input_dim=hidden_size, 
            hidden_dim=proj_hidden,
            output_dim=proj_output,
            dropout=dropout
        )
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
    def forward(self, eeg_input, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            eeg_input: Input EEG features (batch_size, seq_len, feature_dim)
            input_ids: Input token IDs for text
            attention_mask: Attention mask for text tokens
            
        Returns:
            Dictionary with loss, embeddings, and metrics
        """
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_input)
        eeg_embeddings = self.eeg_projection(eeg_features)
        
        # Encode text using BART
        text_outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get text embeddings from BART's output (use CLS token)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Calculate similarity matrix
        logits = torch.matmul(eeg_embeddings, text_embeddings.T) / self.temperature
        
        # Prepare for contrastive loss (InfoNCE)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)  # EEG to text
        loss_t = F.cross_entropy(logits.T, labels)  # Text to EEG
        loss = (loss_i + loss_t) / 2.0
        
        # Calculate accuracy
        predictions_i = torch.argmax(logits, dim=1)
        predictions_t = torch.argmax(logits.T, dim=1)
        accuracy_i = (predictions_i == labels).float().mean()
        accuracy_t = (predictions_t == labels).float().mean()
        accuracy = (accuracy_i + accuracy_t) / 2.0
        
        return {
            'loss': loss,
            'logits': logits,
            'eeg_embeddings': eeg_embeddings,
            'text_embeddings': text_embeddings,
            'accuracy': accuracy,
            'temperature': self.temperature,
            'eeg_to_text_acc': accuracy_i,
            'text_to_eeg_acc': accuracy_t
        }
    
    def freeze_bart_layers(self, unfreeze_embeddings=True, unfreeze_first_layer=True):
        """
        Freeze BART layers for first training step
        
        Args:
            unfreeze_embeddings: Whether to unfreeze embedding parameters
            unfreeze_first_layer: Whether to unfreeze first encoder layer
        """
        # Freeze all parameters first
        for param in self.bart.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze components
        if unfreeze_embeddings:
            # Unfreeze shared embeddings
            for param in self.bart.shared.parameters():
                param.requires_grad = True
            
            # Unfreeze positional embeddings
            for param in self.bart.encoder.embed_positions.parameters():
                param.requires_grad = True
                
        if unfreeze_first_layer:
            # Unfreeze first encoder layer
            for param in self.bart.encoder.layers[0].parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters for second training step"""
        for param in self.parameters():
            param.requires_grad = True
            
    def encode_eeg(self, eeg_input):
        """
        Encode EEG data to the embedding space
        
        Args:
            eeg_input: Input EEG features (batch_size, seq_len, feature_dim)
            
        Returns:
            Normalized EEG embeddings
        """
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg_input)
            eeg_embeddings = self.eeg_projection(eeg_features)
            normalized_embeddings = F.normalize(eeg_embeddings, dim=-1)
            
        return normalized_embeddings
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text to the embedding space
        
        Args:
            input_ids: Input token IDs for text
            attention_mask: Attention mask for text tokens
            
        Returns:
            Normalized text embeddings
        """
        with torch.no_grad():
            text_outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            text_features = text_outputs.last_hidden_state[:, 0, :]
            text_embeddings = self.text_projection(text_features)
            normalized_embeddings = F.normalize(text_embeddings, dim=-1)
            
        return normalized_embeddings