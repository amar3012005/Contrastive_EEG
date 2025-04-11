import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedEEGEncoder(nn.Module):
    """
    Enhanced EEG encoder using transformer architecture
    """
    def __init__(self, input_size, hidden_size, num_layers=6, num_heads=8, 
                 dropout=0.1, dim_feedforward=2048):
        """
        Initialize the EEG encoder
        
        Args:
            input_size: Dimension of input EEG features
            hidden_size: Hidden dimension of the transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            dim_feedforward: Dimension of feed-forward network
        """
        super(EnhancedEEGEncoder, self).__init__()
        
        # Input projection layer to transform EEG features to hidden dimension
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2000, hidden_size))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x, attention_mask=None):
        """
        Forward pass
        
        Args:
            x: Input EEG features (batch_size, seq_len, feature_dim)
            attention_mask: Mask for attention mechanism (1 for tokens to attend to, 0 for padding)
            
        Returns:
            Encoded EEG representation
        """
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # Add positional embedding (truncated to sequence length)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # If attention mask is provided, convert it to transformer format
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
            # where 1 means attend, 0 means mask
            attention_mask = attention_mask.unsqueeze(1).repeat(1, x.size(1), 1)
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')
            ).masked_fill(attention_mask == 1, float(0.0))
        
        # Apply transformer layers
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """
        Initialize the projection head
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features
            
        Returns:
            Projected features
        """
        return self.projection(x)