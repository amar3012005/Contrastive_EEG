import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class TwoStepTrainer:
    """
    Two-step trainer for Contrastive EEG-to-Text model
    
    Step 1: Train with frozen BART except embeddings and first layer
    Step 2: Fine-tune all parameters
    """
    def __init__(self, model, train_loader, val_loader, config, device=None):
        """
        Initialize trainer
        
        Args:
            model: Contrastive EEG-to-Text model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to run on (will auto-detect if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'temperature': [],
            'eeg_to_text_acc': [],
            'text_to_eeg_acc': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
        
    def train(self):
        """
        Run the two-step training process
        
        Returns:
            Training history
        """
        # Step 1: Initial training with frozen BART
        logger.info("Starting Step 1: Training with frozen BART layers")
        self.model.freeze_bart_layers(
            unfreeze_embeddings=self.config['text_encoder']['unfreeze_embeddings'],
            unfreeze_first_layer=self.config['text_encoder']['unfreeze_first_layer']
        )
        
        # Create optimizer for step 1
        if self.config['training']['optimizer'] == 'adam':
            optimizer_step1 = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['training']['step1']['lr']
            )
        else:  # Default to SGD
            optimizer_step1 = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['training']['step1']['lr'],
                momentum=self.config['training']['momentum']
            )
        
        # Create scheduler for step 1
        scheduler_step1 = optim.lr_scheduler.StepLR(
            optimizer_step1, 
            step_size=self.config['training']['step1']['lr_step'],
            gamma=self.config['training']['step1']['lr_gamma']
        )
        
        # Perform step 1 training
        logger.info("Step 1 training")
        self._train_loop(
            optimizer_step1,
            scheduler_step1,
            self.config['training']['step1']['epochs'],
            "step1"
        )
        
        # Step 2: Fine-tuning with all parameters
        logger.info("Starting Step 2: Fine-tuning all parameters")
        self.model.unfreeze_all()
        
        # Create optimizer for step 2
        if self.config['training']['optimizer'] == 'adam':
            optimizer_step2 = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['step2']['lr']
            )
        else:  # Default to SGD
            optimizer_step2 = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['step2']['lr'],
                momentum=self.config['training']['momentum']
            )
        
        # Create scheduler for step 2
        scheduler_step2 = optim.lr_scheduler.StepLR(
            optimizer_step2, 
            step_size=self.config['training']['step2']['lr_step'],
            gamma=self.config['training']['step2']['lr_gamma']
        )
        
        # Perform step 2 training
        logger.info("Step 2 training")
        self._train_loop(
            optimizer_step2,
            scheduler_step2,
            self.config['training']['step2']['epochs'],
            "step2"
        )
        
        # Return training history
        return self.history
    
    def _train_loop(self, optimizer, scheduler, epochs, step_name):
        """
        Training loop for a single step
        
        Args:
            optimizer: Optimizer to use
            scheduler: Learning rate scheduler
            epochs: Number of epochs
            step_name: Name of the step ("step1" or "step2")
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            eeg_to_text_acc = 0.0
            text_to_eeg_acc = 0.0
            
            # Use tqdm for a progress bar
            train_iterator = tqdm(
                self.train_loader, 
                desc=f"{step_name} Epoch {epoch+1}/{epochs} [Train]",
                leave=False
            )
            
            for batch_idx, (eeg_data, input_ids, attention_mask) in enumerate(train_iterator):
                # Move data to device
                eeg_data = eeg_data.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(eeg_data, input_ids, attention_mask)
                loss = outputs['loss']
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_acc += outputs['accuracy'].item()
                eeg_to_text_acc += outputs['eeg_to_text_acc'].item()
                text_to_eeg_acc += outputs['text_to_eeg_acc'].item()
                
                # Update progress bar
                train_iterator.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{outputs['accuracy'].item():.4f}"
                })
            
            # Calculate average metrics
            num_batches = len(self.train_loader)
            train_loss /= num_batches
            train_acc /= num_batches
            eeg_to_text_acc /= num_batches
            text_to_eeg_acc /= num_batches
            
            # Validation phase
            val_loss, val_acc, e2t_acc, t2e_acc, temperature = self._validate(
                epoch, epochs, step_name
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log metrics
            logger.info(f"{step_name} Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"EEG→Text: {eeg_to_text_acc:.4f}, Text→EEG: {text_to_eeg_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                       f"Temp: {temperature:.4f}, LR: {current_lr:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['temperature'].append(temperature)
            self.history['eeg_to_text_acc'].append(eeg_to_text_acc)
            self.history['text_to_eeg_acc'].append(text_to_eeg_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(f"best_model_{step_name}.pt", epoch, val_loss)
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def _validate(self, epoch, epochs, step_name):
        """
        Validate the model
        
        Args:
            epoch: Current epoch
            epochs: Total epochs
            step_name: Name of the step
            
        Returns:
            val_loss, val_acc, eeg_to_text_acc, text_to_eeg_acc, temperature
        """
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        eeg_to_text_acc = 0.0
        text_to_eeg_acc = 0.0
        
        # Use tqdm for a progress bar
        val_iterator = tqdm(
            self.val_loader, 
            desc=f"{step_name} Epoch {epoch+1}/{epochs} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, (eeg_data, input_ids, attention_mask) in enumerate(val_iterator):
                # Move data to device
                eeg_data = eeg_data.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(eeg_data, input_ids, attention_mask)
                
                # Update metrics
                val_loss += outputs['loss'].item()
                val_acc += outputs['accuracy'].item()
                eeg_to_text_acc += outputs['eeg_to_text_acc'].item()
                text_to_eeg_acc += outputs['text_to_eeg_acc'].item()
                
                # Update progress bar
                val_iterator.set_postfix({
                    'loss': f"{outputs['loss'].item():.4f}", 
                    'acc': f"{outputs['accuracy'].item():.4f}"
                })
        
        # Calculate average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_acc /= num_batches
        eeg_to_text_acc /= num_batches
        text_to_eeg_acc /= num_batches
        temperature = outputs['temperature'].item()
        
        return val_loss, val_acc, eeg_to_text_acc, text_to_eeg_acc, temperature
    
    def _save_checkpoint(self, filename, epoch, val_loss):
        """
        Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
            epoch: Current epoch
            val_loss: Validation loss
        """
        # Create save directory if it doesn't exist
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config  # Include config for easy loading
        }
        
        # Save checkpoint
        torch.save(checkpoint, save_dir / filename)