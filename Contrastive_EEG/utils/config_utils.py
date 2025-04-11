import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def validate_config(config):
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises exceptions otherwise
    """
    # Check required sections
    required_sections = [
        'dataset', 'eeg_encoder', 'text_encoder', 
        'projection', 'training'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset config
    if 'data_path' not in config['dataset']:
        raise ValueError("Missing data_path in dataset configuration")
    
    # Check if data path exists
    data_path = Path(config['dataset']['data_path'])
    if not os.path.isabs(data_path):
        # Resolve relative path
        base_dir = Path(os.getcwd())
        data_path = base_dir / data_path
        
    if not data_path.exists():
        logger.warning(f"Data path {data_path} does not exist. Check the path before running.")
    
    # Validate training config
    required_training = ['step1', 'step2', 'optimizer', 'temperature']
    for item in required_training:
        if item not in config['training']:
            raise ValueError(f"Missing {item} in training configuration")
    
    # Validate steps
    for step in ['step1', 'step2']:
        required_step_params = ['epochs', 'lr', 'lr_step', 'lr_gamma']
        for param in required_step_params:
            if param not in config['training'][step]:
                raise ValueError(f"Missing {param} in {step} configuration")
    
    return True

def process_paths(config):
    """
    Process relative paths in configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with absolute paths
    """
    # Create a deep copy to avoid modifying the original
    processed_config = config.copy()
    
    # Process data path
    if 'data_path' in processed_config['dataset']:
        data_path = Path(processed_config['dataset']['data_path'])
        if not os.path.isabs(data_path):
            # Resolve relative path
            base_dir = Path(os.getcwd())
            processed_config['dataset']['data_path'] = str(base_dir / data_path)
    
    # Process save directory
    if 'save_dir' in processed_config['training']:
        save_dir = Path(processed_config['training']['save_dir'])
        if not os.path.isabs(save_dir):
            # Resolve relative path
            base_dir = Path(os.getcwd())
            processed_config['training']['save_dir'] = str(base_dir / save_dir)
    
    return processed_config

def get_config(config_path):
    """
    Load, validate and process configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Processed configuration dictionary
    """
    config = load_config(config_path)
    validate_config(config)
    processed_config = process_paths(config)
    
    return processed_config