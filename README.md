# Contrastive EEG-to-Text Model

This project implements a contrastive learning approach for aligning EEG signals with their corresponding textual representations using the ZuCo dataset.

## Project Structure

```
Contrastive_EEG/
│
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration
│
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   │   └── data_loader.py   # ZuCo dataset loader
│   │
│   ├── models/              # Model implementations
│   │   ├── encoders.py      # EEG encoder and projection head
│   │   └── contrastive_model.py # Main contrastive model
│   │
│   └── training/            # Training utilities
│       └── trainer.py       # Two-step trainer implementation
│
├── utils/                   # Utility functions
│   ├── config_utils.py      # Configuration utilities
│   └── visualization.py     # Visualization tools
│
├── scripts/                 # Executable scripts
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
│
└── notebooks/               # Jupyter notebooks for demonstration
    └── Contrastive_EEG_Demo.ipynb  # Demo notebook
```

## Features

- **Contrastive Learning Framework**: Aligns EEG signals with text representations in a shared embedding space
- **Two-Step Training Strategy**:
  1. First step trains with frozen BART layers (except embeddings and first layer)
  2. Second step fine-tunes all parameters
- **Transformer-Based EEG Encoder**: Processes EEG signals using transformer architecture
- **InfoNCE Loss**: Maximizes mutual information between EEG and text representations
- **Visualization Tools**: Tools for embedding visualization and similarity matrix analysis

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers (Hugging Face)
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

## Dataset

This project uses the ZuCo dataset, which contains EEG recordings during reading tasks:
- Task1-SR (Sentence Reading)
- Task2-NR (Natural Reading)
- Task3-TSR (Task-Specific Reading)
- TaskNRv2 (Natural Reading 2.0)

## Usage

### Configuration

Modify `config/config.yaml` to adjust dataset paths, model architecture, and training parameters.

### Training

To train the model:

```bash
python scripts/train.py --config config/config.yaml
```

Optional arguments:
- `--output-dir`: Output directory for results (overrides config)
- `--data-path`: Path to ZuCo dataset (overrides config)
- `--batch-size`: Batch size (overrides config)
- `--step1-epochs`: Number of epochs for step 1 (overrides config)
- `--step2-epochs`: Number of epochs for step 2 (overrides config)
- `--seed`: Random seed (default: 42)

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --model-path <path_to_model> --data-path <path_to_data>
```

Required arguments:
- `--model-path`: Path to trained model checkpoint
- `--data-path`: Path to ZuCo dataset

Optional arguments:
- `--output-dir`: Directory to save evaluation results (default: output/evaluation)
- `--batch-size`: Batch size for evaluation (default: 8)
- `--seed`: Random seed (default: 42)

## Model Architecture

The model consists of:

1. **EEG Encoder**: Transformer-based encoder for EEG signals
   - Input projection layer
   - Positional embeddings
   - Transformer encoder layers
   - Global average pooling

2. **Text Encoder**: BART-based encoder for text
   - Pre-trained BART model
   - CLS token representation

3. **Projection Heads**: For both EEG and text
   - Layer normalization
   - Linear projection
   - GELU activation
   - Dropout
   - Second linear projection
   - Final layer normalization

## Training Process

The training uses a two-step approach:
1. **Step 1**: Train EEG encoder and selected BART components with higher learning rate
2. **Step 2**: Fine-tune all parameters with lower learning rate

Each step uses:
- SGD optimizer with momentum (or Adam)
- StepLR scheduler for learning rate decay
- InfoNCE contrastive loss

## Results

After training, the model produces:
- Training and validation metrics (loss, accuracy)
- Embedding visualizations (t-SNE)
- Similarity matrix visualizations

## References

- ZuCo dataset: [ZuCo: A simultaneous EEG and eye-tracking resource for natural sentence reading](https://www.nature.com/articles/sdata2018291)
- InfoNCE loss: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- BART: [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
