# Guide to Running the Contrastive EEG-to-Text Scripts

This guide will walk you through the process of running the Contrastive EEG-to-Text model implementation, from environment setup to training and evaluation.

## Prerequisites

Before running the scripts, ensure you have:

1. Python 3.8+ installed
2. Required packages installed (see below)
3. Access to the ZuCo dataset (properly located in the ZuCo directory)

## Environment Setup

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

### Fixing NumPy Issues

If you encounter errors with NumPy (like "No module named 'numpy._core'"), try the following:

```bash
# Uninstall numpy
pip uninstall -y numpy

# Reinstall numpy
pip install numpy --force-reinstall
```

## Running the Demo Notebook

The demo notebook `Contrastive_EEG_Demo.ipynb` provides an interactive demonstration of the model. To run it:

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/Contrastive_EEG_Demo.ipynb` and open it

3. Run cells sequentially from top to bottom

4. For the data loading cell, if you encounter NumPy errors, you may need to restart the kernel after fixing the NumPy installation

## Running the Training Script

To train the model using the full training pipeline:

```bash
# Navigate to the Contrastive_EEG directory
cd Contrastive_EEG

# Run the training script with default configuration
python scripts/train.py --config config/config.yaml

# Or with custom parameters
python scripts/train.py --config config/config.yaml --output-dir custom_output --data-path ../ZuCo/task1-SR/pickle/task1-SR-dataset.pickle --batch-size 16 --step1-epochs 30 --step2-epochs 15
```

Key parameters:
- `--config`: Path to the config file
- `--output-dir`: Directory to save results
- `--data-path`: Path to ZuCo dataset
- `--batch-size`: Batch size for training
- `--step1-epochs`: Number of epochs for step 1 training
- `--step2-epochs`: Number of epochs for step 2 training
- `--seed`: Random seed (default: 42)

## Running the Evaluation Script

To evaluate a trained model:

```bash
# Navigate to the Contrastive_EEG directory
cd Contrastive_EEG

# Run the evaluation script
python scripts/evaluate.py --model-path output/contrastive_model/run_YYYYMMDD_HHMMSS/best_model_step2.pt --data-path ../ZuCo/task1-SR/pickle/task1-SR-dataset.pickle
```

Key parameters:
- `--model-path`: Path to trained model checkpoint (required)
- `--data-path`: Path to ZuCo dataset (required)
- `--output-dir`: Directory to save evaluation results
- `--batch-size`: Batch size for evaluation
- `--seed`: Random seed

## Troubleshooting

### Fixing Import Errors

If you encounter import errors:

1. Make sure you're running the scripts from the right directory (Contrastive_EEG)
2. Check that your Python environment has all dependencies installed
3. Verify the path to the ZuCo dataset is correct

### Common NumPy Issues

NumPy version conflicts are common. Try:

```bash
pip install numpy==1.24.3  # Specify a specific version
```

### CUDA Issues

If you encounter CUDA errors:

1. Verify your torch installation matches your CUDA version
2. If no GPU is available, the code will automatically fall back to CPU

### Dataset Path Issues

Ensure the ZuCo dataset path is correct. The default path is:

```
../ZuCo/task1-SR/pickle/task1-SR-dataset.pickle
```

You may need to adjust this path in the config file or via command line arguments.

## Monitoring Training

When training runs, you should see progress bars and logs showing:
- Loss values
- Accuracy metrics
- Learning rate updates
- Validation results

Model checkpoints and visualizations will be saved to the specified output directory.

## Viewing Results

After training or evaluation, results are stored in the output directory:
- Training metrics plots
- Embedding visualizations
- Similarity matrix visualizations
- Evaluation reports

You can view these files to analyze the model's performance.