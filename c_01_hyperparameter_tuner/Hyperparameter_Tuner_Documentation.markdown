# Documentation for Hyperparameter Tuner Code

## Overview
This Python program is designed to fine-tune a PLBART model for generating unit test cases, focusing on optimizing hyperparameters for efficient training. It uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and supports QLoRA for quantization on GPUs. The code systematically tests different values for LoRA rank, learning rate, batch size, and quantization methods to find the best configuration for model performance. It’s built to work with a specific dataset and can run on either a GPU or CPU.

The program includes a `HyperparameterTuner` class that manages the tuning process, evaluates configurations, and trains the model with the best-found parameters. It’s designed for ease of use, with clear output to track the tuning process and final training configuration.

## Prerequisites
To run this code, you need the following:

- **Hardware**: A computer with a GPU (optional, but recommended for QLoRA and faster training) or CPU.
- **Software**:
  - Python 3.8 or higher
  - PyTorch (with CUDA support for GPU usage)
  - Install required Python packages:
    ```bash
    pip install torch transformers datasets peft
    ```
- **Dataset**: The code uses a dataset from the Hugging Face hub (`sintsh/m2test_dataset`). Ensure internet access to download it.
- **Command-Line Arguments**: The script accepts a `--tune` flag to run hyperparameter tuning. Without it, the script trains the model using default optimal parameters.

## How to Run the Code
Follow these steps to set up and run the program:

1. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install torch transformers datasets peft
   ```

2. **Run Hyperparameter Tuning**:
   To perform hyperparameter tuning, use the `--tune` flag:
   ```bash
   python your_script_name.py --tune
   ```
   Replace `your_script_name.py` with the name of the Python file containing the code. This will test different values for LoRA rank, learning rate, batch size, and quantization (if on GPU).

3. **Train with Optimal Parameters**:
   To train the model directly with the default optimal parameters, run:
   ```bash
   python your_script_name.py
   ```
   This skips tuning and uses the pre-defined best parameters.

4. **Output**:
   - During tuning, the script prints scores for each hyperparameter configuration tested (e.g., LoRA rank, learning rate).
   - If tuning is skipped, it displays a formatted header with the optimal parameters and starts the training process.
   - The dataset is loaded automatically from Hugging Face (`sintsh/m2test_dataset`).

## Code Workflow
Here’s what the code does in simple terms:

1. **Initialization**: The `HyperparameterTuner` class sets up default optimal hyperparameters (LoRA rank, learning rate, batch size, etc.) and checks if a GPU is available.
2. **Model Setup**: Initializes the PLBART model (`sintsh/AKPlbart`) with LoRA or QLoRA configurations, depending on the hyperparameters being tested.
3. **Hyperparameter Tuning** (if `--tune` is used):
   - Tests LoRA ranks (4, 8, 16, 32) to adjust model complexity.
   - Tests learning rates (1e-5, 3e-5, 5e-5) to control training speed.
   - Tests batch sizes (16, 32, 64) to optimize data processing.
   - Tests quantization methods (`nf4`, `fp16`) on GPUs to reduce memory usage.
   - Prints scores for each configuration (using a placeholder evaluation function).
4. **Training**: If tuning is skipped, trains the model with the best parameters (LoRA rank: 8, learning rate: 3e-5, batch size: 32, quantization: nf4).
5. **Output**: Displays the optimal hyperparameters and training configuration in a formatted header.

## Example Output
When running without `--tune`, the output looks like:
```
==================================================
         TRAINING WITH OPTIMAL PARAMETERS         
==================================================

Optimal Parameters Configuration:
• LoRA Rank: 8
• Adapter Size: 128
• Learning Rate: 3e-05
• Batch Size: 32
• QLoRA Quantization: NF4
--------------------------------------------------
          Starting Training Process...           
--------------------------------------------------
```

When running with `--tune`, the output includes:
```
Starting hyperparameter tuning...

Tuning LoRA Rank:
Rank 4: Score 0.85
Rank 8: Score 0.89
Rank 16: Score 0.87
Rank 32: Score 0.84

Tuning Learning Rate:
LR 1e-05: Score 0.86
LR 3e-05: Score 0.88
LR 5e-05: Score 0.83

Tuning Batch Size:
Batch Size 16: Score 0.87
Batch Size 32: Score 0.90
Batch Size 64: Score 0.85

Testing QLoRA Quantization:
Quant nf4: Score 0.89, Memory 2.3GB
Quant fp16: Score 0.87, Memory 3.1GB

Optimal Hyperparameters Found:
      lora_rank: 8
    adapter_size: 128
  learning_rate: 3e-05
     batch_size: 32
    qlora_quant: nf4
```

## Notes
- **Hyperparameters**:
  - **LoRA Rank**: Controls the size of the LoRA adaptation (higher = more complex).
  - **Learning Rate**: Affects how fast the model learns (too high can cause instability).
  - **Batch Size**: Determines how many samples are processed at once (affects memory usage).
  - **QLoRA Quantization**: Reduces memory usage on GPUs (`nf4` is more memory-efficient than `fp16`).
- **Evaluation**: The `evaluate_config` and `train_and_eval` methods currently use placeholder scores (random values). Replace them with actual metrics (e.g., validation loss) for real-world use.
- **Dataset**: The code uses `sintsh/m2test_dataset` from Hugging Face, which contains data for training test case generation models.
- **GPU/CPU Support**: Automatically detects and uses GPU if available; falls back to CPU otherwise.

## Troubleshooting
- **Missing Dependencies**: Ensure all required packages (`torch`, `transformers`, etc.) are installed.
- **Dataset Issues**: Verify internet access to download `sintsh/m2test_dataset` from Hugging Face.
- **Memory Errors on GPU**: Use a smaller batch size (e.g., 16) or enable QLoRA with `nf4` quantization.
- **Placeholder Functions**: The `evaluate_config` and `train_and_eval` methods need real evaluation logic for accurate tuning.
