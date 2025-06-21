# Documentation for PLBART Atlas Data Training Script

## Overview
This Python script trains a PLBART model (`uclanlp/plbart-base`) to generate Java code assertions, such as those used in JUnit tests, using datasets stored on Google Drive. It processes text data (Java code snippets) from CSV files, tokenizes it, and fine-tunes the model for tasks like generating assertion-based test cases. The script leverages the Hugging Face `transformers` library for training, supports GPU acceleration, and pushes the trained model to the Hugging Face Hub. It includes data preprocessing, training with masked language modeling, TensorBoard logging, and model saving.

The script assumes the datasets contain Java code snippets with assertions (e.g., JUnit test cases) in CSV format. It tokenizes the data, groups it into fixed-size blocks, and trains the model to understand and generate assertion-related Java code.

## Prerequisites
To run this script, you need the following:

- **Hardware**: A computer with a GPU (recommended for faster training) or CPU. The script uses GPU 1 (`cuda:1`) if available.
- **Software**:
  - Python 3.8 or higher
  - Install required Python packages:
    ```bash
    pip install torch transformers datasets pandas huggingface_hub
    ```
  - TensorBoard for visualizing training progress:
    ```bash
    pip install tensorboard
    ```
- **Dataset**: Two CSV files containing Java code with assertions:
  - Training dataset: `/research_project/A3Test/SinteA3Test/AtlasDataset/training_dataset.csv`
  - Validation dataset: `/research_project/A3Test/SinteA3Test/AtlasDataset/testing_dataset.csv`
  - Ensure these files are accessible on your system or Google Drive (mounted locally).
- **Hugging Face Account**: A Hugging Face token to push the trained model to the Hub. Replace `"your_token_here"` in the script with your actual token.
- **Storage**: Write access to the current directory for saving the model and logs.

## How to Run the Code
Follow these steps to set up and run the script:

1. **Install Dependencies**:
   Install the required packages:
   ```bash
   pip install torch transformers datasets pandas huggingface_hub tensorboard
   ```

2. **Set Up Datasets**:
   Ensure the training and validation CSV files are located at the specified paths or update the `path_to_train` and `path_to_validation` variables in the script to point to your dataset locations.

3. **Set Up Hugging Face Token**:
   Replace `"your_token_here"` in the script with your Hugging Face API token. You can get a token from your Hugging Face account settings (https://huggingface.co/settings/tokens).

4. **Run the Script**:
   Execute the script from the command line:
   ```bash
   python plbart_atlas_data.py
   ```
   This will:
   - Load and preprocess the datasets containing Java code with assertions.
   - Train the PLBART model for 5 epochs to learn assertion generation.
   - Save the model and tokenizer to the current directory.
   - Push the model to the Hugging Face Hub under the repository name `AKPlbart`.
   - Log training progress to `./logs` for TensorBoard.

5. **Monitor Training**:
   View training progress using TensorBoard:
   ```bash
   tensorboard --logdir=./logs
   ```
   Open the provided URL (usually `http://localhost:6006`) in a browser to see training and validation loss.

6. **Output**:
   - The trained model and tokenizer are saved in the current directory (`./`).
   - Training logs are saved in `./logs`.
   - The model is pushed to the Hugging Face Hub as `AKPlbart`.
   - A sample of 10 random dataset entries (Java code with assertions) is printed to the console during execution.

## Code Workflow
Here’s what the script does in simple terms:

1. **Setup**:
   - Selects GPU 1 (`cuda:1`) if available, otherwise uses CPU.
   - Loads training and validation datasets from CSV files using the `datasets` library.

2. **Data Inspection**:
   - Displays 10 random Java code snippets with assertions from the training dataset to verify its contents.

3. **Model and Tokenizer**:
   - Loads the PLBART model and tokenizer (`uclanlp/plbart-base`) from Hugging Face.
   - Moves the model to the specified device (GPU 1 or CPU).

4. **Data Preprocessing**:
   - Tokenizes the Java code (with assertions) with a maximum length of 1024 tokens, truncating longer sequences.
   - Groups tokenized data into fixed-size blocks (1024 tokens) for efficient training.
   - Prepares labels for masked language modeling by copying input IDs.

5. **Training**:
   - Configures training with the `TrainingArguments` class, specifying:
     - 5 epochs, learning rate of 2e-5, batch size of 4, and mixed precision (if GPU is available).
     - Logging to TensorBoard every 1000 steps.
     - Saving the model at each epoch (keeping only the latest checkpoint).
     - Pushing the model to the Hugging Face Hub (`AKPlbart`).
   - Uses a data collator to apply masked language modeling (15% of tokens masked).
   - Trains the model to learn patterns in Java code assertions.

6. **Saving and Sharing**:
   - Saves the trained model and tokenizer to the current directory.
   - Pushes the model to the Hugging Face Hub for sharing.

## Example Output
- **Dataset Sample**: When the script runs, it prints a sample of 10 random rows from the training dataset, displayed as a pandas DataFrame containing Java code with assertions.
- **Training Progress**: Training logs are written to `./logs`, visible in TensorBoard, showing metrics like loss per epoch.
- **Console Output**: Example of dataset sample (assuming the CSV contains a `text` column with Java assertions):
  ```
     text
  0  public class CalculatorTest { 
		@Test 
		public void testAdd() { 
			Calculator calc = new Calculator(); 
			assertEquals(5, calc.add(2, 3)); 
			} 
	}
  1  public class StringUtilTest { 
		@Test 
		public void testIsEmpty() { 
			StringUtil util = new StringUtil(); 
			assertTrue(util.isEmpty("")); 
			} 
	}
  ...
  ```
- **Saved Files**: The current directory will contain model files (e.g., `pytorch_model.bin`, `config.json`) and tokenizer files (e.g., `vocab.json`, `merges.txt`).
- **Hugging Face Hub**: The model is uploaded to `AKPlbart` on your Hugging Face account.

## Notes
- **Dataset Format**: The CSV files should have a `text` column containing Java code with assertions (e.g., JUnit test cases). Ensure the files are correctly formatted.
- **GPU Selection**: The script uses `cuda:1` (GPU 1). If you have only one GPU, change `cuda:1` to `cuda:0`. For multiple GPUs, adjust as needed.
- **Batch Size**: The batch size is set to 4 to avoid memory issues. Increase it if your GPU has more memory, or decrease it if you encounter out-of-memory errors.
- **Hugging Face Token**: You must provide a valid token to push to the Hub. Without it, the push will fail, but local training and saving will still work.
- **Mixed Precision**: Enabled automatically on GPU to speed up training and reduce memory usage.
- **Assertion Focus**: The model is trained to understand and generate Java assertions, making it suitable for generating JUnit test cases or similar tasks.

## Troubleshooting
- **Dataset Not Found**: Ensure the CSV files exist at the specified paths. If using Google Drive, mount it correctly or update the paths.
- **Hugging Face Login Error**: Verify your token is correct and has write permissions for the `AKPlbart` repository.
- **Out of Memory Errors**: Reduce `per_device_train_batch_size` (e.g., to 2) or use a smaller `block_size` (e.g., 512).
- **TensorBoard Issues**: Ensure the `./logs` directory exists and TensorBoard is installed.
- **Slow Tokenization**: If preprocessing is slow, reduce `num_proc` (e.g., to 2) to use fewer CPU cores.
