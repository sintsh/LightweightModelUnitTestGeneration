# Documentation for Optimized Unit Test Generator Code

## Overview
This code is a Python program that uses a machine learning model called PLBART to automatically generate unit test cases for Java code. It uses advanced techniques like distributed training across multiple GPUs and Parameter-Efficient Fine-Tuning (PEFT) to make the model more efficient. The program supports three PEFT methods: LoRA, QLoRA, and Adapter, which help fine-tune the model with less memory and computation.

The code loads a dataset, preprocesses it, trains the model, and can generate JUnit test cases for given Java code. It’s designed to run on multiple GPUs for faster processing and includes features like early stopping, logging, and error handling to make training reliable.

## Prerequisites
To run this code, you need the following:

- **Hardware**: A computer with one or more NVIDIA GPUs (CUDA-compatible).
- **Software**:
  - Python 3.8 or higher
  - PyTorch (with CUDA support)
  - Install required Python packages:
    ```bash
    pip install torch transformers datasets peft bitsandbytes tqdm tensorboard
    ```
- **Dataset**: The code expects a dataset from the Hugging Face hub (`sintsh/m2test_dataset`). Make sure you have internet access to download it.
- **Environment Variables**: Set up environment variables for distributed training:
  - `WORLD_SIZE`: Number of GPUs available.
  - `LOCAL_RANK`: Rank of the current GPU (0, 1, 2, etc.).
  - Example: `export WORLD_SIZE=4` and `export LOCAL_RANK=0` for the first GPU.
- **Optional**: TensorBoard for visualizing training progress.

## How to Run the Code
Follow these steps to set up and run the program:

1. **Install Dependencies**:
   Run the following command to install all required packages:
   ```bash
   pip install torch transformers datasets peft bitsandbytes tqdm tensorboard
   ```

2. **Set Up Distributed Training**:
   If you have multiple GPUs, use the `torchrun` command to launch the script. For example, to run on 2 GPUs:
   ```bash
   torchrun --nproc_per_node=2 OptimizedUnitTestGeneratorNew.py --peft_type=lora
   ```
   Replace `your_script_name.py` with the name of the Python file containing the code. The `--peft_type` argument can be `lora`, `qlora`, or `adapter`.

3. **Single GPU or CPU**:
   If you’re using a single GPU or CPU, you can run the script directly:
   ```bash
   python OptimizedUnitTestGeneratorNew.py --peft_type=lora
   ```

4. **Monitor Training**:
   - The code saves training logs to a directory like `./peft_logs/lora_YYYYMMDD`.
   - Use TensorBoard to view training progress:
     ```bash
     tensorboard --logdir=./peft_logs
     ```
     Open the provided URL (usually `http://localhost:6006`) in a browser to see loss and learning rate graphs.

5. **Generate Test Cases**:
   After training, the code automatically generates a sample JUnit test case for a simple Java `Calculator` class (defined in the code). To generate test cases for your own Java code, modify the `java_code` variable in the `main()` function with your Java code.

6. **Output**:
   - Trained model checkpoints are saved to `./peft_models/[peft_type]_YYYYMMDD`.
   - Generated test cases are printed to the console.
   - Training progress and metrics (like loss) are logged to TensorBoard and printed during training.

## Code Workflow
Here’s what the code does in simple terms:

1. **Setup**: Initializes distributed training across multiple GPUs using PyTorch’s distributed module.
2. **Model Loading**: Loads the PLBART model and tokenizer from Hugging Face (`sintsh/AKPlbartT`). Applies PEFT (LoRA, QLoRA, or Adapter) to make training efficient.
3. **Dataset Preparation**: Downloads a dataset (`sintsh/m2test_dataset`), splits it into training and validation sets, and preprocesses it for the model.
4. **Training**: Trains the model using the dataset, with features like:
   - Gradient accumulation for handling large batches.
   - Early stopping if the model stops improving.
   - Learning rate scheduling for better convergence.
   - Logging to TensorBoard for monitoring.
5. **Test Case Generation**: After training, generates JUnit test cases for provided Java code, with error handling and post-processing to ensure valid output.
6. **Cleanup**: Safely shuts down distributed processes and clears GPU memory.

## Example Output
For the sample Java code:
```java
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public boolean isEven(int n) { return n % 2 == 0; }
}
```

The code might generate a test case like:
```
   
	 @Test
	 public void add() throws Exception {
	 Calculator calculator = new Calculator ();
	 assertEquals (0 , calculator .add(0 , 0));
	 assertEquals (1 , calculator .add(0 , 1));
	 }

    @Test
    public void isEven() {
        Calculator calc = new Calculator();
        assertTrue(calc.isEven(4));
        assertFalse(calc.isEven(5));
    }

```

## Notes
- **PEFT Types**:
  - `lora`: Lightweight fine-tuning, good for most setups.
  - `qlora`: Quantized LoRA, uses less memory (good for low-memory GPUs).
  - `adapter`: Adds small layers to the model, another efficient fine-tuning method.
- **Error Handling**: The code handles GPU memory issues and provides fallback test cases if generation fails.
- **Customization**: You can tweak hyperparameters (e.g., learning rate, batch size) in the `_initialize_config` method.

## Troubleshooting
- **Out of Memory Errors**: Reduce `batch_size` or use `qlora` for lower memory usage.
- **Dataset Issues**: Ensure you have internet access to download the dataset from Hugging Face.
- **Distributed Training Errors**: Verify that `WORLD_SIZE` and `LOCAL_RANK` are set correctly.
- **TensorBoard Not Working**: Check that the log directory exists and TensorBoard is installed.

