# How to Use the Phase 4 Test Validation Script with Per-Project Results

This document explains a Python script that checks the quality of unit tests created by three computer models: `A3Test-LoRA`, `A3Test-QLoRA`, and `A3Test-Adapter`. These tests are for Java code from the Defects4J dataset (version 2.5.0), which includes six projects: `Lang`, `Chart`, `Gson`, `Csv`, `Json`, and `Cli`. The script uses tools to test if the generated tests are correct, cover the right parts of the code, and can find bugs. It also measures how fast the models work and how much computer memory they use. The script saves two output files: `evaluation_results.csv` (overall average scores for all projects combined) and `evaluation_results_per_project.csv` (scores for each project separately).

## What the Script Does

The script does the following:

1. **Loads the Models**: It loads the three trained models and a tool (called a tokenizer) to read and understand Java code.
2. **Creates Tests**: It uses each model to generate unit tests for Java methods (called focal methods) from the Defects4J dataset.
3. **Checks Test Quality**: It uses several tools to evaluate the tests:
   - `ANTLR4` checks if the tests are written correctly (like checking for typos in code).
   - `JUnit` checks if the tests give the right results compared to known correct tests.
   - `JaCoCo` measures how much of the Java code the tests cover.
   - `PITest` checks how well the tests find bugs by making small changes to the code.
4. **Tracks Performance**: It measures how long it takes to create each test and how much memory the computer uses.
5. **Saves Results**: It saves two files:
   - `evaluation_results.csv`: Shows average scores for all projects combined.
   - `evaluation_results_per_project.csv`: Shows scores for each project (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`) separately.

The script uses a file called `filtered_ground_truth.csv`, which lists the Java methods, their correct tests or descriptions, and the project they belong to (e.g., `Lang`). It processes around 5,278 methods across the six projects.

## How the Script Works

Here’s a simple breakdown of the script’s key parts:

- **Setup**: The script checks if a GPU (a fast computer chip) is available and uses it if possible. It loads the models, tokenizer, and Defects4J data from `filtered_ground_truth.csv`.
- **Checking Code**: The `validate_syntax` function uses `ANTLR4` to ensure the generated tests are written correctly and won’t break when run.
- **Creating Tests**: The `generate_test` function uses each model to create a test for a Java method, trying several options (called beam search) to make high-quality tests.
- **Testing and Measuring**: The `evaluate_tests` function:
  - Runs the tests with `JUnit` to check if they’re correct.
  - Uses `JaCoCo` to see how much code the tests cover.
  - Uses `PITest` to check how well the tests find bugs.
  - Tracks time and memory usage for each test.
  - Groups results by project (e.g., `Lang`, `Chart`) and overall.
- **Saving Results**: The script saves:
  - Overall average scores in `evaluation_results.csv`.
  - Project-specific scores in `evaluation_results_per_project.csv`.


## How to Run the Script

To use this script, follow these steps:

1. **Set Up Your Computer**:
   - Make sure you have **Python 3** installed (version 3.8 or higher works best).
   - Install the required Python libraries by running this command in your terminal or command prompt:
     ```bash
     pip install pandas torch transformers antlr4-python3-runtime psutil datasets
     ```
     - `pandas` creates the output CSV files.
     - `torch` and `transformers` manage the models and tokenizer.
     - `antlr4-python3-runtime` checks the test code for errors.
     - `psutil` tracks memory usage.
     - `datasets` loads the Defects4J data.
   - Install the Java tools (`ANTLR4`, `JUnit`, `JaCoCo`, `PITest`) and configure them so the script can use them. Check their websites for setup instructions:
     - [ANTLR4](https://www.antlr.org/)
     - [JUnit](https://junit.org/)
     - [JaCoCo](https://www.jacoco.org/)
     - [PITest](http://pitest.org/)
   - If you have a GPU (like an NVIDIA card), the script will use it to run faster. Otherwise, it will use your CPU, but it may be slower.

2. **Get the Required Files**:
   - **Defects4J Data**: You need a file called `filtered_ground_truth.csv`, which lists Java methods, their correct tests or descriptions, and the project they belong to (e.g., `Lang`, `Chart`). You can create this file using a separate preprocessing script (see its instructions for details). The file must have a `project` column to identify the project for each method.
   - **Trained Models**: You need the three trained models, saved in folders named `./UTLOAK-PLBART`, `./UTQOAK-PLBART`, and `./UTAOAK-PLBART`. These come from an earlier step (called Phase 3) where the models were trained.
   - Place `filtered_ground_truth.csv` and the model folders in the same folder where you’ll run the script.

3. **Add a `project` Column (if needed)**:
   - The script expects `filtered_ground_truth.csv` to have a `project` column (e.g., `Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`). If it doesn’t, add it manually or modify the preprocessing script to include it. Here’s a quick Python snippet to add the column (replace the logic to assign correct project names):
     ```python
     import pandas as pd
     df = pd.read_csv("./filtered_ground_truth.csv")
     # Assign project names based on source (e.g., map methods to project folders)
     df["project"] = "Lang"  # Replace with actual project names
     df.to_csv("./filtered_ground_truth.csv", index=False)
     ```
   - For example, methods from `defects4j/projects/Lang` should have `project=Lang`.

4. **Save the Script**:
   - Copy the Python code above into a file named `phase4_test_validation_per_project.py`.
   - Save it in a folder that contains `filtered_ground_truth.csv` and the model folders (`UTLOAK-PLBART`, `UTQOAK-PLBART`, `UTAOAK-PLBART`).

5. **Run the Script**:
   - Open a terminal or command prompt and navigate to the folder where you saved the script.
   - Run the script with this command:
     ```bash
     python phase4_test_validation_per_project.py
     ```
   - The script will:
     - Load each model and the Defects4J data.
     - Generate tests for each Java method.
     - Check the tests using `ANTLR4`, `JUnit`, `JaCoCo`, and `PITest`.
     - Save results in a folder called `test_validation_results`, with two files:
       - `evaluation_results.csv`: Average scores for all projects combined.
       - `evaluation_results_per_project.csv`: Scores for each project separately.

6. **Check the Output**:
   - While running, you’ll see messages like `Evaluating A3Test-LoRA...` and, at the end, `Aggregated Evaluation Results:` and `Per-Project Evaluation Results:` with tables of scores.
   - After it finishes, check the `test_validation_results` folder for:
     - **evaluation_results.csv**: Shows average scores for each model across all projects, with columns for `A3Test-LoRA`, `A3Test-QLoRA`, `A3Test-Adapter` and rows for metrics (CTC, FMC, LC, CA, MS, ATT, Memory).
     - **evaluation_results_per_project.csv**: Shows scores for each project, with columns `Project`, `Metric`, `A3Test-LoRA`, `A3Test-QLoRA`, `A3Test-Adapter`.
   - The metrics include:
     - **CTC**: Percentage of tests that are correct.
     - **FMC**: Percentage of methods covered by tests.
     - **LC**: Percentage of code lines covered.
     - **CA**: Percentage of test checks (assertions) that are correct.
     - **MS**: Percentage of bugs found by tests.
     - **ATT**: Average time to create a test (in seconds).
     - **Memory**: Maximum memory used (in gigabytes).

## What to Expect

- The script processes thousands of methods, creating and checking tests for each one. This can take a while, especially without a GPU.
- The results will show how well each model performed overall and for each project. Based on earlier tests, `A3Test-LoRA` should have the highest correctness scores (around 38% correct tests), while `A3Test-QLoRA` uses less memory (around 15GB).
- If a generated test isn’t written correctly, the script will give it a score of 0 for that method and continue.
- The output CSVs are ready for analysis. You can open them in a spreadsheet program (like Excel or Google Sheets) to compare models or projects.

## Things That Might Go Wrong

- **Missing `project` Column**: If `filtered_ground_truth.csv` doesn’t have a `project` column, the script will label all results as `Unknown` in `evaluation_results_per_project.csv`. Add the column as described above.
- **Missing Files**: If `filtered_ground_truth.csv` or any model folders (`UTLOAK-PLBART`, etc.) are missing, the script will fail. Ensure they’re in the right folder.
- **Java Tools Not Set Up**: If `ANTLR4`, `JUnit`, `JaCoCo`, or `PITest` aren’t installed or configured, the script won’t work. Verify their setup.
- **Memory Problems**: If your computer lacks enough memory (especially without a GPU), the script might crash. Try running on a more powerful machine.
- **Slow Performance**: Without a GPU, the script will be slower. Be patient or use a GPU-enabled computer.
- **Incorrect Paths**: If the paths to the models or dataset don’t match your setup, update them in the script (e.g., change `./UTLOAK-PLBART` to the correct folder).

