# Overview and Documentation of the Thesis Project: A Lightweight Model for Balancing Efficiency and Precision in PEFT-Optimized Java Unit Test Generation

This document provides a comprehensive overview of the thesis project titled *"A Lightweight Model for Balancing Efficiency and Precision in PEFT-Optimized Java Unit Test Generation"* by Sintayehu Zekarias Esubalew, submitted in partial fulfillment of the requirements for a Master of Science in Artificial Intelligence at the School of Information Technology and Engineering, College of Technology and Built Environment, Addis Ababa University, in June 2025. The project was supervised by Dr. Beakal Gizachew Assefa. The documentation covers the project’s objectives, methodology, implementation details, and setup instructions, ensuring clarity for both technical and non-technical audiences.

## Project Overview

The project focuses on developing a lightweight artificial intelligence (AI) model to generate high-quality unit tests for Java code while minimizing computational resources. Unit testing is a critical process in software development to ensure code reliability, but manually creating tests is time-consuming and error-prone. Large language models (LLMs) can automate test generation, but they often require significant memory and processing power, making them impractical for resource-constrained environments. This project addresses this challenge by using Parameter-Efficient Fine-Tuning (PEFT) techniques to create lightweight models that balance test quality (precision) and computational efficiency.

The project evaluates three PEFT-optimized models—`A3Test-LoRA`, `A3Test-QLoRA`, and `A3Test-Adapter`—built on the `PLBart` base model. These models generate unit tests for Java methods in the Defects4J v2.5.0 dataset, which includes six open-source projects: `Lang`, `Chart`, `Gson`, `Csv`, `Json`, and `Cli`. The tests are assessed for correctness, code coverage, and bug detection using tools like `ANTLR4`, `JUnit`, `JaCoCo`, and `PITest`. The project measures seven metrics: Correct Test Cases (CTC), Focal Method Coverage (FMC), Line Coverage (LC), Correct Assertions (CA), Mutation Score (MS), Average Time per Test (ATT), and Peak Memory Usage. Results are saved in two CSV files: `evaluation_results.csv` (overall averages) and `evaluation_results_per_project.csv` (per-project results).

### Objectives
- Develop lightweight models using PEFT techniques (LoRA, QLoRA, Adapters) to generate Java unit tests.
- Achieve high test quality (e.g., ~38% CTC for `A3Test-LoRA`) with reduced resource usage (e.g., ~15GB memory for `A3Test-QLoRA`).
- Compare model performance across Defects4J projects to identify the best PEFT strategy.
- Provide a scalable solution for automated testing in resource-constrained settings.

### Significance
The project contributes to AI-driven software testing by demonstrating that lightweight models can produce effective unit tests with lower computational costs, benefiting developers and researchers in both industry and academia. It addresses the growing need for efficient testing tools in large-scale Java projects.

## Who Prepared and Advised the Project

- **Author**: Sintayehu Zekarias Esubalew, a Master of Science in Artificial Intelligence student at Addis Ababa University, designed and implemented the project. This included developing the PEFT-optimized models, preprocessing the Defects4J dataset, and creating the test validation script to evaluate model performance.
- **Supervisor**: Dr. Beakal Gizachew Assefa, a faculty member at the School of Information Technology and Engineering, provided guidance on the research methodology, model design, and evaluation framework, ensuring the project met academic standards and addressed relevant challenges in AI and software engineering.

## Methodology

The project follows a structured pipeline:

1. **Dataset Acquisition**: Download and preprocess the Defects4J v2.5.0 dataset, which contains 835 bugs across 17 projects, focusing on six (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`) with ~5,278 focal methods.
2. **Preprocessing**: Create `filtered_ground_truth.csv` with columns `focal_method` (Java method code), `ground_truth` (correct test code), and `project` (e.g., `Lang`).
3. **Model Development**: Fine-tune the `PLBart` model using PEFT techniques to create `A3Test-LoRA`, `A3Test-QLoRA`, and `A3Test-Adapter`, stored in `./UTLOAK-PLBART`, `./UTQOAK-PLBART`, and `./UTAOAK-PLBART`.
4. **Test Generation and Evaluation**: Generate unit tests for focal methods using each model, validate syntax with `ANTLR4`, check correctness with `JUnit`, measure coverage with `JaCoCo`, and assess bug detection with `PITest`. Track time (ATT) and memory usage.
5. **Result Analysis**: Save results in `evaluation_results.csv` (overall averages) and `evaluation_results_per_project.csv` (per-project results) for comparison.

## Implementation Details

The project relies on a Python script (`phase4_test_validation_per_project.py`) to evaluate the models. Below is the script, which generates and evaluates tests, producing both aggregated and per-project results:



## How to Set Up and Run the Project

Follow these steps to set up and run the project, ensuring you can replicate the results.

### 1. Install Dependencies
- **Python 3.8+**: Install Python and required libraries:
  ```bash
  pip install pandas torch transformers antlr4-python3-runtime psutil datasets
  ```
- **Java 1.8**: Required for Defects4J and Java tools:
  ```bash
  sudo apt-get install openjdk-8-jdk  # On Ubuntu/Debian
  ```
- **Perl and Git**: For Defects4J setup:
  ```bash
  sudo apt-get install perl git
  cpan App::cpanminus
  ```
- **Java Tools**: Install and configure:
  - [ANTLR4](https://www.antlr.org/) for syntax validation.
  - [JUnit](https://junit.org/) for test execution.
  - [JaCoCo](https://www.jacoco.org/) for code coverage.
  - [PITest](http://pitest.org/) for mutation testing.
- **Hardware**: A GPU (e.g., NVIDIA Tesla V100-SXM2 32GB) is recommended for faster processing, but a CPU works (slower).

### 2. Download Defects4J v2.5.0
The script uses the Defects4J dataset for evaluation. Follow these steps (detailed in artifact ID `fdf2a827-eaa4-4941-b5fd-1316f16fe51d`):
1. Clone the Defects4J repository:
   ```bash
   git clone https://github.com/rjust/defects4j.git
   cd defects4j
   ```
2. Install Perl dependencies:
   ```bash
   cpanm --installdeps .
   ```
3. Initialize Defects4J to download projects:
   ```bash
   ./init.sh
   ```
4. Add Defects4J to your `PATH`:
   ```bash
   export PATH=$PATH:$(pwd)/framework/bin
   echo 'export PATH=$PATH:/path/to/defects4j/framework/bin' >> ~/.bashrc
   source ~/.bashrc
   ```
5. Verify installation:
   ```bash
   defects4j info -p Lang
   ```

### 3. Prepare `filtered_ground_truth.csv`
The script requires `filtered_ground_truth.csv` with columns `focal_method`, `ground_truth`, and `project`. This file is generated by preprocessing Defects4J:
- **Steps**:
  - Check out buggy and fixed versions of each project (e.g., `Lang-1b`, `Lang-1f`):
    ```bash
    defects4j checkout -p Lang -v 1b -w lang_bug1
    defects4j checkout -p Lang -v 1f -w lang_fix1
    ```
  - Extract focal methods (buggy methods) and ground-truth tests using `defects4j export -p classes.modified`.
  - Assign the `project` column (e.g., `Lang`, `Chart`) based on the source project.
  - Save as `filtered_ground_truth.csv`.
- **Example Python Snippet** (customize for actual method/test extraction):
  ```python
  import pandas as pd
  import os
  data = []
  projects = ["Lang", "Chart", "Gson", "Csv", "Json", "Cli"]
  for project in projects:
      # Example bug IDs (adjust per project)
      bug_ids = range(1, 65) if project == "Lang" else range(1, 27) if project == "Chart" else ...
      for bug_id in bug_ids:
          os.system(f"defects4j checkout -p {project} -v {bug_id}b -w temp_bug")
          # Extract focal method and ground-truth test (requires parsing)
          focal_method = "method_code"  # Placeholder
          ground_truth = "test_code"  # Placeholder
          data.append({"project": project, "focal_method": focal_method, "ground_truth": ground_truth})
          os.system("rm -rf temp_bug")
  df = pd.DataFrame(data)
  df.to_csv("filtered_ground_truth.csv", index=False)
  ```
- **Note**: This step is complex and may require a separate preprocessing script or tools like `javalang` to parse Java code. Consult Defects4J documentation or related papers for guidance.

### 4. Obtain Trained Models
The script expects three trained models in folders:
- `./UTLOAK-PLBART` (`A3Test-LoRA`)
- `./UTQOAK-PLBART` (`A3Test-QLoRA`)
- `./UTAOAK-PLBART` (`A3Test-Adapter`)
These models are assumed to be fine-tuned versions of `PLBart` using PEFT techniques, created in an earlier phase (not provided). Place them in the same directory as the script.

### 5. Run the Script
1. Save the script as `phase4_test_validation_per_project.py`.
2. Place it in a directory with `filtered_ground_truth.csv` and the model folders.
3. Run the script:
   ```bash
   python phase4_test_validation_per_project.py
   ```
4. The script will:
   - Load models and dataset.
   - Generate and evaluate tests using `ANTLR4`, `JUnit`, `JaCoCo`, and `PITest`.
   - Save results in `test_validation_results/`:
     - `evaluation_results.csv`: Overall averages for CTC, FMC, LC, CA, MS, ATT, Memory.
     - `evaluation_results_per_project.csv`: Per-project results for each model.

### 6. Analyze Results
- **Output Files**:
  - `evaluation_results.csv`: Aggregated metrics (e.g., CTC ~38% for `A3Test-LoRA`, Memory ~15GB for `A3Test-QLoRA`).
  - `evaluation_results_per_project.csv`: Metrics for each project (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`), formatted as `Project,Metric,A3Test-LoRA,A3Test-QLoRA,A3Test-Adapter`.
- **Analysis**: Open CSVs in Excel or Google Sheets, or use Python/R for visualization (e.g., bar charts of CTC by project).
- **Expected Outcomes**: `A3Test-LoRA` should have the highest correctness (~38% CTC), `A3Test-QLoRA` the lowest memory usage (~15GB), and `A3Test-Adapter` lower performance due to its design.


## Project Preparation and Contributions

- **Prepared by**: Sintayehu Zekarias Esubalew developed the entire pipeline, including:
  - Preprocessing the Defects4J dataset to create `filtered_ground_truth.csv`.
  - Fine-tuning `PLBart` with PEFT techniques to create the three models.
  - Writing the test validation script to generate and evaluate tests.
  - Analyzing results to compare model performance.
- **Advised by**: Dr. Beakal Gizachew Assefa provided expert guidance on:
  - Selecting PEFT techniques (LoRA, QLoRA, Adapters) for lightweight model design.
  - Structuring the evaluation framework using Defects4J and standard testing tools.
  - Ensuring the research aligns with AI and software engineering principles.
  - Reviewing the methodology and results for academic rigor.

## Conclusion

This project, developed by Sintayehu Zekarias Esubalew under the supervision of Dr. Beakal Gizachew Assefa, advances automated software testing by introducing lightweight, PEFT-optimized models for Java unit test generation. By leveraging the Defects4J dataset and evaluating three models across six projects, it demonstrates a balance of precision and efficiency, offering a scalable solution for resource-constrained environments. The provided script and setup instructions enable replication, and the results provide valuable insights for improving software quality assurance.
