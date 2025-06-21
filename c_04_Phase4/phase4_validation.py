import os
import time
import psutil
import pandas as pd
import torch
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ConsoleErrorListener
from JavaLexer import JavaLexer
from JavaParser import JavaParser
import junit
import jacoco
import pitest
from datasets import load_dataset
from collections import defaultdict

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Defining paths
model_paths = {
    "A3Test-LoRA": "./UTLOAK-PLBART",
    "A3Test-QLoRA": "./UTQOAK-PLBART",
    "A3Test-Adapter": "./UTAOAK-PLBART"
}
dataset_path = "./filtered_ground_truth.csv"
output_dir = "./test_validation_results"

# Loading tokenizer
tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")

# Loading Defects4J dataset
dataset = load_dataset("csv", data_files={"test": dataset_path})["test"]

def validate_syntax(test_code):
    """Validates syntactic correctness of generated test code using ANTLR4."""
    try:
        input_stream = InputStream(test_code)
        lexer = JavaLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(ConsoleErrorListener())
        tree = parser.compilationUnit()
        return True
    except Exception:
        return False

def generate_test(model, tokenizer, focal_method, max_length=512):
    """Generates a unit test for a given focal method using the specified model."""
    inputs = tokenizer(focal_method, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_tests(model_name, model_path):
    """Evaluates generated tests for a given model across Defects4J projects."""
    model = PLBartForConditionalGeneration.from_pretrained(model_path).to(device)
    # Initialize results for aggregation (overall)
    agg_results = {"CTC": [], "FMC": [], "LC": [], "CA": [], "MS": [], "ATT": [], "Memory": []}
    # Initialize per-project results
    proj_results = defaultdict(lambda: {"CTC": [], "FMC": [], "LC": [], "CA": [], "MS": [], "ATT": [], "Memory": []})
    
    for example in dataset:
        focal_method = example["focal_method"]
        ground_truth = example["ground_truth"]
        project = example.get("project", "Unknown")  # Assumes 'project' column exists
        
        # Measuring time and memory
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024**3  # Convert to GB
        
        # Generating test
        generated_test = generate_test(model, tokenizer, focal_method)
        
        # Measuring end time and memory
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024**3
        att = end_time - start_time
        peak_memory = max(start_memory, end_memory)
        
        # Validating syntax
        is_syntactically_correct = validate_syntax(generated_test)
        if not is_syntactically_correct:
            ctc, ca, fmc, lc, ms = 0, 0, 0, 0, 0
        else:
            # Running JUnit for semantic correctness
            ctc, ca = junit.run_test(generated_test, ground_truth)
            # Measuring coverage
            fmc, lc = jacoco.compute_coverage(generated_test, focal_method)
            # Running mutation testing
            ms = pitest.compute_mutation_score(generated_test, focal_method)
        
        # Append to aggregate results
        agg_results["CTC"].append(ctc)
        agg_results["FMC"].append(fmc)
        agg_results["LC"].append(lc)
        agg_results["CA"].append(ca)
        agg_results["MS"].append(ms)
        agg_results["ATT"].append(att)
        agg_results["Memory"].append(peak_memory)
        
        # Append to per-project results
        proj_results[project]["CTC"].append(ctc)
        proj_results[project]["FMC"].append(fmc)
        proj_results[project]["LC"].append(lc)
        proj_results[project]["CA"].append(ca)
        proj_results[project]["MS"].append(ms)
        proj_results[project]["ATT"].append(att)
        proj_results[project]["Memory"].append(peak_memory)
    
    # Aggregating overall results
    avg_results = {k: sum(v) / len(v) for k, v in agg_results.items()}
    
    # Aggregating per-project results
    per_project_avg = {
        project: {k: sum(v) / len(v) for k, v in metrics.items()}
        for project, metrics in proj_results.items()
    }
    
    return avg_results, per_project_avg

# Running evaluation for all models
all_agg_results = {}
all_per_project_results = {}
for model_name, model_path in model_paths.items():
    print(f"Evaluating {model_name}...")
    agg_results, per_project_results = evaluate_tests(model_name, model_path)
    all_agg_results[model_name] = agg_results
    all_per_project_results[model_name] = per_project_results

# Saving aggregated results
os.makedirs(output_dir, exist_ok=True)
agg_results_df = pd.DataFrame(all_agg_results)
agg_results_df.to_csv(f"{output_dir}/evaluation_results.csv")

# Saving per-project results
per_project_data = []
projects = ["Lang", "Chart", "Gson", "Csv", "Json", "Cli"]
metrics = ["CTC", "FMC", "LC", "CA", "MS", "ATT", "Memory"]
for project in projects:
    for metric in metrics:
        row = {
            "Project": project,
            "Metric": metric,
            "A3Test-LoRA": all_per_project_results["A3Test-LoRA"].get(project, {}).get(metric, 0),
            "A3Test-QLoRA": all_per_project_results["A3Test-QLoRA"].get(project, {}).get(metric, 0),
            "A3Test-Adapter": all_per_project_results["A3Test-Adapter"].get(project, {}).get(metric, 0)
        }
        per_project_data.append(row)
per_project_df = pd.DataFrame(per_project_data)
per_project_df.to_csv(f"{output_dir}/evaluation_results_per_project.csv", index=False)

# Printing results
print("\nAggregated Evaluation Results:")
print(agg_results_df)
print("\nPer-Project Evaluation Results:")
print(per_project_df)