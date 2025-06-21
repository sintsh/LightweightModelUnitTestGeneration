import os
import pandas as pd
import javalang
from pathlib import Path
import re
from typing import List, Tuple, Dict

# Defining paths
DEFECTS4J_ROOT = "./defects4j/projects"
OUTPUT_PATH = "./filtered_ground_truth.csv"
PROJECTS = ["Lang", "Chart", "Gson", "Csv", "Json", "Cli"]

def read_file(file_path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_methods_and_javadoc(file_content: str) -> List[Tuple[str, str]]:
    """Extracts methods and their Javadoc comments from a Java file."""
    try:
        tree = javalang.parse.parse(file_content)
        methods = []
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            # Extracting method signature and body
            method_code = f"{node.modifiers} {node.return_type.name if node.return_type else 'void'} {node.name}("
            method_code += ", ".join(f"{p.type.name} {p.name}" for p in node.parameters) + ") {"
            method_code += " /* Method body */ }"  # Simplified body for focal method
            # Extracting Javadoc
            javadoc = ""
            if node.documentation:
                javadoc = node.documentation.strip()
            methods.append((method_code, javadoc))
        return methods
    except javalang.parser.JavaSyntaxError:
        return []

def extract_junit_tests(file_content: str) -> List[str]:
    """Extracts JUnit test methods from a test file."""
    try:
        tree = javalang.parse.parse(file_content)
        tests = []
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            # Checking for @Test annotation
            if any(ann.name == "Test" for ann in node.annotations):
                test_code = f"public void {node.name}() "
                test_code += "{ /* Test logic */ }"  # Simplified test code
                tests.append(test_code)
        return tests
    except javalang.parser.JavaSyntaxError:
        return []

def map_tests_to_methods(test_files: List[str], methods: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Maps JUnit tests to focal methods based on naming conventions and Javadoc."""
    mappings = []
    for method_code, javadoc in methods:
        method_name = re.search(r"\w+\s+(\w+)\(", method_code).group(1)
        ground_truth = ""
        # Finding matching test by method name or Javadoc reference
        for test_file in test_files:
            test_content = read_file(test_file)
            test_methods = extract_junit_tests(test_content)
            for test in test_methods:
                if method_name.lower() in test.lower() or any(keyword in test for keyword in javadoc.split() if keyword):
                    ground_truth = test
                    break
            if ground_truth:
                break
        # Fallback to Javadoc as ground truth if no test found
        if not ground_truth and javadoc:
            ground_truth = f"// Javadoc-based test: {javadoc"
        if ground_truth:
            mappings.append({"focal_method": method_code, "ground_truth": ground_truth})
        return mappings

def process_project(project_name: str) -> List[Dict[str, str]]:
    """Processes a Defects4J project to extract method-test mappings."""
    project_dir = Path(DEFECTS4J_ROOT) / project_name
    source_dir = project_dir / "src"
    test_dir = project_dir / "test"
    
    mappings = []
    source_files = list(source_dir.rglob("*.java"))
    test_files = list(test_dir.rglob("*.java"))
    
    for source_file in source_files:
        file_content = read_file(source_file)
        methods = extract_methods_and_javadoc(file_content)
        if methods:
            mappings.extend(map_tests_to_methods(test_files, methods))
    
    return mappings

def main():
    """Main function to process all projects and save to CSV."""
    all_mappings = []
    for project in PROJECTS:
        print(f"Processing project: {project}")
        mappings = process_project(project)
        all_mappings.extend(mappings)
    
    # Saving to CSV
    df = pd.DataFrame(all_mappings, columns=["focal_method", "ground_truth"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} mappings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()