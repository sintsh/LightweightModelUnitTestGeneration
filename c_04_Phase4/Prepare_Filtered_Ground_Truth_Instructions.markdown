# How to Use the Defects4J Preprocessing Script

This document explains a Python script that helps create a file called `filtered_ground_truth.csv` from a collection of Java projects in the Defects4J dataset (version 2.5.0). The script looks at six projects—`Lang`, `Chart`, `Gson`, `Csv`, `Json`, and `Cli`—and pulls out important pieces of code (called "focal methods") along with their matching test code or documentation. The output file is a table with two columns: one for the Java method and another for its test (either a real test or a description from the code's comments). This file is used later to check if new tests created by a computer are correct.

## What the Script Does

The script does the following:

1. **Finds Java Code**: It looks through the Java files in each project to find methods (chunks of code that do specific tasks).
2. **Grabs Comments**: It also grabs any special comments (called Javadoc) that explain what each method does.
3. **Finds Tests**: It searches for test code (written using a tool called JUnit) that checks if the methods work correctly.
4. **Matches Methods to Tests**: It pairs each method with its test by looking for similar names or words from the Javadoc comments. If no test is found, it uses the Javadoc comment as a backup.
5. **Saves Results**: It saves all the pairs in a file called `filtered_ground_truth.csv`, which lists each method and its test.

The script works with about 5,278 methods across the six projects, but the exact number of pairs depends on how many tests or comments it can find.

## How the Script Works

Here’s a simple breakdown of the script’s key parts:

- **Reading Files**: The `read_file` function opens and reads Java files, handling any errors if a file can’t be read.
- **Finding Methods**: The `extract_methods_and_javadoc` function uses a tool called `javalang` to find methods in Java code and their Javadoc comments (if any).
- **Finding Tests**: The `extract_junit_tests` function looks for test methods marked with `@Test` in test files.
- **Pairing Methods and Tests**: The `map_tests_to_methods` function matches methods to tests by checking if the method’s name appears in the test or if words from the Javadoc comment show up in the test. If no test is found, it uses the Javadoc comment.
- **Processing Projects**: The `process_project` function goes through each project, collecting all method-test pairs.
- **Saving the Output**: The `main` function runs everything, combines the results, and saves them to `filtered_ground_truth.csv`.

## How to Run the Script

To use this script, follow these steps:

1. **Set Up Your Computer**:

   - Make sure you have **Python 3** installed (version 3.8 or higher works best).
   - Install the required Python libraries by running this command in your terminal or command prompt:
     ```bash
     pip install pandas javalang
     ```
     - `pandas` helps create the CSV file.
     - `javalang` helps read and understand Java code.

2. **Get the Defects4J Dataset**:

   - Download and set up Defects4J v2.5.0. You can find instructions at [Defects4J's website](https://github.com/rjust/defects4j).
   - Make sure the projects (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`) are in a folder called `defects4j/projects`. Each project should have:
     - A `src` folder with the Java code.
     - A `test` folder with the test code.
   - For example, the folder structure should look like:
     ```
     defects4j/projects/Lang/src/
     defects4j/projects/Lang/test/
     defects4j/projects/Chart/src/
     ...
     ```

3. **Save the Script**:

   - Copy the Python code above into a file named `prepare_filtered_ground_truth.py`.
   - Save it in the same folder as your `defects4j` folder (or adjust the `DEFECTS4J_ROOT` path in the script if needed).

4. **Run the Script**:

   - Open a terminal or command prompt and go to the folder where you saved the script.
   - Run the script with this command:
     ```bash
     python prepare_filtered_ground_truth.py
     ```
   - The script will:
     - Read through each project’s Java files.
     - Match methods to tests or Javadoc comments.
     - Create a file called `filtered_ground_truth.csv` in the same folder.

5. **Check the Output**:
   - After running, you’ll see messages like `Processing project: Lang` and finally `Saved X mappings to filtered_ground_truth.csv`.
   - Open `filtered_ground_truth.csv` to see the results. It will have two columns:
     - `focal_method`: The Java method (e.g., `public int add(int a, int b) { /* Method body */ }`).
     - `ground_truth`: The test or Javadoc comment (e.g., `public void testAdd() { /* Test logic */ }`).

## What to Expect

- The script processes thousands of methods, but not every method will have a test or Javadoc comment. The final CSV might have fewer entries than the total number of methods (around 5,278).
- If there are errors (e.g., a Java file can’t be read), the script will print a message but keep going.
- The output CSV is ready to use with other tools, like the test validation script in Phase 4 of a project.

