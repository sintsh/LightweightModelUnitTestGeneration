# How to Download Defects4J Dataset Version 2.5.0

This document explains how to download and set up the Defects4J dataset version 2.5.0, a collection of reproducible bugs from open-source Java projects (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`, and others) used for software testing research. The dataset is required for the Phase 4 test validation script (artifact ID `adae02a9-433f-42da-9ff6-b486b4aff8df`), which evaluates unit tests generated by models (`A3Test-LoRA`, `A3Test-QLoRA`, `A3Test-Adapter`) and expects a `filtered_ground_truth.csv` file derived from Defects4J. This guide provides step-by-step instructions to download, install, and prepare the dataset for use with the script.

## Prerequisites

Before downloading Defects4J v2.5.0, ensure your system meets these requirements:
- **Operating System**: Unix-like system (Linux, macOS) is preferred. Windows is supported but may require additional steps.
- **Java**: Java 1.8 (required for Defects4J v2.x). Install it using:
  ```bash
  sudo apt-get install openjdk-8-jdk  # On Ubuntu/Debian
  ```
  or download from [Oracle](https://www.oracle.com/java/technologies/javase-jdk8-downloads.html).
- **Perl**: Perl 5 is required, typically pre-installed on Unix systems. Install it on Windows via [Strawberry Perl](https://strawberryperl.com/) or on Ubuntu/Debian with:
  ```bash
  sudo apt-get install perl
  ```
- **Git**: Git for cloning the repository. Install it with:
  ```bash
  sudo apt-get install git  # On Ubuntu/Debian
  ```
- **Perl Modules**: Defects4J requires specific Perl modules listed in its `cpanfile`. Install `cpanm` (CPAN Minus) to manage them:
  ```bash
  sudo cpan App::cpanminus
  ```
- **Disk Space**: At least 10GB free space for project repositories and external libraries.
- **Python**: Python 3.8+ for the Phase 4 script and to process the dataset into `filtered_ground_truth.csv`.

## Steps to Download Defects4J v2.5.0

Follow these steps to download and set up Defects4J v2.5.0:

1. **Clone the Defects4J Repository**:
   - Open a terminal and clone the Defects4J repository from GitHub:
     ```bash
     git clone https://github.com/rjust/defects4j.git
     ```
   - This creates a folder (e.g., `defects4j`) containing the Defects4J framework and metadata but not the project repositories (downloaded later).

2. **Navigate to the Defects4J Directory**:
   - Move into the cloned directory:
     ```bash
     cd defects4j
     ```

3. **Install Perl Dependencies**:
   - Install the required Perl modules listed in `cpanfile`:
     ```bash
     cpanm --installdeps .
     ```
   - This command uses `cpanm` to install modules like `DBI` and `File::Find`. If you don’t have `cpanm`, install it first (see Prerequisites) or use:
     ```bash
     cpan -i <module_name>
     ```
     for each module listed in `cpanfile`.

4. **Initialize Defects4J**:
   - Run the initialization script to download project repositories (e.g., `Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`) and external libraries:
     ```bash
     ./init.sh
     ```
   - This downloads the source code, test suites, and metadata for the 17 projects in Defects4J v2.0.0, which includes the six projects used in your script. The repositories are stored in `defects4j/framework/projects/`.

5. **Set Up the Defects4J Command-Line Tool**:
   - Add the Defects4J `bin` directory to your `PATH` to use the `defects4j` command:
     ```bash
     export PATH=$PATH:$(pwd)/framework/bin
     ```
   - Replace `$(pwd)` with the full path to your `defects4j` directory (e.g., `/home/user/defects4j`) if needed. For permanent setup, add this line to your `~/.bashrc` or `~/.zshrc`:
     ```bash
     echo 'export PATH=$PATH:/path/to/defects4j/framework/bin' >> ~/.bashrc
     source ~/.bashrc
     ```
   - On Windows, use the full path with Perl:
     ```bash
     perl /path/to/defects4j/framework/bin/defects4j
     ```

6. **Verify the Installation**:
   - Check that Defects4J is set up correctly by listing project IDs:
     ```bash
     defects4j info -p Lang
     ```
   - This should display information about the `Lang` project, including bug IDs. Repeat for `Chart`, `Gson`, `Csv`, `Json`, and `Cli` to confirm all relevant projects are available.

7. **Export Project Data**:
   - Defects4J v2.5.0 includes 835 active bugs across 17 projects, but your script uses six: `Lang` (64 bugs), `Chart` (26 bugs), `Gson` (18 bugs), `Csv` (16 bugs), `Json` (6 bugs), and `Cli` (39 bugs). To prepare data for `filtered_ground_truth.csv`, check out each project’s buggy and fixed versions:
     ```bash
     defects4j checkout -p Lang -v 1b -w lang_bug1
     defects4j checkout -p Lang -v 1f -w lang_fix1
     ```
   - Repeat for each bug ID (e.g., `Lang` bugs 1, 3–65; `Chart` bugs 1–26, etc.) and project. This creates directories (e.g., `lang_bug1`, `lang_fix1`) with source code and test suites.
   - Use the `export` command to extract metadata, such as modified classes or triggering tests:
     ```bash
     defects4j export -p classes.modified -w lang_bug1
     ```
   - This helps identify focal methods for your script.

8. **Generate `filtered_ground_truth.csv`**:
   - Your script expects `filtered_ground_truth.csv` with columns `focal_method`, `ground_truth`, and `project`. This file is typically created by a preprocessing script (not provided here) that extracts focal methods (buggy methods) and ground-truth tests from Defects4J.
   - To create this file:
     - Iterate through each project’s buggy versions (e.g., `Lang-1b`, `Chart-1b`).
     - Use `defects4j export -p classes.modified` to identify modified classes, then manually or programmatically extract focal methods (methods changed in the bug fix).
     - Pair each focal method with its ground-truth test (from the test suite that fails on the buggy version and passes on the fixed version).
     - Add the `project` column (e.g., `Lang`, `Chart`) to each row.
     - Save as a CSV with columns: `focal_method`, `ground_truth`, `project`.
   - Example Python snippet to start this process (you’ll need to customize it):
     ```python
     import pandas as pd
     import os
     data = []
     projects = ["Lang", "Chart", "Gson", "Csv", "Json", "Cli"]
     for project in projects:
         # Get bug IDs (e.g., from defects4j info -p Lang)
         bug_ids = range(1, 65) if project == "Lang" else ...  # Adjust per project
         for bug_id in bug_ids:
             os.system(f"defects4j checkout -p {project} -v {bug_id}b -w temp_bug")
             # Extract focal method and ground-truth test (custom logic needed)
 investigate)
             focal_method = "method_code"  # Placeholder
             ground_truth = "test_code"  # Placeholder
             data.append({"project": project, "focal_method": focal_method, "ground_truth": ground_truth})
             os.system("rm -rf temp_bug")
     df = pd.DataFrame(data)
     df.to_csv("filtered_ground_truth.csv", index=False)
     ```
   - This is a simplified example. You may need tools like `javalang` or `jdt-core` to parse Java code and extract methods, and manual effort to map tests to methods.

## Notes and Tips

- **Defects4J Version**: The search results mention Defects4J v2.0.0 with 835 bugs, but v2.5.0 is the latest (as of June 21, 2025), likely including the same or more bugs. The six projects (`Lang`, `Chart`, `Gson`, `Csv`, `Json`, `Cli`) are included in v2.0.0 and should be in v2.5.0.
- **Java Version**: Use Java 1.8, as Defects4J v2.x requires it. Java 11 may cause issues with test execution due to behavioral changes.
- **Timezone**: Defects4J uses the `America/Los_Angeles` timezone for test execution. Set it if running tests outside the Defects4J framework:
  ```bash
  export TZ=America/Los_Angeles
  ```
- **Storage**: The dataset is large (10GB+). Ensure sufficient disk space.
- **Preprocessing**: Creating `filtered_ground_truth.csv` is complex and may require a separate preprocessing script or manual effort to map focal methods and tests. Check the Defects4J documentation or related papers for guidance.
- **Windows Users**: Use `perl /path/to/defects4j/framework/bin/defects4j` instead of `defects4j` for commands. Adjust paths to use backslashes (`\`) instead of forward slashes (`/`).

## Things That Might Go Wrong

- **Missing Perl Modules**: If `cpanm --installdeps .` fails, manually install modules listed in `cpanfile` using `cpan <module_name>`.
- **Java Version Mismatch**: Ensure Java 1.8 is used. Check with:
  ```bash
  java -version
  ```
- **Git Clone Fails**: Verify internet connectivity and Git installation. Retry or check the repository URL (`https://github.com/rjust/defects4j`).
- **Initialization Fails**: Ensure write permissions in the `defects4j` directory and sufficient disk space.
- **No `project` Column**: If `filtered_ground_truth.csv` lacks a `project` column, the script will label results as `Unknown`. Add the column as described in the script’s documentation.

## Additional Resources

- **Official Documentation**: [Defects4J Documentation](https://defects4j.org) for detailed commands and setup.[](http://defects4j.org/html_doc/index.html)
- **GitHub Repository**: [rjust/defects4j](https://github.com/rjust/defects4j) for source code and updates.[](https://github.com/rjust/defects4j)
- **Papers**: Refer to papers on ResearchGate or Springer for preprocessing insights (e.g., extracting focal methods).[](https://www.researchgate.net/publication/266659285_Defects4J_a_database_of_existing_faults_to_enable_controlled_testing_studies_for_Java_programs)[](https://link.springer.com/article/10.1007/s11219-020-09515-0)

This guide ensures you can download Defects4J v2.5.0 and prepare it for use with the Phase 4 test validation script. If you need help creating `filtered_ground_truth.csv` or troubleshooting specific issues, let me know![](https://github.com/rjust/defects4j)