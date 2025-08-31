import subprocess
import sys

def run_experiment(memory: str, cores: str):
    """
    Runs the training script with a specific Spark configuration.
    """
    print("="*50)
    print(f"Running experiment with Memory='{memory}', Cores='{cores}'")
    print("="*50)
    
    # We use subprocess to run the training script as a separate process.
    # This ensures that each run gets a fresh Spark session with the new configuration.
    # We also pass the memory and cores as command-line arguments.
    command = [
        sys.executable,  # This ensures we use the same python interpreter
        "src/train_spark.py",
        "--memory", memory,
        "--cores", cores
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"\nExperiment with {memory} memory and {cores} cores completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Experiment with {memory} memory and {cores} cores failed.")
        print(f"Error details: {e}")
    except FileNotFoundError:
        print("\nERROR: Could not find 'src/train_spark.py'.")
        print("Please ensure you are running this script from the project's root directory.")


if __name__ == "__main__":
    # --- Define the list of configurations to test ---
    configurations = [
        {"memory": "8g", "cores": "8"},  # Baseline
        {"memory": "8g", "cores": "4"},  # CPU-constrained
        {"memory": "4g", "cores": "8"},  # Memory-constrained
        {"memory": "4g", "cores": "4"}   # Balanced
    ]

    for config in configurations:
        run_experiment(memory=config["memory"], cores=config["cores"])

    print("="*50)
    print("All experiments have been completed.")
    print("Please check the MLflow UI to compare the results.")
    print("="*50)

