import os
import json
import wandb

wandb_project = "PerfectTransitions"  # Change this to your actual project name

wandb_dirs = [d for d in os.listdir() if d.startswith("run-")]

for run_dir in wandb_dirs:
    print(f"Restoring {run_dir}...")

    # Initialize a new run with the same name
    wandb.init(project=wandb_project, name=run_dir, resume="allow")

    history_file = os.path.join(run_dir, "wandb-history.jsonl")
    summary_file = os.path.join(run_dir, "wandb-summary.json")

    # Restore history (if available)
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    wandb.log(data)
                except json.JSONDecodeError:
                    print(f"Skipping corrupted line in {history_file}")

    # Restore summary metrics
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
            wandb.run.summary.update(summary_data)

    # Finish the run
    wandb.finish()

print("All runs have been restored!")
