import wandb

# Initialize wandb API
api = wandb.Api()

# List all runs in the project
project_path = "timoshka3-tel-aviv-university/GNN"

print(f"Listing all runs in project: {project_path}")
print("="*80)

try:
    runs = api.runs(project_path)
    print(f"Found {len(runs)} total runs")
    
    for i, run in enumerate(runs):
        print(f"{i+1:2d}. ID: {run.id:30s} Name: {run.name:30s} State: {run.state}")
        if hasattr(run, 'tags') and run.tags:
            print(f"    Tags: {run.tags}")
        print()
        
        # Stop after first 20 for readability
        if i >= 19:
            print(f"... and {len(runs) - 20} more runs")
            break
            
except Exception as e:
    print(f"Error accessing project: {e}")
