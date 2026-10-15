import argparse
import yaml
import wandb
from src.fdlbiomass.tomograph_reconstruction.train_model import full_loop_train_model  # Assuming you've moved the train_model function to a separate file
from dotenv import load_dotenv 
import os 


def load_sweep_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_api_key)

    # Load the sweep configuration
    sweep_config = load_sweep_config(args.config)

    # Set the project name
    project_name = args.project_name if args.project_name else sweep_config.get('project_name', 'default_project')
    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_config,entity='fdl-biomass', project=project_name)

    # Start the sweep
    wandb.agent(sweep_id, full_loop_train_model, count=args.num_runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a wandb sweep with a specified configuration.")
    parser.add_argument('--config', type=str, required=True, help='Path to the sweep configuration YAML file')
    parser.add_argument('--project_name', type=str, help='WandB project name (overrides the one in config if provided)')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to execute in this sweep')
    args = parser.parse_args()

    main(args)


   
# python train_tomo_recon.py  --project_name tomo_recon_height --num_runs 2 --config /home/grace.bcolverd/2024-ESL-BiomassSAR/src/sweep_configs/tomo_recon.yaml