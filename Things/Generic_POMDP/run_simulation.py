"""
Simulation runner for Generic POMDP with comprehensive testing and visualization.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from generic_pomdp import GenericPOMDP

def setup_simulation_dirs(base_dir="Output"):
    """Create simulation directory structure."""
    dirs = {
        "logs": "logs",
        "plots": "plots",
        "simulations": "simulations"
    }
    
    base_path = Path(base_dir)
    created_dirs = {}
    
    for name, subdir in dirs.items():
        path = base_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        created_dirs[name] = path
        
    return created_dirs

def setup_logging(log_dir):
    """Configure logging for simulation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    # Load configuration
    with open("configuration.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    dirs = setup_simulation_dirs()
    logger = setup_logging(dirs["logs"])
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dirs["simulations"] / f"run_{timestamp}"
    run_dir.mkdir()
    
    # Initialize model
    model = GenericPOMDP(
        num_observations=config['dimensions']['observations'],
        num_states=config['dimensions']['states'],
        num_actions=config['dimensions']['actions']
    )
    
    # Run simulation
    logger.info("Starting simulation")
    history = {
        'observations': [],
        'actions': [],
        'free_energies': [],
        'beliefs': []
    }
    
    for step in range(config['dimensions']['total_timesteps']):
        logger.info(f"Step {step + 1}/{config['dimensions']['total_timesteps']}")
        obs, fe = model.step()
        
        # Store history
        history['observations'].append(int(obs))
        history['actions'].append(int(model.state.history['actions'][-1]))
        history['free_energies'].append(float(fe))
        history['beliefs'].append(model.state.beliefs.tolist())
    
    # Save results
    results_file = run_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate plots
    model.plot_belief_evolution(save_path=run_dir / "belief_evolution.png")
    model.plot_free_energy(save_path=run_dir / "free_energy.png")
    
    logger.info("Simulation completed successfully")

if __name__ == "__main__":
    main() 