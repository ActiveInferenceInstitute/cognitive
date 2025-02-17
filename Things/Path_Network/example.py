"""
Example script demonstrating the Path Network simulation.
"""

import os
import sys
import logging
import numpy as np
import matplotlib
from datetime import datetime
from pathlib import Path
import yaml
from path_network.core.network import NetworkConfig
from path_network.core.dynamics import DynamicsConfig, WaveComponent
from path_network.simulation.runner import SimulationRunner, SimulationConfig
from path_network.utils.visualization import NetworkVisualizer
from path_network.utils.advanced_visualization import AdvancedVisualizer

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('path_network')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(output_dir / 'simulation.log')
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def setup_matplotlib_backend(logger: logging.Logger) -> bool:
    """
    Set up appropriate matplotlib backend.
    
    Returns:
        bool: True if interactive backend is available
    """
    interactive = True
    
    # First try Qt5Agg
    try:
        matplotlib.use('Qt5Agg')
        logger.info("Using Qt5Agg backend")
        return interactive
    except Exception:
        logger.warning("Could not use Qt5Agg backend")
    
    # Then try TkAgg
    try:
        matplotlib.use('TkAgg')
        logger.info("Using TkAgg backend")
        return interactive
    except Exception:
        logger.warning("Could not use TkAgg backend")
    
    # Finally fallback to Agg
    try:
        matplotlib.use('Agg')
        logger.info("Using non-interactive Agg backend")
        interactive = False
        return interactive
    except Exception as e:
        logger.error(f"Could not set up any matplotlib backend: {e}")
        raise

def create_output_directory() -> Path:
    """Create and return output directory for simulation results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_simulation(
    config: dict,
    logger: logging.Logger,
    output_dir: Path,
    interactive: bool
):
    """Run the main simulation with comprehensive logging and visualization."""
    logger.info("Initializing simulation configurations...")
    
    # Create network configuration
    network_config = NetworkConfig(
        num_nodes=config['network']['num_nodes'],
        initial_connectivity=config['network']['initial_connectivity'],
        min_weight=config['network']['min_weight'],
        max_weight=config['network']['max_weight'],
        dynamic_topology=config['network']['dynamic_topology'],
        topology_update_interval=config['network']['topology_update_interval']
    )
    
    # Create dynamics configuration
    dynamics_config = DynamicsConfig(
        base_components=[
            WaveComponent(
                amplitude=comp['amplitude'],
                frequency=comp['frequency'],
                phase=comp['phase']
            )
            for comp in config['dynamics']['wave_components']
        ],
        noise_std=config['dynamics']['noise_std'],
        time_scale=config['dynamics']['time_scale']
    )
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        network_config=network_config,
        dynamics_config=dynamics_config,
        num_steps=config['simulation']['total_steps'],
        save_interval=config['simulation']['save_interval'],
        visualization_interval=(
            config['simulation']['visualization_interval']
            if interactive
            else config['simulation']['total_steps'] // 10
        )
    )
    
    logger.info("Creating visualizers...")
    basic_visualizer = NetworkVisualizer()
    advanced_visualizer = AdvancedVisualizer()
    
    logger.info("Initializing simulation runner...")
    runner = SimulationRunner(sim_config, basic_visualizer)
    
    # Initial simulation period
    logger.info("Running initial simulation period...")
    initial_states = runner.run(config['simulation']['initial_period'])
    
    # Save initial period results
    basic_visualizer.save(output_dir / 'initial_state.png')
    logger.info("Saved initial state visualization")
    
    # Add perturbation
    logger.info("Adding perturbation to the system...")
    runner.add_perturbation(
        magnitude=config['perturbation']['magnitude'],
        duration=config['perturbation']['duration'],
        decay=config['perturbation']['decay']
    )
    
    # Continue simulation
    logger.info("Running post-perturbation simulation...")
    final_states = runner.run(
        config['simulation']['total_steps'] -
        config['simulation']['initial_period']
    )
    
    # Save final basic visualization
    basic_visualizer.save(output_dir / 'final_state.png')
    logger.info("Saved final state visualization")
    
    # Create advanced visualizations
    logger.info("Creating advanced visualizations...")
    
    # Get full history
    history = runner.get_history()
    
    # Create animations
    if config['visualization']['create_animations']:
        logger.info("Creating animations...")
        advanced_visualizer.create_network_animation(history, output_dir)
        advanced_visualizer.create_phase_space_animation(history, output_dir)
    
    # Create static visualizations
    if config['visualization']['plots']['spectral_analysis']:
        logger.info("Creating spectral analysis...")
        advanced_visualizer.create_spectral_analysis(history, output_dir)
    
    if config['visualization']['plots']['correlation_matrix']:
        logger.info("Creating correlation analysis...")
        advanced_visualizer.create_correlation_analysis(history, output_dir)
    
    if config['visualization']['enable_interactive']:
        logger.info("Creating interactive dashboard...")
        advanced_visualizer.create_interactive_dashboard(history, output_dir)
    
    basic_visualizer.close()
    logger.info("Simulation complete!")
    
    return initial_states, final_states

def main():
    """Main entry point for the simulation."""
    # Load configuration
    config = load_config()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        # Setup matplotlib backend
        interactive = setup_matplotlib_backend(logger)
        
        # Run simulation
        initial_states, final_states = run_simulation(
            config,
            logger,
            output_dir,
            interactive
        )
        
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 