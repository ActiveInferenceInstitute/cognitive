"""
Main entry point for the Ant Colony Simulation.
"""

import argparse
import logging
from pathlib import Path
from ant_colony.simulation import Simulation

def main():
    """Main entry point for the ant colony simulation."""
    parser = argparse.ArgumentParser(description='Run Ant Colony Simulation')
    parser.add_argument('--config', type=str, default='config/simulation_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run simulation
        sim = Simulation(args.config)
        sim.run(headless=args.headless)
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 