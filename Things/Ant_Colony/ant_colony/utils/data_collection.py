"""
Data collection and analysis utilities for ant colony simulation.
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

class DataCollector:
    """Handles data collection and storage for simulation analysis."""
    
    def __init__(self, config: dict):
        """Initialize data collector."""
        self.config = config
        self.output_dir = config.get('output_directory', 'output')
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data storage
        self.data = {
            'time': [],
            'colony_stats': [],
            'resource_stats': [],
            'task_distribution': [],
            'pheromone_stats': []
        }
        
    def collect(self, time: float, world_state: dict) -> None:
        """Collect data for current timestep."""
        self.data['time'].append(time)
        
        # Collect colony statistics
        colony_stats = self._collect_colony_stats(world_state)
        self.data['colony_stats'].append(colony_stats)
        
        # Collect resource statistics
        resource_stats = self._collect_resource_stats(world_state)
        self.data['resource_stats'].append(resource_stats)
        
        # Collect task distribution
        task_dist = self._collect_task_distribution(world_state)
        self.data['task_distribution'].append(task_dist)
        
        # Collect pheromone statistics
        pheromone_stats = self._collect_pheromone_stats(world_state)
        self.data['pheromone_stats'].append(pheromone_stats)
        
    def _collect_colony_stats(self, world_state: dict) -> dict:
        """Collect colony-level statistics."""
        agents = world_state['agents']
        return {
            'population': len(agents),
            'mean_energy': np.mean([agent.energy for agent in agents]),
            'total_food_collected': sum(1 for agent in agents if agent.carrying),
            'mean_distance_from_nest': np.mean([
                np.sqrt(agent.position.x**2 + agent.position.y**2)
                for agent in agents
            ])
        }
        
    def _collect_resource_stats(self, world_state: dict) -> dict:
        """Collect resource-related statistics."""
        resources = world_state['resources']
        return {
            'total_food': sum(r.amount for r in resources if r.type == 'food'),
            'num_food_sources': sum(1 for r in resources if r.type == 'food'),
            'mean_food_per_source': np.mean([
                r.amount for r in resources if r.type == 'food'
            ] or [0])
        }
        
    def _collect_task_distribution(self, world_state: dict) -> dict:
        """Collect task distribution statistics."""
        agents = world_state['agents']
        tasks = {}
        for agent in agents:
            task = agent.current_task.value
            tasks[task] = tasks.get(task, 0) + 1
        return tasks
        
    def _collect_pheromone_stats(self, world_state: dict) -> dict:
        """Collect pheromone-related statistics."""
        pheromones = world_state['pheromones']
        return {
            ptype: {
                'mean': float(np.mean(grid)),
                'max': float(np.max(grid)),
                'coverage': float(np.sum(grid > 0) / grid.size)
            }
            for ptype, grid in pheromones.items()
        }
        
    def save_data(self, filename_prefix: str = None) -> None:
        """Save collected data to files."""
        if filename_prefix is None:
            filename_prefix = f'simulation_{self.run_id}'
            
        # Save raw data as HDF5
        h5_path = os.path.join(self.output_dir, f'{filename_prefix}.h5')
        with h5py.File(h5_path, 'w') as f:
            # Save time series
            f.create_dataset('time', data=np.array(self.data['time']))
            
            # Save colony stats
            colony_stats = pd.DataFrame(self.data['colony_stats'])
            f.create_dataset('colony_stats', data=colony_stats.to_numpy())
            
            # Save task distribution
            task_dist = pd.DataFrame(self.data['task_distribution'])
            f.create_dataset('task_distribution', data=task_dist.to_numpy())
            
            # Save metadata
            f.attrs['config'] = json.dumps(self.config)
            f.attrs['run_id'] = self.run_id
            
        # Save summary statistics as JSON
        summary = self._compute_summary_statistics()
        json_path = os.path.join(self.output_dir, f'{filename_prefix}_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from collected data."""
        colony_stats = pd.DataFrame(self.data['colony_stats'])
        task_dist = pd.DataFrame(self.data['task_distribution'])
        
        summary = {
            'run_id': self.run_id,
            'duration': len(self.data['time']),
            'colony_stats': {
                'mean_population': float(np.mean([stats['population'] for stats in self.data['colony_stats']])),
                'peak_population': int(max(stats['population'] for stats in self.data['colony_stats'])),
                'total_food_collected': float(sum(stats['total_food_collected'] for stats in self.data['colony_stats']))
            },
            'task_distribution': {
                task: float(np.mean([dist.get(task, 0) for dist in self.data['task_distribution']]))
                for task in set().union(*self.data['task_distribution'])
            }
        }
        
        return summary 