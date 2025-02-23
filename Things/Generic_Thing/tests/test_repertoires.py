"""Tests for visualizing Thing repertoires and their interactions."""

import unittest
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generic_thing.visualization import Visualizer
from generic_thing.core import GenericThing
from generic_thing.tests.test_utils import log_test_case, generate_test_data

logger = logging.getLogger(__name__)

class TestRepertoires(unittest.TestCase):
    """Test suite for visualizing Thing repertoires."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz = Visualizer()
        
        # Create test Things with different repertoires
        self.things = {
            'sensor': GenericThing(id="sensor", name="Sensor Thing"),
            'processor': GenericThing(id="processor", name="Processor Thing"),
            'actuator': GenericThing(id="actuator", name="Actuator Thing")
        }
        
        # Initialize repertoires for each Thing
        self.initialize_repertoires()
        logger.info("Created test Things with repertoires")
        
    def initialize_repertoires(self):
        """Initialize repertoires for each Thing with meaningful patterns."""
        # Sensor Thing specializes in sensory patterns
        self.things['sensor'].internal_state.update({
            'sensory_patterns': np.random.randn(10, 5),  # 10 patterns, 5 features
            'recognition_weights': np.random.randn(5, 3),  # 5 features to 3 categories
            'sensory_precision': np.abs(np.random.randn(5, 5)),  # Precision matrix
            'sensory_uncertainty': np.random.uniform(0, 1, (5,)),  # Uncertainty per feature
            'pattern_activations': np.random.randn(10, 1)  # Pattern activation strengths
        })
        
        # Processor Thing handles transformations
        self.things['processor'].internal_state.update({
            'transform_matrix': np.random.randn(8, 8),  # State transformation matrix
            'belief_states': np.random.randn(8, 4),  # Current beliefs about 4 variables
            'prediction_weights': np.random.randn(4, 6),  # Prediction network weights
            'attention_mask': np.random.uniform(0, 1, (8,)),  # Attention weights
            'error_signals': np.random.randn(8, 1),  # Prediction errors
            'precision_weights': np.abs(np.random.randn(8, 8))  # Precision weighting matrix
        })
        
        # Actuator Thing manages action patterns
        self.things['actuator'].internal_state.update({
            'action_repertoire': np.random.randn(6, 4),  # 6 actions, 4 parameters each
            'policy_matrix': np.random.randn(4, 3),  # Policy mapping
            'motor_commands': np.random.randn(3, 2),  # Motor command patterns
            'action_costs': np.random.uniform(0, 1, (6,)),  # Cost per action
            'success_rates': np.random.uniform(0.5, 1, (6,)),  # Success rate per action
            'action_precision': np.abs(np.random.randn(4, 4))  # Action precision matrix
        })
    
    @log_test_case
    def test_repertoire_visualization(self):
        """Test visualization of individual Thing repertoires."""
        for thing_name, thing in self.things.items():
            self.plot_thing_repertoire(thing, f"{thing_name}_repertoire.png")
            self.plot_repertoire_details(thing, f"{thing_name}_details.png")
    
    def plot_thing_repertoire(self, thing: GenericThing, filename: str):
        """Create detailed visualization of a Thing's repertoire matrices."""
        # Calculate number of subplots needed
        n_matrices = len(thing.internal_state)
        n_cols = min(3, n_matrices)
        n_rows = (n_matrices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5*n_cols, 4*n_rows),
                                squeeze=False)
        fig.suptitle(f"Repertoire Matrices for {thing.name}")
        
        # Plot each matrix
        for idx, (name, matrix) in enumerate(thing.internal_state.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Create heatmap with appropriate normalization
            if matrix.ndim == 2:
                sns.heatmap(matrix, ax=ax, cmap='viridis', 
                           center=0 if np.min(matrix) < 0 else None,
                           xticklabels=False, yticklabels=False)
            else:  # 1D arrays
                sns.heatmap(matrix.reshape(-1, 1), ax=ax, cmap='viridis',
                           center=0 if np.min(matrix) < 0 else None,
                           xticklabels=False, yticklabels=False)
            ax.set_title(name.replace('_', ' ').title())
        
        # Remove empty subplots
        for idx in range(n_matrices, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    def plot_repertoire_details(self, thing: GenericThing, filename: str):
        """Create detailed analysis plots for Thing's repertoire."""
        n_plots = 3  # Number of analysis plots per matrix
        n_matrices = len(thing.internal_state)
        
        fig = plt.figure(figsize=(15, 5*n_matrices))
        gs = plt.GridSpec(n_matrices, n_plots, figure=fig)
        fig.suptitle(f"Detailed Analysis of {thing.name} Repertoire")
        
        for idx, (name, matrix) in enumerate(thing.internal_state.items()):
            # Distribution plot
            ax1 = fig.add_subplot(gs[idx, 0])
            if matrix.ndim == 2:
                sns.histplot(matrix.flatten(), ax=ax1, kde=True)
            else:
                sns.histplot(matrix, ax=ax1, kde=True)
            ax1.set_title(f"{name} Distribution")
            
            # Correlation/Structure plot
            ax2 = fig.add_subplot(gs[idx, 1])
            if matrix.ndim == 2:
                sns.heatmap(np.corrcoef(matrix), ax=ax2, cmap='coolwarm', center=0)
                ax2.set_title(f"{name} Correlation Structure")
            else:
                sns.barplot(x=np.arange(len(matrix)), y=matrix, ax=ax2)
                ax2.set_title(f"{name} Values")
            
            # Special analysis based on matrix type
            ax3 = fig.add_subplot(gs[idx, 2])
            if 'precision' in name.lower():
                # For precision matrices, show eigenvalue spectrum
                if matrix.ndim == 2:
                    eigvals = np.linalg.eigvals(matrix)
                    sns.scatterplot(x=np.arange(len(eigvals)), y=np.abs(eigvals), ax=ax3)
                    ax3.set_title("Eigenvalue Spectrum")
            elif 'weights' in name.lower():
                # For weight matrices, show weight magnitude distribution
                sns.boxplot(data=np.abs(matrix), ax=ax3)
                ax3.set_title("Weight Magnitudes")
            else:
                # Default to showing basic statistics
                stats = {
                    'mean': np.mean(matrix),
                    'std': np.std(matrix),
                    'min': np.min(matrix),
                    'max': np.max(matrix)
                }
                ax3.bar(range(len(stats)), list(stats.values()))
                ax3.set_xticks(range(len(stats)))
                ax3.set_xticklabels(list(stats.keys()))
                ax3.set_title("Basic Statistics")
        
        plt.tight_layout()
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    @log_test_case
    def test_repertoire_interactions(self):
        """Test visualization of interactions between Thing repertoires."""
        # Create interaction matrices
        interactions = {
            ('sensor', 'processor'): np.random.randn(5, 8),  # Sensory to processing
            ('processor', 'actuator'): np.random.randn(6, 4),  # Processing to action
            ('actuator', 'sensor'): np.random.randn(2, 5)  # Action feedback to sensory
        }
        
        self.plot_repertoire_interactions(interactions, "repertoire_interactions.png")
        self.plot_interaction_analysis(interactions, "interaction_analysis.png")
    
    def plot_repertoire_interactions(self, 
                                   interactions: Dict[Tuple[str, str], np.ndarray],
                                   filename: str):
        """Visualize interactions between Thing repertoires."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create positions for Things in a circle
        n_things = len(self.things)
        angles = np.linspace(0, 2*np.pi, n_things, endpoint=False)
        radius = 0.4
        positions = {
            name: (radius * np.cos(angle) + 0.5,
                  radius * np.sin(angle) + 0.5)
            for name, angle in zip(self.things.keys(), angles)
        }
        
        # Plot Thing repertoires
        for name, pos in positions.items():
            # Create small heatmap for main repertoire matrix
            main_matrix = list(self.things[name].internal_state.values())[0]
            ax = plt.axes([pos[0]-0.15, pos[1]-0.15, 0.3, 0.3])
            sns.heatmap(main_matrix, cmap='viridis', 
                       cbar=False, xticklabels=False, yticklabels=False)
            ax.set_title(name)
        
        # Plot interaction matrices as arrows
        for (source, target), matrix in interactions.items():
            source_pos = positions[source]
            target_pos = positions[target]
            
            # Calculate arrow position
            dx = target_pos[0] - source_pos[0]
            dy = target_pos[1] - source_pos[1]
            
            # Plot arrow
            plt.arrow(source_pos[0], source_pos[1], dx*0.7, dy*0.7,
                     head_width=0.05, head_length=0.05,
                     fc='gray', ec='gray', alpha=0.5)
            
            # Add small interaction matrix visualization
            mid_x = source_pos[0] + dx*0.35
            mid_y = source_pos[1] + dy*0.35
            ax = plt.axes([mid_x-0.1, mid_y-0.1, 0.2, 0.2])
            sns.heatmap(matrix, cmap='viridis',
                       cbar=False, xticklabels=False, yticklabels=False)
        
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    def plot_interaction_analysis(self,
                                interactions: Dict[Tuple[str, str], np.ndarray],
                                filename: str):
        """Create detailed analysis of interaction matrices."""
        n_interactions = len(interactions)
        fig, axes = plt.subplots(n_interactions, 3, figsize=(15, 5*n_interactions))
        fig.suptitle("Detailed Analysis of Thing Interactions")
        
        for idx, ((source, target), matrix) in enumerate(interactions.items()):
            # Singular value decomposition analysis
            u, s, vh = np.linalg.svd(matrix)
            
            # Plot singular values
            axes[idx, 0].bar(range(len(s)), s)
            axes[idx, 0].set_title(f"{source}->{target}\nSingular Values")
            
            # Plot left singular vectors (input patterns)
            axes[idx, 1].imshow(u, aspect='auto', cmap='coolwarm')
            axes[idx, 1].set_title("Input Patterns")
            
            # Plot right singular vectors (output patterns)
            axes[idx, 2].imshow(vh, aspect='auto', cmap='coolwarm')
            axes[idx, 2].set_title("Output Patterns")
            
        plt.tight_layout()
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    @log_test_case
    def test_temporal_repertoire_evolution(self):
        """Test visualization of how repertoires evolve over time."""
        # Generate temporal evolution data
        n_timesteps = 5
        temporal_data = {
            name: {
                matrix_name: np.stack([
                    matrix + 0.1 * np.random.randn(*matrix.shape)
                    for _ in range(n_timesteps)
                ])
                for matrix_name, matrix in thing.internal_state.items()
            }
            for name, thing in self.things.items()
        }
        
        self.plot_temporal_evolution(temporal_data, "repertoire_evolution.png")
        self.plot_temporal_analysis(temporal_data, "temporal_analysis.png")
    
    def plot_temporal_evolution(self, 
                              temporal_data: Dict[str, Dict[str, np.ndarray]],
                              filename: str):
        """Visualize temporal evolution of repertoires."""
        n_things = len(temporal_data)
        n_timesteps = next(iter(temporal_data.values()))['sensory_patterns'].shape[0]
        
        fig, axes = plt.subplots(n_things, n_timesteps,
                                figsize=(4*n_timesteps, 3*n_things),
                                squeeze=False)
        fig.suptitle("Temporal Evolution of Thing Repertoires")
        
        for row, (thing_name, matrices) in enumerate(temporal_data.items()):
            # Select the first matrix for visualization
            matrix_name = list(matrices.keys())[0]
            matrix_sequence = matrices[matrix_name]
            
            for col in range(n_timesteps):
                ax = axes[row, col]
                sns.heatmap(matrix_sequence[col], ax=ax,
                           cmap='viridis', cbar=False,
                           xticklabels=False, yticklabels=False)
                
                if col == 0:
                    ax.set_ylabel(thing_name)
                if row == 0:
                    ax.set_title(f"t={col}")
        
        plt.tight_layout()
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    def plot_temporal_analysis(self,
                             temporal_data: Dict[str, Dict[str, np.ndarray]],
                             filename: str):
        """Create detailed analysis of temporal evolution."""
        n_things = len(temporal_data)
        fig, axes = plt.subplots(n_things, 3, figsize=(15, 5*n_things))
        fig.suptitle("Analysis of Temporal Evolution")
        
        for idx, (thing_name, matrices) in enumerate(temporal_data.items()):
            # Select first matrix for analysis
            matrix_name = list(matrices.keys())[0]
            matrix_sequence = matrices[matrix_name]
            
            # Plot mean evolution
            means = np.mean(matrix_sequence, axis=(1,2))
            axes[idx, 0].plot(means)
            axes[idx, 0].set_title(f"{thing_name}\nMean Evolution")
            
            # Plot variance evolution
            vars = np.var(matrix_sequence, axis=(1,2))
            axes[idx, 1].plot(vars)
            axes[idx, 1].set_title("Variance Evolution")
            
            # Plot eigenvalue evolution for 2D matrices
            if matrix_sequence.ndim == 3 and matrix_sequence.shape[1] == matrix_sequence.shape[2]:
                eigs = [np.linalg.eigvals(m) for m in matrix_sequence]
                eig_means = [np.mean(np.abs(e)) for e in eigs]
                axes[idx, 2].plot(eig_means)
                axes[idx, 2].set_title("Eigenvalue Evolution")
            else:
                axes[idx, 2].text(0.5, 0.5, "Non-square matrix\nNo eigenvalues", 
                                ha='center', va='center')
                axes[idx, 2].set_title("Eigenvalue Evolution")
        
        plt.tight_layout()
        plt.savefig(self.viz._ensure_viz_path(filename))
        plt.close()
    
    def tearDown(self):
        """Clean up test resources."""
        plt.close('all')

if __name__ == '__main__':
    unittest.main() 