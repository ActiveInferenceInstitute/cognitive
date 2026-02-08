"""Comprehensive visualization tests for continuous-time active inference."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Things.Continuous_Generic.continuous_generic import ContinuousActiveInference
from Things.Continuous_Generic.visualization import ContinuousVisualizer

# Test output directory
TEST_OUTPUT_DIR = Path("Output/tests/visualization")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def visualizer():
    """Create test visualizer."""
    return ContinuousVisualizer(TEST_OUTPUT_DIR)

@pytest.fixture
def agent():
    """Create test agent with rich dynamics."""
    return ContinuousActiveInference(
        n_states=2,
        n_obs=2,
        n_orders=3,
        dt=0.001,
        alpha=0.1
    )

def generate_oscillatory_data(agent, n_steps=1000):
    """Generate oscillatory test data."""
    # Initialize with harmonic oscillator state
    agent.state.belief_means[:, 0] = np.array([0.1, 0.0])    # Position
    agent.state.belief_means[:, 1] = np.array([0.0, 0.1])    # Velocity
    agent.state.belief_means[:, 2] = np.array([-0.1, 0.0])   # Acceleration
    
    history = {
        'belief_means': [],
        'belief_precisions': [],
        'free_energy': [],
        'actions': [],
        'time': []
    }
    
    for t in range(n_steps):
        # Record state
        history['belief_means'].append(agent.state.belief_means.copy())
        history['belief_precisions'].append(agent.state.belief_precisions.copy())
        history['time'].append(t * agent.dt)
        
        # Generate observation (harmonic motion)
        obs = agent._sensory_mapping(agent.state.belief_means)
        
        # Take step
        action, F = agent.step(obs)
        
        history['actions'].append(action)
        history['free_energy'].append(F)
    
    return history

@pytest.mark.visualization
class TestBasicVisualization:
    """Test basic visualization capabilities."""
    
    def test_belief_evolution(self, agent, visualizer):
        """Test belief evolution visualization."""
        history = generate_oscillatory_data(agent)
        
        # Basic belief evolution plot
        visualizer.plot_belief_evolution(
            history['belief_means'],
            history['belief_precisions'],
            TEST_OUTPUT_DIR / 'belief_evolution.png'
        )
        
        # Enhanced belief evolution with uncertainty
        plt.figure(figsize=(15, 10))
        means = np.array([b[:,0] for b in history['belief_means']])
        precisions = np.array([b[:,0] for b in history['belief_precisions']])
        time = np.array(history['time'])
        
        for i in range(means.shape[1]):
            plt.plot(time, means[:,i], label=f'State {i+1}')
            std = 1.0 / np.sqrt(precisions[:,i])
            plt.fill_between(time, means[:,i] - 2*std, means[:,i] + 2*std, alpha=0.2)
        
        plt.title('Belief Evolution with Uncertainty')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'belief_evolution_with_uncertainty.png')
        plt.close()

@pytest.mark.visualization
class TestPhaseSpaceVisualization:
    """Test phase space visualization capabilities."""
    
    def test_phase_space_plots(self, agent, visualizer):
        """Test phase space visualization."""
        history = generate_oscillatory_data(agent)
        
        # Basic phase space
        plt.figure(figsize=(10, 10))
        means = np.array([b[:,0] for b in history['belief_means']])
        
        plt.plot(means[:,0], means[:,1], 'b-', alpha=0.5)
        plt.plot(means[0,0], means[0,1], 'go', label='Start')
        plt.plot(means[-1,0], means[-1,1], 'ro', label='End')
        
        # Add velocity vectors
        step = len(means) // 20  # Plot 20 vectors
        for i in range(0, len(means), step):
            plt.arrow(means[i,0], means[i,1],
                     means[i,1]*0.1, -means[i,0]*0.1,
                     head_width=0.01, head_length=0.02,
                     fc='k', ec='k', alpha=0.5)
        
        plt.title('Phase Space with Velocity Vectors')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'phase_space_with_vectors.png')
        plt.close()
        
        # Energy contours in phase space
        plt.figure(figsize=(10, 10))
        x = np.linspace(-0.15, 0.15, 100)
        y = np.linspace(-0.15, 0.15, 100)
        X, Y = np.meshgrid(x, y)
        E = 0.5 * (X**2 + Y**2)  # Harmonic oscillator energy
        
        plt.contour(X, Y, E, levels=20, cmap='viridis', alpha=0.3)
        plt.plot(means[:,0], means[:,1], 'r-', alpha=0.7, label='Trajectory')
        plt.colorbar(label='Energy')
        
        plt.title('Phase Space with Energy Contours')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'phase_space_energy.png')
        plt.close()

@pytest.mark.visualization
class TestEnergyVisualization:
    """Test energy-related visualization capabilities."""
    
    def test_energy_plots(self, agent, visualizer):
        """Test energy visualization."""
        history = generate_oscillatory_data(agent)
        means = np.array([b[:,0] for b in history['belief_means']])
        time = np.array(history['time'])
        
        # Calculate energies
        ke = 0.5 * means[:,1]**2  # Kinetic energy
        pe = 0.5 * means[:,0]**2  # Potential energy
        total = ke + pe
        
        # Energy components plot
        plt.figure(figsize=(15, 5))
        plt.plot(time, ke, 'r-', label='Kinetic', alpha=0.7)
        plt.plot(time, pe, 'b-', label='Potential', alpha=0.7)
        plt.plot(time, total, 'k--', label='Total')
        
        plt.title('Energy Components')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'energy_components.png')
        plt.close()
        
        # Energy ratio plot
        plt.figure(figsize=(15, 5))
        plt.plot(time, ke/pe, 'g-', label='KE/PE Ratio')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Equipartition')
        
        plt.title('Kinetic to Potential Energy Ratio')
        plt.xlabel('Time')
        plt.ylabel('KE/PE Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'energy_ratio.png')
        plt.close()

@pytest.mark.visualization
class TestGeneralizedCoordinates:
    """Test visualization of generalized coordinates relationships."""
    
    def test_coordinate_relationships(self, agent, visualizer):
        """Test generalized coordinates visualization."""
        history = generate_oscillatory_data(agent)
        
        # Plot relationships between orders
        visualizer.plot_generalized_coordinates_relationships(
            history['belief_means'],
            np.array(history['time']),
            TEST_OUTPUT_DIR / 'generalized_coordinates.png'
        )
        
        # Create animation
        visualizer.save_animation(
            history,
            TEST_OUTPUT_DIR / 'belief_animation.gif',
            fps=30
        )

@pytest.mark.visualization
class TestFreeEnergyLandscape:
    """Test visualization of free energy landscape."""
    
    def test_free_energy_landscape(self, agent, visualizer):
        """Test free energy landscape visualization."""
        # Generate grid of beliefs
        x = np.linspace(-0.2, 0.2, 50)
        y = np.linspace(-0.2, 0.2, 50)
        X, Y = np.meshgrid(x, y)
        F = np.zeros_like(X)
        
        # Calculate free energy for each point
        for i in range(len(x)):
            for j in range(len(y)):
                agent.state.belief_means[:,0] = np.array([X[i,j], Y[i,j]])
                obs = agent._sensory_mapping(agent.state.belief_means)
                F[i,j] = agent._compute_free_energy(obs, agent.state.belief_means)
        
        # Plot free energy landscape
        plt.figure(figsize=(12, 10))
        plt.contour(X, Y, F, levels=30, cmap='viridis')
        plt.colorbar(label='Free Energy')
        
        # Add minimum point
        min_idx = np.unravel_index(np.argmin(F), F.shape)
        plt.plot(X[min_idx], Y[min_idx], 'r*', markersize=15, label='Minimum')
        
        plt.title('Free Energy Landscape')
        plt.xlabel('State 1')
        plt.ylabel('State 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'free_energy_landscape.png')
        plt.close()

@pytest.mark.visualization
class TestActionVisualization:
    """Test visualization of action selection."""
    
    def test_action_plots(self, agent, visualizer):
        """Test action visualization."""
        history = generate_oscillatory_data(agent)
        actions = np.array(history['actions'])
        time = np.array(history['time'])
        
        # Action evolution
        plt.figure(figsize=(15, 5))
        for i in range(actions.shape[1]):
            plt.plot(time, actions[:,i], label=f'Action {i+1}')
        
        plt.title('Action Evolution')
        plt.xlabel('Time')
        plt.ylabel('Action Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(TEST_OUTPUT_DIR / 'action_evolution.png')
        plt.close()
        
        # Action phase space
        if actions.shape[1] >= 2:
            plt.figure(figsize=(10, 10))
            plt.plot(actions[:,0], actions[:,1], 'b-', alpha=0.5)
            plt.plot(actions[0,0], actions[0,1], 'go', label='Start')
            plt.plot(actions[-1,0], actions[-1,1], 'ro', label='End')
            
            plt.title('Action Phase Space')
            plt.xlabel('Action 1')
            plt.ylabel('Action 2')
            plt.legend()
            plt.grid(True)
            plt.savefig(TEST_OUTPUT_DIR / 'action_phase_space.png')
            plt.close()

@pytest.mark.visualization
class TestSummaryVisualization:
    """Test comprehensive summary visualization."""
    
    def test_summary_plots(self, agent, visualizer):
        """Test summary visualization."""
        history = generate_oscillatory_data(agent)
        
        # Create comprehensive summary plot
        visualizer.create_summary_plot(
            history,
            TEST_OUTPUT_DIR / 'summary.png'
        )
        
        # Create detailed diagnostic plots
        plt.figure(figsize=(20, 15))
        
        # Belief evolution with uncertainty
        plt.subplot(3, 2, 1)
        means = np.array([b[:,0] for b in history['belief_means']])
        precisions = np.array([b[:,0] for b in history['belief_precisions']])
        time = np.array(history['time'])
        
        for i in range(means.shape[1]):
            plt.plot(time, means[:,i], label=f'State {i+1}')
            std = 1.0 / np.sqrt(precisions[:,i])
            plt.fill_between(time, means[:,i] - 2*std, means[:,i] + 2*std, alpha=0.2)
        
        plt.title('Belief Evolution')
        plt.xlabel('Time')
        plt.ylabel('State Value')
        plt.legend()
        plt.grid(True)
        
        # Phase space
        plt.subplot(3, 2, 2)
        plt.plot(means[:,0], means[:,1], 'b-', alpha=0.5)
        plt.plot(means[0,0], means[0,1], 'go', label='Start')
        plt.plot(means[-1,0], means[-1,1], 'ro', label='End')
        
        plt.title('Phase Space')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.legend()
        plt.grid(True)
        
        # Free energy
        plt.subplot(3, 2, 3)
        plt.plot(time, history['free_energy'])
        plt.title('Free Energy Evolution')
        plt.xlabel('Time')
        plt.ylabel('Free Energy')
        plt.grid(True)
        
        # Actions
        plt.subplot(3, 2, 4)
        actions = np.array(history['actions'])
        for i in range(actions.shape[1]):
            plt.plot(time, actions[:,i], label=f'Action {i+1}')
        
        plt.title('Action Evolution')
        plt.xlabel('Time')
        plt.ylabel('Action Value')
        plt.legend()
        plt.grid(True)
        
        # Energy components
        plt.subplot(3, 2, 5)
        ke = 0.5 * means[:,1]**2
        pe = 0.5 * means[:,0]**2
        total = ke + pe
        
        plt.plot(time, ke, 'r-', label='Kinetic', alpha=0.7)
        plt.plot(time, pe, 'b-', label='Potential', alpha=0.7)
        plt.plot(time, total, 'k--', label='Total')
        
        plt.title('Energy Components')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        
        # Generalized coordinates relationships
        plt.subplot(3, 2, 6)
        dt = time[1] - time[0]
        dx = np.diff(means[:,0]) / dt
        v = means[:-1,1]
        
        plt.scatter(v, dx, alpha=0.5, s=10)
        lims = [
            min(plt.xlim()[0], plt.ylim()[0]),
            max(plt.xlim()[1], plt.ylim()[1])
        ]
        plt.plot(lims, lims, 'r--', alpha=0.5)
        
        plt.title('Velocity vs Position Derivative')
        plt.xlabel('Velocity')
        plt.ylabel('Position Derivative')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(TEST_OUTPUT_DIR / 'detailed_summary.png')
        plt.close() 