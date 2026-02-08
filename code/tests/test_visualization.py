"""
Tests for visualization components.
"""

import pytest
import numpy as np
from src.visualization.matrix_plots import MatrixPlotter, StateSpacePlotter, NetworkPlotter
import matplotlib.pyplot as plt

@pytest.fixture
def style_config():
    """Default style configuration for testing."""
    return {
        'theme': 'default',
        'figure_size': (8, 6),
        'dpi': 100,
        'colormap': 'viridis',
        'font_size': 12,
        'line_width': 1.5
    }

class TestMatrixPlotter:
    """Test matrix plotting utilities."""
    
    def test_plot_heatmap(self, sample_matrix_2d, output_dir, style_config):
        """Test heatmap plotting."""
        plotter = MatrixPlotter(output_dir, style_config)
        fig = plotter.plot_heatmap(
            matrix=sample_matrix_2d,
            title="Test Heatmap",
            xlabel="States",
            ylabel="Observations",
            save_name="test_heatmap"
        )
        
        # Check figure properties
        assert isinstance(fig, plt.Figure)
        # Main axis and colorbar
        assert len(fig.axes) == 2
        # Check main axis properties
        main_ax = fig.axes[0]
        assert main_ax.get_title() == "Test Heatmap"
        assert main_ax.get_xlabel() == "States"
        assert main_ax.get_ylabel() == "Observations"
        # Verify file was saved
        assert (output_dir / "test_heatmap.png").exists()
    
    def test_plot_multi_heatmap(self, sample_matrix_3d, output_dir, style_config):
        """Test multiple heatmap plotting."""
        plotter = MatrixPlotter(output_dir, style_config)
        fig = plotter.plot_multi_heatmap(
            tensor=sample_matrix_3d,
            title="Test Multi-Heatmap",
            xlabel="Current State",
            ylabel="Next State",
            slice_names=["Action 1", "Action 2"],
            save_name="test_multi_heatmap"
        )
        
        # Check figure properties
        assert isinstance(fig, plt.Figure)
        # Two main axes and two colorbars
        assert len(fig.axes) == 4
        # Check titles
        assert fig.axes[0].get_title() == "Test Multi-Heatmap - Action 1"
        assert fig.axes[1].get_title() == "Test Multi-Heatmap - Action 2"
        # Verify file was saved
        assert (output_dir / "test_multi_heatmap.png").exists()
    
    def test_plot_bar(self, sample_belief_vector, output_dir, style_config):
        """Test bar plot creation."""
        plotter = MatrixPlotter(output_dir, style_config)
        fig = plotter.plot_bar(
            values=sample_belief_vector,
            title="Test Bar Plot",
            xlabel="States",
            ylabel="Probability",
            save_name="test_bar"
        )
        
        # Check figure properties
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Test Bar Plot"
        assert ax.get_xlabel() == "States"
        assert ax.get_ylabel() == "Probability"
        assert (output_dir / "test_bar.png").exists()

class TestStateSpacePlotter:
    """Test state space plotting utilities."""
    
    def test_plot_belief_evolution(self, output_dir):
        """Test belief evolution plotting."""
        plotter = StateSpacePlotter(output_dir)
        beliefs = np.array([[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]])
        fig = plotter.plot_belief_evolution(
            beliefs=beliefs,
            title="Belief Evolution",
            state_labels=["State 1", "State 2"],
            save_name="test_belief_evolution"
        )
        assert isinstance(fig, plt.Figure)
        # Verify file was saved
        assert (output_dir / "test_belief_evolution.png").exists()
    
    def test_plot_free_energy_landscape(self, output_dir):
        """Test free energy landscape plotting."""
        plotter = StateSpacePlotter(output_dir)
        free_energy = np.array([[1.0, 2.0], [2.0, 3.0]])
        fig = plotter.plot_free_energy_landscape(
            free_energy=free_energy,
            title="Free Energy Landscape",
            save_name="test_landscape"
        )
        assert isinstance(fig, plt.Figure)
        # Verify file was saved
        assert (output_dir / "test_landscape.png").exists()
    
    def test_plot_policy_evaluation(self, output_dir):
        """Test policy evaluation plotting."""
        plotter = StateSpacePlotter(output_dir)
        policy_values = np.array([0.8, 0.6, 0.4])
        fig = plotter.plot_policy_evaluation(
            policy_values=policy_values,
            policy_labels=["Policy 1", "Policy 2", "Policy 3"],
            title="Policy Evaluation",
            save_name="test_policy_eval"
        )
        assert isinstance(fig, plt.Figure)
        # Verify file was saved
        assert (output_dir / "test_policy_eval.png").exists()

class TestNetworkPlotter:
    """Test network plotting utilities."""
    
    def test_plot_belief_network(self, output_dir):
        """Test belief network plotting."""
        plotter = NetworkPlotter(output_dir)
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        fig = plotter.plot_belief_network(
            adjacency=adjacency,
            node_labels=["A", "B", "C"],
            title="Belief Network",
            save_name="test_network"
        )
        assert isinstance(fig, plt.Figure)
        # Verify file was saved
        assert (output_dir / "test_network.png").exists()

@pytest.mark.parametrize("matrix_shape,expected_axes", [
    ((2, 2), 2),  # Main axis + colorbar
    ((3, 3), 2),
    ((4, 4), 2)
])
def test_heatmap_shapes(matrix_shape, expected_axes, output_dir, style_config):
    """Test heatmap plotting with different matrix shapes."""
    plotter = MatrixPlotter(output_dir, style_config)
    matrix = np.random.rand(*matrix_shape)
    fig = plotter.plot_heatmap(
        matrix=matrix,
        title=f"Test {matrix_shape[0]}x{matrix_shape[1]} Heatmap",
        save_name=f"test_heatmap_{matrix_shape[0]}x{matrix_shape[1]}"
    )
    
    assert len(fig.axes) == expected_axes
    # Get the data from the heatmap
    heatmap_data = fig.axes[0].collections[0].get_array()
    # Reshape the flattened data back to the original shape
    heatmap_data = heatmap_data.reshape(matrix_shape)
    assert heatmap_data.shape == matrix_shape
    # Verify file was saved
    assert (output_dir / f"test_heatmap_{matrix_shape[0]}x{matrix_shape[1]}.png").exists() 