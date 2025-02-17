"""
Visualization implementation for BioFirm framework.
Provides comprehensive plotting and visualization tools.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import networkx as nx
from scipy.stats import gaussian_kde

from ..core.state_spaces import BioregionalState
from ..core.stewardship import Intervention, StewardshipMetrics

class BioregionalVisualization:
    """Comprehensive bioregional visualization tools."""
    
    def __init__(self, style: str = "seaborn-whitegrid"):
        """Initialize visualization suite with style."""
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        
    def plot_system_state(self,
                         bioregional_state: BioregionalState,
                         time_series: Optional[np.ndarray] = None
                         ) -> Figure:
        """Visualize multi-dimensional system state."""
        fig = plt.figure(figsize=(15, 10))
        
        if time_series is not None:
            # Plot time series
            self._plot_time_series(fig, time_series, bioregional_state)
        else:
            # Plot current state
            self._plot_current_state(fig, bioregional_state)
            
        plt.tight_layout()
        return fig
        
    def _plot_time_series(self,
                         fig: Figure,
                         time_series: np.ndarray,
                         state: BioregionalState):
        """Plot time series of state variables."""
        gs = fig.add_gridspec(2, 2)
        
        # Ecological time series
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_domain_time_series(
            ax1, time_series, state.ecological_state,
            "Ecological Metrics", self.colors[0]
        )
        
        # Climate time series
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_domain_time_series(
            ax2, time_series, state.climate_state,
            "Climate Metrics", self.colors[1]
        )
        
        # Social time series
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_domain_time_series(
            ax3, time_series, state.social_state,
            "Social Metrics", self.colors[2]
        )
        
        # Economic time series
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_domain_time_series(
            ax4, time_series, state.economic_state,
            "Economic Metrics", self.colors[3]
        )
        
    def _plot_domain_time_series(self,
                                ax: plt.Axes,
                                time_series: np.ndarray,
                                state_dict: Dict[str, float],
                                title: str,
                                color: Tuple[float, float, float]):
        """Plot time series for a specific domain."""
        for i, (var, _) in enumerate(state_dict.items()):
            ax.plot(time_series[:, i],
                   label=var,
                   color=color,
                   alpha=0.5 + 0.5 * i/len(state_dict))
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
    def _plot_current_state(self, fig: Figure, state: BioregionalState):
        """Plot current state as radar charts."""
        gs = fig.add_gridspec(2, 2)
        
        # Ecological radar
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        self._plot_domain_radar(
            ax1, state.ecological_state,
            "Ecological State", self.colors[0]
        )
        
        # Climate radar
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        self._plot_domain_radar(
            ax2, state.climate_state,
            "Climate State", self.colors[1]
        )
        
        # Social radar
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        self._plot_domain_radar(
            ax3, state.social_state,
            "Social State", self.colors[2]
        )
        
        # Economic radar
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        self._plot_domain_radar(
            ax4, state.economic_state,
            "Economic State", self.colors[3]
        )
        
    def _plot_domain_radar(self,
                          ax: plt.Axes,
                          state_dict: Dict[str, float],
                          title: str,
                          color: Tuple[float, float, float]):
        """Plot radar chart for a specific domain."""
        variables = list(state_dict.keys())
        values = list(state_dict.values())
        
        angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # complete the loop
        angles = np.concatenate((angles, [angles[0]]))  # complete the loop
        
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables)
        ax.set_title(title)
        
    def plot_intervention_impacts(self,
                                before_state: BioregionalState,
                                after_state: BioregionalState,
                                intervention_data: Dict[str, Any]) -> Figure:
        """Visualize intervention outcomes."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # State changes
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_state_changes(ax1, before_state, after_state)
        
        # Intervention details
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_intervention_details(ax2, intervention_data)
        
        # Uncertainty analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_uncertainty_analysis(ax3, intervention_data)
        
        plt.tight_layout()
        return fig
        
    def _plot_state_changes(self,
                           ax: plt.Axes,
                           before: BioregionalState,
                           after: BioregionalState):
        """Plot before/after state changes."""
        variables = []
        before_values = []
        after_values = []
        
        # Collect all variables and values
        for domain in ["ecological", "climate", "social", "economic"]:
            before_dict = getattr(before, f"{domain}_state")
            after_dict = getattr(after, f"{domain}_state")
            
            for var in before_dict.keys():
                variables.append(f"{domain}.{var}")
                before_values.append(before_dict[var])
                after_values.append(after_dict[var])
                
        # Plot
        x = np.arange(len(variables))
        width = 0.35
        
        ax.bar(x - width/2, before_values, width, label='Before',
               color='lightgray')
        ax.bar(x + width/2, after_values, width, label='After',
               color='darkgreen')
        
        ax.set_ylabel('Value')
        ax.set_title('State Changes from Intervention')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        
    def _plot_intervention_details(self,
                                 ax: plt.Axes,
                                 intervention_data: Dict[str, Any]):
        """Plot intervention details."""
        # Extract key metrics
        metrics = {
            'Duration': intervention_data.get('duration', 0),
            'Intensity': intervention_data.get('intensity', 0),
            'Cost': intervention_data.get('resources', {}).get('budget', 0),
            'Success Prob': 1 - intervention_data.get('uncertainty', 0)
        }
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metrics))
        ax.barh(y_pos, list(metrics.values()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(metrics.keys()))
        ax.set_title('Intervention Metrics')
        
    def _plot_uncertainty_analysis(self,
                                 ax: plt.Axes,
                                 intervention_data: Dict[str, Any]):
        """Plot uncertainty analysis."""
        # Generate synthetic uncertainty data
        outcomes = intervention_data.get('expected_outcomes', {})
        uncertainties = np.random.normal(
            loc=list(outcomes.values()),
            scale=intervention_data.get('uncertainty', 0.1),
            size=(1000, len(outcomes))
        )
        
        # Plot density
        for i, (var, _) in enumerate(outcomes.items()):
            density = gaussian_kde(uncertainties[:, i])
            xs = np.linspace(0, 1, 200)
            ax.plot(xs, density(xs), label=var)
            
        ax.set_title('Outcome Uncertainty')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        
    def plot_cross_scale_dynamics(self,
                                states: Dict[str, np.ndarray],
                                scales: List[str],
                                interactions: np.ndarray) -> Figure:
        """Visualize cross-scale ecological dynamics."""
        fig = plt.figure(figsize=(15, 10))
        
        # Network visualization
        ax1 = fig.add_subplot(121)
        self._plot_scale_network(ax1, scales, interactions)
        
        # Scale correlations
        ax2 = fig.add_subplot(122)
        self._plot_scale_correlations(ax2, states, scales)
        
        plt.tight_layout()
        return fig
        
    def _plot_scale_network(self,
                           ax: plt.Axes,
                           scales: List[str],
                           interactions: np.ndarray):
        """Plot network of cross-scale interactions."""
        G = nx.DiGraph()
        
        # Add nodes
        for scale in scales:
            G.add_node(scale)
            
        # Add edges
        n_scales = len(scales)
        for i in range(n_scales):
            for j in range(n_scales):
                if i != j and interactions[i, j] > 0:
                    G.add_edge(scales[i], scales[j],
                             weight=interactions[i, j])
                    
        # Draw network
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                             node_size=1000, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        edges = nx.draw_networkx_edges(
            G, pos,
            edge_color=[G[u][v]['weight'] for u, v in G.edges()],
            edge_cmap=plt.cm.YlOrRd,
            width=2,
            ax=ax
        )
        
        plt.colorbar(edges, ax=ax, label='Interaction Strength')
        ax.set_title('Cross-scale Interactions')
        
    def _plot_scale_correlations(self,
                                ax: plt.Axes,
                                states: Dict[str, np.ndarray],
                                scales: List[str]):
        """Plot correlations between scales."""
        n_scales = len(scales)
        correlations = np.zeros((n_scales, n_scales))
        
        # Compute correlations
        for i, scale1 in enumerate(scales):
            for j, scale2 in enumerate(scales):
                if i != j:
                    corr = np.corrcoef(
                        states[scale1].mean(axis=1),
                        states[scale2].mean(axis=1)
                    )[0, 1]
                    correlations[i, j] = corr
                    
        # Plot heatmap
        sns.heatmap(correlations,
                   xticklabels=scales,
                   yticklabels=scales,
                   cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   ax=ax)
        ax.set_title('Cross-scale Correlations') 