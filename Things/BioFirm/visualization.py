"""
Visualization suite for Earth Systems Active Inference.
Provides comprehensive tools for visualizing multi-scale dynamics and interventions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .earth_systems import (
    SystemState, EcologicalState, ClimateState, HumanImpactState
)

class MultiScaleViz:
    """Visualization tools for multi-scale dynamics."""
    
    @staticmethod
    def plot_temporal_hierarchy(
        states: Dict[str, List[SystemState]],
        metrics: Dict[str, List[float]],
        time_range: Tuple[float, float]
    ) -> plt.Figure:
        """Plot temporal hierarchy dynamics."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.5])
        
        # Plot micro scale
        ax1 = fig.add_subplot(gs[0])
        MultiScaleViz._plot_micro_dynamics(ax1, states["micro"], metrics["micro"])
        
        # Plot meso scale
        ax2 = fig.add_subplot(gs[1])
        MultiScaleViz._plot_meso_dynamics(ax2, states["meso"], metrics["meso"])
        
        # Plot macro scale
        ax3 = fig.add_subplot(gs[2])
        MultiScaleViz._plot_macro_dynamics(ax3, states["macro"], metrics["macro"])
        
        # Plot coupling strengths
        ax4 = fig.add_subplot(gs[3])
        MultiScaleViz._plot_scale_coupling(ax4, states)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_spatial_hierarchy(
        states: Dict[str, SystemState],
        scale_maps: Dict[str, np.ndarray]
    ) -> plt.Figure:
        """Plot spatial hierarchy visualization."""
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(2, 2)
        
        # Plot local scale
        ax1 = fig.add_subplot(gs[0, 0])
        MultiScaleViz._plot_local_map(ax1, scale_maps["local"])
        
        # Plot regional scale
        ax2 = fig.add_subplot(gs[0, 1])
        MultiScaleViz._plot_regional_map(ax2, scale_maps["regional"])
        
        # Plot biome scale
        ax3 = fig.add_subplot(gs[1, :])
        MultiScaleViz._plot_biome_map(ax3, scale_maps["biome"])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_micro_dynamics(ax: plt.Axes,
                           states: List[SystemState],
                           metrics: List[float]):
        """Plot micro-scale dynamics."""
        times = [s.timestamp for s in states]
        ax.plot(times, metrics, label='Performance', color='blue')
        ax.set_title('Micro-scale Dynamics (Hourly)')
        ax.set_ylabel('System State')
        ax.grid(True)
    
    @staticmethod
    def _plot_meso_dynamics(ax: plt.Axes,
                           states: List[SystemState],
                           metrics: List[float]):
        """Plot meso-scale dynamics."""
        times = [s.timestamp for s in states]
        ax.plot(times, metrics, label='Performance', color='green')
        ax.set_title('Meso-scale Dynamics (Daily)')
        ax.set_ylabel('System State')
        ax.grid(True)
    
    @staticmethod
    def _plot_macro_dynamics(ax: plt.Axes,
                           states: List[SystemState],
                           metrics: List[float]):
        """Plot macro-scale dynamics."""
        times = [s.timestamp for s in states]
        ax.plot(times, metrics, label='Performance', color='red')
        ax.set_title('Macro-scale Dynamics (Monthly)')
        ax.set_ylabel('System State')
        ax.grid(True)
    
    @staticmethod
    def _plot_scale_coupling(ax: plt.Axes,
                           states: Dict[str, List[SystemState]]):
        """Plot scale coupling strengths."""
        scales = list(states.keys())
        coupling_matrix = np.random.rand(len(scales), len(scales))  # Placeholder
        sns.heatmap(coupling_matrix, ax=ax, xticklabels=scales, yticklabels=scales)
        ax.set_title('Scale Coupling Strengths')

class StateSpaceViz:
    """Visualization tools for state space analysis."""
    
    @staticmethod
    def plot_state_space(
        state: SystemState,
        history: List[SystemState],
        predictions: Optional[List[SystemState]] = None
    ) -> go.Figure:
        """Create interactive 3D state space visualization."""
        fig = go.Figure()
        
        # Plot historical trajectory
        historical_points = StateSpaceViz._extract_trajectory(history)
        fig.add_trace(go.Scatter3d(
            x=historical_points[:, 0],
            y=historical_points[:, 1],
            z=historical_points[:, 2],
            mode='lines',
            name='History',
            line=dict(color='blue', width=2)
        ))
        
        # Plot current state
        current_point = StateSpaceViz._state_to_point(state)
        fig.add_trace(go.Scatter3d(
            x=[current_point[0]],
            y=[current_point[1]],
            z=[current_point[2]],
            mode='markers',
            name='Current',
            marker=dict(size=8, color='red')
        ))
        
        # Plot predictions if available
        if predictions:
            predicted_points = StateSpaceViz._extract_trajectory(predictions)
            fig.add_trace(go.Scatter3d(
                x=predicted_points[:, 0],
                y=predicted_points[:, 1],
                z=predicted_points[:, 2],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='green', dash='dash'),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Ecological Health',
                yaxis_title='Climate Stability',
                zaxis_title='Social Wellbeing'
            ),
            title='System State Space Trajectory'
        )
        
        return fig
    
    @staticmethod
    def _extract_trajectory(states: List[SystemState]) -> np.ndarray:
        """Extract trajectory points from states."""
        return np.array([StateSpaceViz._state_to_point(s) for s in states])
    
    @staticmethod
    def _state_to_point(state: SystemState) -> np.ndarray:
        """Convert system state to 3D point."""
        return np.array([
            np.mean(list(state.ecological.biodiversity.values())),
            np.mean(list(state.climate.temperature.values())),
            np.mean(list(state.human.social_indicators.values()))
        ])

class InterventionViz:
    """Visualization tools for intervention analysis."""
    
    @staticmethod
    def plot_intervention_impacts(
        before: SystemState,
        after: SystemState,
        intervention: Dict[str, Any]
    ) -> plt.Figure:
        """Visualize intervention impacts."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Ecological Impacts',
                'Climate Impacts',
                'Social Impacts',
                'Overall Metrics'
            )
        )
        
        # Plot ecological impacts
        InterventionViz._plot_ecological_impacts(
            fig, before.ecological, after.ecological, row=1, col=1
        )
        
        # Plot climate impacts
        InterventionViz._plot_climate_impacts(
            fig, before.climate, after.climate, row=1, col=2
        )
        
        # Plot social impacts
        InterventionViz._plot_social_impacts(
            fig, before.human, after.human, row=2, col=1
        )
        
        # Plot overall metrics
        InterventionViz._plot_overall_metrics(
            fig, before, after, intervention, row=2, col=2
        )
        
        fig.update_layout(height=800, width=1000, title_text="Intervention Impacts")
        return fig
    
    @staticmethod
    def _plot_ecological_impacts(fig: go.Figure,
                               before: EcologicalState,
                               after: EcologicalState,
                               row: int,
                               col: int):
        """Plot ecological impact metrics."""
        metrics = ['biodiversity', 'soil_health', 'resilience']
        before_vals = [np.mean(list(getattr(before, m).values())) for m in metrics]
        after_vals = [np.mean(list(getattr(after, m).values())) for m in metrics]
        
        fig.add_trace(
            go.Bar(name='Before', x=metrics, y=before_vals),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(name='After', x=metrics, y=after_vals),
            row=row, col=col
        )
    
    @staticmethod
    def _plot_climate_impacts(fig: go.Figure,
                            before: ClimateState,
                            after: ClimateState,
                            row: int,
                            col: int):
        """Plot climate impact metrics."""
        metrics = ['temperature', 'precipitation', 'carbon_cycles']
        before_vals = [np.mean(list(getattr(before, m).values())) for m in metrics]
        after_vals = [np.mean(list(getattr(after, m).values())) for m in metrics]
        
        fig.add_trace(
            go.Bar(name='Before', x=metrics, y=before_vals),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(name='After', x=metrics, y=after_vals),
            row=row, col=col
        )
    
    @staticmethod
    def _plot_social_impacts(fig: go.Figure,
                           before: HumanImpactState,
                           after: HumanImpactState,
                           row: int,
                           col: int):
        """Plot social impact metrics."""
        metrics = ['social_indicators', 'restoration_efforts']
        before_vals = [np.mean(list(getattr(before, m).values())) for m in metrics]
        after_vals = [np.mean(list(getattr(after, m).values())) for m in metrics]
        
        fig.add_trace(
            go.Bar(name='Before', x=metrics, y=before_vals),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(name='After', x=metrics, y=after_vals),
            row=row, col=col
        )
    
    @staticmethod
    def _plot_overall_metrics(fig: go.Figure,
                            before: SystemState,
                            after: SystemState,
                            intervention: Dict[str, Any],
                            row: int,
                            col: int):
        """Plot overall impact metrics."""
        from .earth_systems import EarthSystemMetrics
        
        before_metrics = EarthSystemMetrics.compute_metrics(before)
        after_metrics = EarthSystemMetrics.compute_metrics(after)
        
        metrics = list(before_metrics.keys())
        before_vals = [before_metrics[m] for m in metrics]
        after_vals = [after_metrics[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(name='Before', x=metrics, y=before_vals),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(name='After', x=metrics, y=after_vals),
            row=row, col=col
        ) 