#!/usr/bin/env python3
"""
Run script for SimplePOMDP simulation.

This script follows a structured cognitive modeling workflow:
1. Model Configuration
2. Matrix Validation & Visualization
3. Model Component Analysis
4. Simulation Execution
5. Results Analysis & Visualization
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from simple_pomdp import SimplePOMDP

def create_config():
    """Create configuration for the POMDP model."""
    return {
        'model': {
            'name': 'SimplePOMDP',
            'description': 'Three-state POMDP with Active Inference',
            'version': '0.1.0'
        },
        'state_space': {
            'num_states': 3,
            'state_labels': ['Low', 'Medium', 'High'],
            'initial_state': 1  # Start in Medium state
        },
        'observation_space': {
            'num_observations': 3,
            'observation_labels': ['Low', 'Medium', 'High'],
        },
        'action_space': {
            'num_actions': 3,
            'action_labels': ['Decrease', 'Stay', 'Increase']
        },
        'matrices': {
            'A_matrix': {
                'shape': [3, 3],  # [num_observations, num_states]
                'initialization': 'identity_based',
                'initialization_params': {'strength': 0.7},  # 70% accurate observations
                'constraints': ['column_stochastic']
            },
            'B_matrix': {
                'shape': [3, 3, 3],  # [next_state, current_state, action]
                'initialization': 'custom',
                'initialization_params': {
                    'strength': 0.8,  # 80% success rate for actions
                    'transitions': {
                        'Decrease': [  # Action 0: Tend to decrease state
                            [0.8, 0.2, 0.0],  # From Low
                            [0.7, 0.2, 0.1],  # From Medium
                            [0.2, 0.7, 0.1]   # From High
                        ],
                        'Stay': [     # Action 1: Tend to maintain state
                            [0.8, 0.1, 0.1],  # From Low
                            [0.1, 0.8, 0.1],  # From Medium
                            [0.1, 0.1, 0.8]   # From High
                        ],
                        'Increase': [  # Action 2: Tend to increase state
                            [0.1, 0.7, 0.2],  # From Low
                            [0.1, 0.2, 0.7],  # From Medium
                            [0.0, 0.2, 0.8]   # From High
                        ]
                    }
                },
                'constraints': ['column_stochastic']
            },
            'C_matrix': {
                'shape': [3],  # [num_observations] - log preferences over observations
                'initialization': 'log_preferences',
                'initialization_params': {
                    'preferences': [0.1, 2.0, 0.1],  # Strong preference for medium observations
                    'description': 'Log-preferences: Low=0.1 (avoid), Medium=2.0 (prefer), High=0.1 (avoid)'
                }
            },
            'D_matrix': {
                'shape': [3],  # [num_states] - initial state prior
                'initialization': 'uniform',
                'description': 'Uniform prior over states'
            },
            'E_matrix': {
                'shape': [3],  # [num_actions] - initial action prior
                'initialization': 'uniform',
                'description': 'Initial uniform prior over actions (Decrease, Stay, Increase)',
                'learning_rate': 0.2  # Rate at which policy prior is updated
            }
        },
        'inference': {
            'time_horizon': 1,  # Single-step policies
            'num_iterations': 10,
            'learning_rate': 0.5,  # For belief updates
            'temperature': 1.0,    # For policy selection
            'policy_learning_rate': 0.2  # For E matrix updates
        },
        'visualization': {
            'output_dir': 'Output',
            'style': {
                'figure_size': (10, 8),
                'dpi': 100,
                'colormap': 'RdYlBu_r',  # Better for showing preferences
                'colormap_3d': 'viridis',
                'font_size': 12,
                'file_format': 'png'
            }
        }
    }

def validate_matrices(model, output_dir: Path):
    """Validate and visualize model matrices.
    
    Args:
        model: SimplePOMDP model instance
        output_dir: Output directory for visualizations
    """
    print("\n=== Matrix Validation and Visualization ===")
    
    # Create validation report
    report = []
    report.append("Matrix Validation Report")
    report.append("======================")
    
    # A Matrix (Observation Model)
    report.append("\nA Matrix (Observation Model):")
    report.append("- Shape: {}".format(model.A.shape))
    report.append("- Column stochastic: {}".format(np.allclose(model.A.sum(axis=0), 1.0)))
    report.append("- Non-negative: {}".format(np.all(model.A >= 0)))
    report.append("\nObservation probabilities:")
    for i, obs in enumerate(model.config['observation_space']['observation_labels']):
        for j, state in enumerate(model.config['state_space']['state_labels']):
            report.append(f"  P({obs}|{state}) = {model.A[i,j]:.3f}")
    
    # B Matrix (Transition Model)
    report.append("\nB Matrix (Transition Model):")
    report.append("- Shape: {}".format(model.B.shape))
    for a, action in enumerate(model.config['action_space']['action_labels']):
        report.append(f"\nAction: {action}")
        report.append("- Column stochastic: {}".format(np.allclose(model.B[:,:,a].sum(axis=0), 1.0)))
        report.append("- Non-negative: {}".format(np.all(model.B[:,:,a] >= 0)))
        report.append("\nTransition probabilities:")
        for i, next_state in enumerate(model.config['state_space']['state_labels']):
            for j, curr_state in enumerate(model.config['state_space']['state_labels']):
                report.append(f"  P({next_state}|{curr_state},{action}) = {model.B[i,j,a]:.3f}")
    
    # C Matrix (Log Preferences)
    report.append("\nC Matrix (Log Preferences over Observations):")
    report.append("- Shape: {}".format(model.C.shape))
    report.append("\nLog preference values:")
    for i, obs in enumerate(model.config['observation_space']['observation_labels']):
        report.append(f"  ln P({obs}) = {model.C[i]:.3f}")
    report.append("\nNormalized preference probabilities:")
    probs = np.exp(model.C) / np.sum(np.exp(model.C))
    for i, obs in enumerate(model.config['observation_space']['observation_labels']):
        report.append(f"  P({obs}) = {probs[i]:.3f}")
    
    # Save report
    report_file = output_dir / "matrix_validation.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    print(f"Validation report saved to: {report_file}")
    
    # Visualize matrices
    print("\nGenerating matrix visualizations...")
    
    # Plot A matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(model.A, 
                annot=True, 
                fmt='.2f',
                xticklabels=model.config['state_space']['state_labels'],
                yticklabels=model.config['observation_space']['observation_labels'],
                ax=ax)
    ax.set_title('A Matrix: Observation Model')
    ax.set_xlabel('State')
    ax.set_ylabel('Observation')
    plt.tight_layout()
    plt.savefig(output_dir / 'A_matrix.png')
    plt.close()
    
    # Plot B matrices (one for each action)
    fig, axes = plt.subplots(1, model.B.shape[2], figsize=(15, 5))
    for a, (ax, action) in enumerate(zip(axes, model.config['action_space']['action_labels'])):
        sns.heatmap(model.B[:,:,a],
                   annot=True,
                   fmt='.2f',
                   xticklabels=model.config['state_space']['state_labels'],
                   yticklabels=model.config['state_space']['state_labels'],
                   ax=ax)
        ax.set_title(f'B Matrix: {action} Action')
        ax.set_xlabel('Current State')
        ax.set_ylabel('Next State')
    plt.tight_layout()
    plt.savefig(output_dir / 'B_matrices.png')
    plt.close()
    
    # Plot C matrix (log preferences and probabilities)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Log preferences
    sns.barplot(
        x=model.config['observation_space']['observation_labels'],
        y=model.C,
        ax=ax1
    )
    ax1.set_title('C Matrix: Log Preferences')
    ax1.set_xlabel('Observation')
    ax1.set_ylabel('Log Preference')
    
    # Normalized probabilities
    sns.barplot(
        x=model.config['observation_space']['observation_labels'],
        y=probs,
        ax=ax2
    )
    ax2.set_title('Normalized Preference Probabilities')
    ax2.set_xlabel('Observation')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'C_preferences.png')
    plt.close()
    
    # Plot all matrices combined
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(3, 4, height_ratios=[1, 1, 0.8])
    
    # A matrix (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(model.A,
                annot=True,
                fmt='.2f',
                xticklabels=model.config['state_space']['state_labels'],
                yticklabels=model.config['observation_space']['observation_labels'],
                ax=ax1)
    ax1.set_title('A: Observation Model P(o|s)')
    
    # B matrices (top-right three plots)
    for a in range(model.B.shape[2]):
        ax = fig.add_subplot(gs[0, a+1])
        sns.heatmap(model.B[:,:,a],
                   annot=True,
                   fmt='.2f',
                   xticklabels=model.config['state_space']['state_labels'],
                   yticklabels=model.config['state_space']['state_labels'],
                   ax=ax)
        ax.set_title(f'B: Transition P(s\'|s,{model.config["action_space"]["action_labels"][a]})')
    
    # C matrix (middle-left)
    ax_c = fig.add_subplot(gs[1, 0:2])
    sns.barplot(
        x=model.config['observation_space']['observation_labels'],
        y=model.C,
        ax=ax_c
    )
    ax_c.set_title('C: Log Preferences ln P(o)')
    ax_c.set_xlabel('Observation')
    ax_c.set_ylabel('Log Preference')
    
    # D matrix (middle-right)
    ax_d = fig.add_subplot(gs[1, 2:])
    sns.barplot(
        x=model.config['state_space']['state_labels'],
        y=model.D,
        ax=ax_d
    )
    ax_d.set_title('D: Initial State Prior P(s₁)')
    ax_d.set_xlabel('State')
    ax_d.set_ylabel('Probability')
    
    # E matrix (bottom)
    ax_e = fig.add_subplot(gs[2, :])
    sns.barplot(
        x=model.config['action_space']['action_labels'],
        y=model.E,
        ax=ax_e
    )
    ax_e.set_title('E: Action Prior P(a)')
    ax_e.set_xlabel('Action')
    ax_e.set_ylabel('Probability')
    
    # Add overall title
    fig.suptitle('Active Inference POMDP Model Components', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_matrices.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print("Matrix visualizations saved to output directory")

def run_simulation(n_steps: int = 20):
    """Run the POMDP simulation.
    
    Args:
        n_steps: Number of steps to simulate
    """
    # Create output directory
    output_dir = Path('Output')
    output_dir.mkdir(exist_ok=True)
    
    print("\n=== Model Configuration and Initialization ===")
    config = create_config()
    model = SimplePOMDP(config)
    
    # First validate and visualize matrices
    validate_matrices(model, output_dir)
    
    print("\n=== Starting Simulation ===")
    print(f"Initial state: {config['state_space']['state_labels'][model.state.current_state]}")
    
    # Create simulation log file
    log_file = output_dir / "simulation_log.txt"
    with open(log_file, "w") as f:
        f.write("Simulation Log\n")
        f.write("==============\n\n")
        
        # Run simulation
        for step in range(n_steps):
            # Take a step and get observation and free energy
            obs, vfe = model.step()
            
            # Get current state and action
            state = model.state.current_state
            action = model.state.history['actions'][-1]
            
            # Get current expected free energies
            efe = model.state.history['expected_fe'][-1]
            
            # Format step information
            step_info = [
                f"\nStep {step + 1}:",
                f"State: {config['state_space']['state_labels'][state]}",
                f"Observation: {config['observation_space']['observation_labels'][obs]}",
                f"Action: {config['action_space']['action_labels'][action]}",
                f"Variational FE: {vfe:.3f}",
                "\nExpected Free Energies:"
            ]
            
            # Add Expected Free Energies for each action
            for a, efe_a in enumerate(efe):
                step_info.append(f"{config['action_space']['action_labels'][a]}: {efe_a:.3f}")
            
            step_info.append("\nBeliefs:")
            # Add belief distribution
            beliefs = model.state.beliefs
            for label, prob in zip(config['state_space']['state_labels'], beliefs):
                step_info.append(f"{label}: {prob:.3f}")
            
            # Write to log and print to console
            log_text = "\n".join(step_info)
            f.write(log_text + "\n")
            print(log_text)
    
    print("\n=== Generating Simulation Visualizations ===")
    
    # Plot belief evolution
    model.visualize("belief_evolution")
    
    # Plot state transitions
    model.visualize("state_transitions")
    
    # Plot observation likelihood
    model.visualize("observation_likelihood")
    
    # Plot action history
    model.visualize("action_history")
    
    # Plot belief history
    model.visualize("belief_history")
    
    # Plot Free Energies (both VFE and EFE)
    model.visualize("free_energies")
    
    # Plot policy evolution
    model.visualize("policy_evolution")
    
    # Plot detailed EFE components
    print("\nGenerating detailed EFE visualization...")
    model.visualize("efe_components_detailed")
    
    print(f"\nSimulation results and visualizations saved to: {output_dir.absolute()}")
    
    # Print list of generated files
    print("\nGenerated visualization files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"- {file.name}")

if __name__ == "__main__":
    run_simulation()