"""
Comprehensive test runner for Generic POMDP implementation.
"""

import pytest
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from generic_pomdp import GenericPOMDP
from visualization import POMDPVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Comprehensive test runner with detailed logging."""
    
    def __init__(self):
        """Initialize test runner."""
        self.output_dir = Path("Output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'metrics': {},
            'summary': {}
        }
        
    def run_all_tests(self):
        """Run all tests with comprehensive logging."""
        logger.info("Starting comprehensive test suite...")
        
        try:
            # Run initialization tests
            self._run_test_group("initialization", [
                self._test_model_initialization,
                self._test_matrix_properties,
                self._test_state_initialization
            ])
            
            # Run dynamics tests
            self._run_test_group("dynamics", [
                self._test_step_without_action,
                self._test_step_with_action,
                self._test_belief_updating,
                self._test_action_selection
            ])
            
            # Run learning tests
            self._run_test_group("learning", [
                self._test_learning_dynamics,
                self._test_preference_influence,
                self._test_convergence
            ])
            
            # Run numerical stability tests
            self._run_test_group("numerical_stability", [
                self._test_numerical_stability,
                self._test_edge_cases
            ])
            
            # Run persistence tests
            self._run_test_group("persistence", [
                self._test_save_load_state
            ])
            
            # Run visualization tests
            self._run_test_group("visualization", [
                self._test_visualization_outputs
            ])
            
            # Compute summary metrics
            self._compute_summary()
            
            # Save results
            self._save_results()
            
            logger.info("Test suite completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Test suite failed: {str(e)}", exc_info=True)
            return False
    
    def _run_test_group(self, group_name: str, test_functions: List[callable]):
        """Run a group of related tests.
        
        Args:
            group_name: Name of the test group
            test_functions: List of test functions to run
        """
        logger.info(f"Running {group_name} tests...")
        
        self.results['tests'][group_name] = {
            'passed': 0,
            'failed': 0,
            'details': {}
        }
        
        for test_func in test_functions:
            test_name = test_func.__name__
            logger.info(f"Running test: {test_name}")
            
            try:
                metrics = test_func()
                self.results['tests'][group_name]['passed'] += 1
                self.results['tests'][group_name]['details'][test_name] = {
                    'status': 'PASS',
                    'metrics': metrics
                }
                logger.info(f"Test {test_name} PASSED")
                
            except Exception as e:
                self.results['tests'][group_name]['failed'] += 1
                self.results['tests'][group_name]['details'][test_name] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
                logger.error(f"Test {test_name} FAILED: {str(e)}", exc_info=True)
    
    def _test_model_initialization(self) -> Dict:
        """Test model initialization."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        metrics = {
            'matrix_shapes': {
                'A': model.A.shape == (2, 3),
                'B': model.B.shape == (3, 3, 2),
                'C': model.C.shape == (2, 5),
                'D': model.D.shape == (3,),
                'E': model.E.shape == (2,)
            },
            'initial_state': {
                'beliefs_shape': model.state.beliefs.shape == (3,),
                'time_step': model.state.time_step == 0
            }
        }
        
        assert all(metrics['matrix_shapes'].values()), "Invalid matrix shapes"
        assert all(metrics['initial_state'].values()), "Invalid initial state"
        
        return metrics
    
    def _test_matrix_properties(self) -> Dict:
        """Test matrix properties."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        metrics = {
            'A_matrix': {
                'column_stochastic': np.allclose(model.A.sum(axis=0), 1.0),
                'non_negative': np.all(model.A >= 0)
            },
            'B_matrix': {
                'action_0': {
                    'column_stochastic': np.allclose(model.B[:,:,0].sum(axis=0), 1.0),
                    'non_negative': np.all(model.B[:,:,0] >= 0)
                },
                'action_1': {
                    'column_stochastic': np.allclose(model.B[:,:,1].sum(axis=0), 1.0),
                    'non_negative': np.all(model.B[:,:,1] >= 0)
                }
            },
            'C_matrix': {
                'finite': np.all(np.isfinite(model.C))
            },
            'D_matrix': {
                'normalized': np.allclose(model.D.sum(), 1.0),
                'non_negative': np.all(model.D >= 0)
            },
            'E_matrix': {
                'normalized': np.allclose(model.E.sum(), 1.0),
                'non_negative': np.all(model.E >= 0)
            }
        }
        
        # Verify all properties
        assert all(metrics['A_matrix'].values()), "Invalid A matrix properties"
        assert all(metrics['B_matrix']['action_0'].values()) and \
               all(metrics['B_matrix']['action_1'].values()), "Invalid B matrix properties"
        assert all(metrics['C_matrix'].values()), "Invalid C matrix properties"
        assert all(metrics['D_matrix'].values()), "Invalid D matrix properties"
        assert all(metrics['E_matrix'].values()), "Invalid E matrix properties"
        
        return metrics
    
    def _test_state_initialization(self) -> Dict:
        """Test state initialization."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        metrics = {
            'beliefs': {
                'normalized': np.allclose(model.state.beliefs.sum(), 1.0),
                'non_negative': np.all(model.state.beliefs >= 0)
            },
            'history': {
                'observations_empty': len(model.state.history['observations']) == 0,
                'actions_empty': len(model.state.history['actions']) == 0,
                'beliefs_initial': len(model.state.history['beliefs']) == 1
            }
        }
        
        assert all(metrics['beliefs'].values()), "Invalid initial beliefs"
        assert all(metrics['history'].values()), "Invalid history initialization"
        
        return metrics
    
    def _test_step_without_action(self) -> Dict:
        """Test stepping without providing action."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        observation, free_energy = model.step()
        
        metrics = {
            'observation': {
                'valid_range': 0 <= observation < model.num_observations
            },
            'free_energy': {
                'finite': np.isfinite(free_energy)
            },
            'history': {
                'observations_updated': len(model.state.history['observations']) == 1,
                'actions_updated': len(model.state.history['actions']) == 1,
                'beliefs_updated': len(model.state.history['beliefs']) == 2
            }
        }
        
        assert all(metrics['observation'].values()), "Invalid observation"
        assert all(metrics['free_energy'].values()), "Invalid free energy"
        assert all(metrics['history'].values()), "Invalid history update"
        
        return metrics
    
    def _test_step_with_action(self) -> Dict:
        """Test stepping with provided action."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        action = 0
        observation, free_energy = model.step(action)
        
        metrics = {
            'observation': {
                'valid_range': 0 <= observation < model.num_observations
            },
            'free_energy': {
                'finite': np.isfinite(free_energy)
            },
            'action': {
                'correct_action': model.state.history['actions'][-1] == action
            }
        }
        
        assert all(metrics['observation'].values()), "Invalid observation"
        assert all(metrics['free_energy'].values()), "Invalid free energy"
        assert all(metrics['action'].values()), "Invalid action"
        
        return metrics
    
    def _test_belief_updating(self) -> Dict:
        """Test belief updating mechanism."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        initial_beliefs = model.state.beliefs.copy()
        model.step()
        
        metrics = {
            'belief_change': {
                'updated': not np.allclose(model.state.beliefs, initial_beliefs),
                'normalized': np.allclose(model.state.beliefs.sum(), 1.0),
                'non_negative': np.all(model.state.beliefs >= 0)
            }
        }
        
        assert all(metrics['belief_change'].values()), "Invalid belief update"
        
        return metrics
    
    def _test_action_selection(self) -> Dict:
        """Test action selection mechanism."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        action, probs = model._select_action()
        
        metrics = {
            'action': {
                'valid_range': 0 <= action < model.num_actions
            },
            'probabilities': {
                'normalized': np.allclose(probs.sum(), 1.0),
                'non_negative': np.all(probs >= 0),
                'correct_shape': len(probs) == model.num_actions
            }
        }
        
        assert all(metrics['action'].values()), "Invalid action"
        assert all(metrics['probabilities'].values()), "Invalid action probabilities"
        
        return metrics
    
    def _test_learning_dynamics(self) -> Dict:
        """Test learning dynamics over multiple steps."""
        model = GenericPOMDP(num_observations=4, num_states=5, num_actions=3)
        
        # Track belief entropy
        entropies = []
        n_steps = 20
        
        for _ in range(n_steps):
            model.step()
            entropy = -np.sum(model.state.beliefs *
                            np.log(model.state.beliefs +
                                  model.stability_threshold))
            entropies.append(entropy)
        
        metrics = {
            'entropy': {
                'initial_mean': float(np.mean(entropies[:5])),
                'final_mean': float(np.mean(entropies[-5:])),
                'decreased': np.mean(entropies[:5]) > np.mean(entropies[-5:])
            }
        }
        
        assert metrics['entropy']['decreased'], "No learning progress observed"
        
        return metrics
    
    def _test_preference_influence(self) -> Dict:
        """Test influence of preferences on action selection."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        # Set strong preference for first observation
        model.C = np.zeros((model.num_observations, model.time_horizon))
        model.C[0,:] = 2.0  # Strong preference for first observation
        model.C[1,:] = -2.0  # Avoid second observation
        
        # Set deterministic observation model
        model.A = np.zeros((model.num_observations, model.num_states))
        model.A[0,0] = 1.0  # First state -> first observation
        model.A[1,1] = 1.0  # Second state -> second observation
        model.A[:,2] = 0.5  # Third state -> random observation
        
        # Set controllable transition model
        model.B[:,:,0] = np.eye(model.num_states)  # Action 0: stay in place
        model.B[:,:,1] = np.roll(np.eye(model.num_states), 1, axis=1)  # Action 1: cycle states
        
        observations = []
        n_steps = 50  # More steps to ensure preference influence
        
        for _ in range(n_steps):
            obs, _ = model.step()
            observations.append(obs)
        
        obs_counts = np.bincount(observations)
        
        metrics = {
            'observations': {
                'preferred_count': int(obs_counts[0]),
                'other_count': int(obs_counts[1]),
                'preference_followed': obs_counts[0] > obs_counts[1]
            }
        }
        
        assert metrics['observations']['preference_followed'], \
            "Preferences not influencing behavior"
        
        return metrics
    
    def _test_numerical_stability(self) -> Dict:
        """Test numerical stability of computations."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        # Set very small beliefs
        model.state.beliefs = np.ones(model.num_states) * 1e-10
        model.state.beliefs /= model.state.beliefs.sum()
        
        observation, free_energy = model.step()
        
        metrics = {
            'free_energy': {
                'finite': np.isfinite(free_energy)
            },
            'beliefs': {
                'finite': np.all(np.isfinite(model.state.beliefs)),
                'normalized': np.allclose(model.state.beliefs.sum(), 1.0),
                'non_negative': np.all(model.state.beliefs >= 0)
            }
        }
        
        assert all(metrics['free_energy'].values()), "Unstable free energy"
        assert all(metrics['beliefs'].values()), "Unstable beliefs"
        
        return metrics
    
    def _test_convergence(self) -> Dict:
        """Test convergence properties."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        # Set deterministic transition and observation
        model.B[:,:,0] = np.eye(model.num_states)  # Identity transitions
        model.A = np.zeros((model.num_observations, model.num_states))
        model.A[0,0] = 1.0  # First state -> first observation
        model.A[1,1] = 1.0  # Second state -> second observation
        
        beliefs_history = []
        n_steps = 10
        
        for _ in range(n_steps):
            model.step(action=0)
            beliefs_history.append(model.state.beliefs.copy())
        
        # Compute belief changes
        diffs = [np.linalg.norm(b1 - b2)
                for b1, b2 in zip(beliefs_history[:-1], beliefs_history[1:])]
        
        metrics = {
            'convergence': {
                'initial_changes': float(np.mean(diffs[:3])),
                'final_changes': float(np.mean(diffs[-3:])),
                'converging': np.mean(diffs[:3]) > np.mean(diffs[-3:])
            }
        }
        
        assert metrics['convergence']['converging'], "No convergence observed"
        
        return metrics
    
    def _test_save_load_state(self) -> Dict:
        """Test state saving and loading."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        
        # Take some steps
        for _ in range(3):
            model.step()
        
        # Save state
        save_path = self.output_dir / "test_state.yaml"
        model.save_state(save_path)
        
        # Create new model and load state
        new_model = GenericPOMDP(
            num_observations=2,
            num_states=3,
            num_actions=2
        )
        new_model.load_state(save_path)
        
        metrics = {
            'state_match': {
                'beliefs': np.allclose(new_model.state.beliefs,
                                    model.state.beliefs),
                'time_step': new_model.state.time_step == model.state.time_step
            },
            'history_match': {
                'observations': len(new_model.state.history['observations']) ==
                              len(model.state.history['observations']),
                'actions': len(new_model.state.history['actions']) ==
                          len(model.state.history['actions']),
                'beliefs': len(new_model.state.history['beliefs']) ==
                          len(model.state.history['beliefs'])
            }
        }
        
        assert all(metrics['state_match'].values()), "State mismatch after loading"
        assert all(metrics['history_match'].values()), "History mismatch after loading"
        
        return metrics
    
    def _test_visualization_outputs(self) -> Dict:
        """Test visualization outputs."""
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        visualizer = POMDPVisualizer()
        
        # Take some steps
        for _ in range(5):
            model.step()
        
        # Generate visualizations
        visualizer.plot_belief_evolution(model.state.history['beliefs'])
        visualizer.plot_observation_matrix(model.A)
        visualizer.plot_preferences(model.C)
        
        metrics = {
            'files_generated': {
                'belief_evolution': (self.output_dir / "belief_evolution.png").exists(),
                'observation_matrix': (self.output_dir / "observation_matrix.png").exists(),
                'preferences': (self.output_dir / "preferences.png").exists()
            }
        }
        
        assert all(metrics['files_generated'].values()), "Missing visualization outputs"
        
        return metrics
    
    def _test_edge_cases(self) -> Dict:
        """Test edge cases and error handling."""
        metrics = {
            'invalid_dimensions': {
                'caught': False
            },
            'invalid_action': {
                'caught': False
            }
        }
        
        # Test invalid dimensions
        try:
            GenericPOMDP(num_observations=0, num_states=3, num_actions=2)
        except ValueError:
            metrics['invalid_dimensions']['caught'] = True
        
        # Test invalid action
        model = GenericPOMDP(num_observations=2, num_states=3, num_actions=2)
        try:
            model.step(action=model.num_actions)  # Invalid action
        except IndexError:
            metrics['invalid_action']['caught'] = True
        
        assert all(x['caught'] for x in metrics.values()), "Edge cases not handled"
        
        return metrics
    
    def _compute_summary(self):
        """Compute summary metrics for all tests."""
        total_tests = 0
        passed_tests = 0
        
        for group in self.results['tests'].values():
            total_tests += group['passed'] + group['failed']
            passed_tests += group['passed']
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': float(passed_tests) / total_tests if total_tests > 0 else 0.0
        }
        
        logger.info("Test Summary:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests}")
        logger.info(f"  Failed: {total_tests - passed_tests}")
        logger.info(f"  Pass Rate: {self.results['summary']['pass_rate']:.2%}")
    
    def _save_results(self):
        """Save test results to file."""
        results_file = self.output_dir / "test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Test results saved to {results_file}")

def main():
    """Main entry point."""
    runner = TestRunner()
    success = runner.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    main() 