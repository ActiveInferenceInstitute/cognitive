"""
Benchmark suite for Generic Thing performance testing.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from ..core import GenericThing
from ..markov_blanket import MarkovBlanket
from ..free_energy import FreeEnergy
from ..message_passing import MessagePassing, Message
from .test_utils import generate_test_data, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    """Suite of performance benchmarks for Generic Thing components."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results_dir = Path(__file__).parent.parent / 'reports' / 'benchmarks'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def benchmark_thing_creation(self, n_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark thing creation performance."""
        logger.info(f"Running thing creation benchmark ({n_iterations} iterations)")
        
        times = []
        for _ in range(n_iterations):
            start = datetime.now()
            thing = GenericThing(
                id=f"thing_{_}",
                name=f"Thing {_}",
                description="Benchmark thing"
            )
            end = datetime.now()
            times.append((end - start).total_seconds())
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def benchmark_state_updates(self, n_updates: int = 1000) -> Dict[str, float]:
        """Benchmark state update performance."""
        logger.info(f"Running state update benchmark ({n_updates} updates)")
        
        thing = GenericThing(id="test", name="Test")
        times = []
        
        for _ in range(n_updates):
            observation = generate_test_data(size=10)
            start = datetime.now()
            thing.update(observation)
            end = datetime.now()
            times.append((end - start).total_seconds())
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def benchmark_message_passing(self, n_messages: int = 1000) -> Dict[str, float]:
        """Benchmark message passing performance."""
        logger.info(f"Running message passing benchmark ({n_messages} messages)")
        
        mp = MessagePassing(id="benchmark")
        times = []
        
        # Set up network
        for i in range(10):
            for j in range(i + 1, 10):
                mp.connect(f"thing_{i}", f"thing_{j}")
        
        # Measure message propagation and processing time
        for _ in range(n_messages):
            message = Message(
                source_id="thing_0",
                target_id="broadcast",
                content=generate_test_data(size=5),
                message_type="benchmark",
                timestamp=np.datetime64('now')
            )
            
            start = datetime.now()
            mp.send_message(message)
            mp.process_outgoing()
            received = mp.receive("thing_1")  # Test receiving messages
            end = datetime.now()
            
            times.append((end - start).total_seconds())
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def benchmark_free_energy(self, n_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark free energy computation performance."""
        logger.info(f"Running free energy benchmark ({n_iterations} iterations)")
        
        fe = FreeEnergy()
        times = []
        
        for _ in range(n_iterations):
            internal = generate_test_data(size=5)
            external = generate_test_data(size=5)
            observation = generate_test_data(size=5)
            
            start = datetime.now()
            fe.compute_free_energy(internal, external, observation)
            end = datetime.now()
            times.append((end - start).total_seconds())
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def plot_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Generate performance visualization."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Mean execution times
        plt.subplot(2, 2, 1)
        means = [r['mean'] for r in results.values()]
        plt.bar(results.keys(), means)
        plt.title('Mean Execution Time by Component')
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')
        
        # Plot 2: Box plot of distributions
        plt.subplot(2, 2, 2)
        data = []
        labels = []
        for name, metrics in results.items():
            data.append([metrics[m] for m in ['min', 'median', 'max']])
            labels.extend([name] * 3)
        plt.boxplot(data, labels=results.keys())
        plt.title('Performance Distribution by Component')
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')
        
        # Plot 3: Standard deviations
        plt.subplot(2, 2, 3)
        stds = [r['std'] for r in results.values()]
        plt.bar(results.keys(), stds)
        plt.title('Performance Variability by Component')
        plt.xticks(rotation=45)
        plt.ylabel('Standard Deviation (seconds)')
        
        # Plot 4: Min/Max ranges
        plt.subplot(2, 2, 4)
        for i, (name, metrics) in enumerate(results.items()):
            plt.plot([i, i], [metrics['min'], metrics['max']], 'b-')
            plt.plot(i, metrics['median'], 'ro')
        plt.xticks(range(len(results)), results.keys(), rotation=45)
        plt.title('Performance Range by Component')
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'benchmark_results_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save benchmark results to file."""
        output_file = self.results_dir / f'benchmark_results_{self.timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results to {output_file}")
    
    def run_all(self) -> None:
        """Run all benchmarks and generate reports."""
        results = {
            'thing_creation': self.benchmark_thing_creation(),
            'state_updates': self.benchmark_state_updates(),
            'message_passing': self.benchmark_message_passing(),
            'free_energy': self.benchmark_free_energy()
        }
        
        self.plot_results(results)
        self.save_results(results)
        
        # Log summary
        logger.info("\nBenchmark Summary:")
        for component, metrics in results.items():
            logger.info(f"\n{component}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.6f}s")

def main():
    """Main benchmark execution function."""
    try:
        suite = BenchmarkSuite()
        suite.run_all()
        return 0
    except Exception as e:
        logger.error("Benchmark suite failed", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 