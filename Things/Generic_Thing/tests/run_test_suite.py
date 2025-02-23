"""
Comprehensive test runner for Generic Thing test suite.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json
import webbrowser
from typing import Dict, Any, List
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Test runner with comprehensive reporting."""
    
    def __init__(self):
        """Initialize test runner."""
        self.base_dir = Path(__file__).parent.parent
        self.reports_dir = self.base_dir / 'reports'
        self.visualizations_dir = self.base_dir / 'visualizations'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create report directories
        for subdir in ['test_results', 'coverage', 'benchmarks', 'visualizations']:
            (self.reports_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Create visualizations directory if it doesn't exist
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
            
    def run_visualization_tests(self) -> bool:
        """Run visualization tests first and log their locations."""
        logger.info("\n=== Running Visualization Tests ===")
        
        result = subprocess.run([
            'pytest',
            '--verbose',
            '-m', 'visual',
            '--html', str(self.reports_dir / 'test_results' / f'visual_tests_{self.timestamp}.html'),
            '--self-contained-html'
        ], capture_output=True, text=True)
        
        # Log visualization file locations
        logger.info("\nGenerated Visualizations:")
        for ext in ['*.png', '*.pdf', '*.html', '*.svg']:
            for file in glob.glob(str(self.visualizations_dir / '**' / ext), recursive=True):
                logger.info(f"- {file}")
                
        logger.info("\nVisualization test output:")
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
            
        return result.returncode == 0

    def run_unit_tests(self) -> bool:
        """Run unit tests with pytest."""
        logger.info("\n=== Running Unit Tests ===")
        
        result = subprocess.run([
            'pytest',
            '--verbose',
            '-m', 'not (slow or integration or visual)',
            '--html', str(self.reports_dir / 'test_results' / f'unit_tests_{self.timestamp}.html'),
            '--self-contained-html',
            '--cov', str(self.base_dir),
            '--cov-report', f'html:{self.reports_dir}/coverage/unit',
            '--cov-report', 'term-missing'
        ], capture_output=True, text=True)
        
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        
        return result.returncode == 0
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        result = subprocess.run([
            'pytest',
            '--verbose',
            '-m', 'integration',
            '--html', str(self.reports_dir / 'test_results' / f'integration_tests_{self.timestamp}.html'),
            '--self-contained-html',
            '--cov', str(self.base_dir),
            '--cov-report', f'html:{self.reports_dir}/coverage/integration',
            '--cov-report', 'term-missing'
        ], capture_output=True, text=True)
        
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        
        return result.returncode == 0
    
    def run_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            from .run_benchmarks import main as run_benchmarks
            return run_benchmarks() == 0
        except Exception as e:
            logger.error("Benchmark execution failed", exc_info=True)
            return False
    
    def run_type_checks(self) -> bool:
        """Run static type checking with mypy."""
        logger.info("Running type checks...")
        
        result = subprocess.run([
            'mypy',
            str(self.base_dir),
            '--html-report', str(self.reports_dir / 'type_checking')
        ], capture_output=True, text=True)
        
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        
        return result.returncode == 0
    
    def run_code_analysis(self) -> bool:
        """Run code analysis with pylint."""
        logger.info("Running code analysis...")
        
        result = subprocess.run([
            'pylint',
            str(self.base_dir),
            '--output-format=json'
        ], capture_output=True, text=True)
        
        try:
            analysis_results = json.loads(result.stdout)
            with open(self.reports_dir / f'code_analysis_{self.timestamp}.json', 'w') as f:
                json.dump(analysis_results, f, indent=2)
        except Exception as e:
            logger.error("Failed to save code analysis results", exc_info=True)
        
        return result.returncode == 0
    
    def generate_summary_report(self, results: Dict[str, bool]) -> None:
        """Generate HTML summary report."""
        report_file = self.reports_dir / f'summary_{self.timestamp}.html'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Suite Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>Test Suite Summary Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Test Results</h2>
                <ul>
        """
        
        for test_type, passed in results.items():
            status_class = 'success' if passed else 'failure'
            status_text = 'PASSED' if passed else 'FAILED'
            html_content += f"""
                    <li>
                        <strong>{test_type}:</strong> 
                        <span class="{status_class}">{status_text}</span>
                    </li>
            """
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Reports</h2>
                <ul>
                    <li><a href="test_results/unit_tests_{timestamp}.html">Unit Test Results</a></li>
                    <li><a href="test_results/integration_tests_{timestamp}.html">Integration Test Results</a></li>
                    <li><a href="coverage/unit/index.html">Unit Test Coverage</a></li>
                    <li><a href="coverage/integration/index.html">Integration Test Coverage</a></li>
                    <li><a href="benchmarks/benchmark_results_{timestamp}.html">Performance Benchmarks</a></li>
                    <li><a href="type_checking/index.html">Type Check Results</a></li>
                    <li><a href="code_analysis_{timestamp}.json">Code Analysis Results</a></li>
                </ul>
            </div>
        </body>
        </html>
        """.format(timestamp=self.timestamp)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated summary report: {report_file}")
        webbrowser.open(str(report_file))
    
    def run_all(self) -> int:
        """Run all tests and generate reports."""
        try:
            # Run visualization tests first
            vis_result = self.run_visualization_tests()
            logger.info("\n=== Visualization Test Results ===")
            logger.info("Visualization tests %s", "PASSED" if vis_result else "FAILED")
            
            # Run remaining tests
            results = {
                'Visualization Tests': vis_result,
                'Unit Tests': self.run_unit_tests(),
                'Integration Tests': self.run_integration_tests(),
                'Performance Benchmarks': self.run_benchmarks(),
                'Type Checks': self.run_type_checks(),
                'Code Analysis': self.run_code_analysis()
            }
            
            self.generate_summary_report(results)
            
            if all(results.values()):
                logger.info("\n=== All tests passed successfully! ===")
                return 0
            else:
                logger.error("\n=== Some tests failed. Check the summary report for details. ===")
                return 1
                
        except Exception as e:
            logger.error("Test suite execution failed", exc_info=True)
            return 1

def main():
    """Main test suite execution function."""
    runner = TestRunner()
    return runner.run_all()

if __name__ == '__main__':
    sys.exit(main()) 