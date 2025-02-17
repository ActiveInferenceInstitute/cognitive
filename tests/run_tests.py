#!/usr/bin/env python3
"""
Test runner with detailed reporting.
"""

import pytest
import sys
from pathlib import Path
import time
import json
from typing import Dict, List

class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results: Dict = {}
        self.start_time = 0.0
        self.end_time = 0.0
    
    def run_tests(self) -> bool:
        """Run all tests and collect results."""
        print("\n=== Starting Test Suite ===\n")
        
        self.start_time = time.time()
        
        # Run pytest with detailed output
        exit_code = pytest.main([
            str(self.test_dir),
            '-v',
            '--tb=short',
            '--cov=src',
            '--cov-report=term-missing',
            '--junit-xml=test-results.xml'
        ])
        
        self.end_time = time.time()
        
        return exit_code == 0
    
    def generate_report(self) -> Dict:
        """Generate detailed test report."""
        duration = self.end_time - self.start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': f"{duration:.2f}s",
            'test_files': self._collect_test_files(),
            'coverage': self._parse_coverage(),
            'status': 'PASSED' if self.run_tests() else 'FAILED'
        }
        
        return report
    
    def _collect_test_files(self) -> List[str]:
        """Collect all test files."""
        return [
            str(f.relative_to(self.test_dir))
            for f in self.test_dir.glob('test_*.py')
        ]
    
    def _parse_coverage(self) -> Dict:
        """Parse coverage data if available."""
        try:
            with open('.coverage', 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_report(self, report: Dict, output_file: Path):
        """Save test report to file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nTest report saved to: {output_file}")

def main():
    """Main entry point."""
    test_dir = Path(__file__).parent
    
    runner = TestRunner(test_dir)
    success = runner.run_tests()
    
    report = runner.generate_report()
    runner.save_report(report, test_dir / 'test_report.json')
    
    print("\n=== Test Summary ===")
    print(f"Status: {report['status']}")
    print(f"Duration: {report['duration']}")
    print(f"Test Files: {len(report['test_files'])}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 