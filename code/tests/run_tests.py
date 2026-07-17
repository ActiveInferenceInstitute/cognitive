#!/usr/bin/env python3
"""
Test runner with detailed reporting.
"""

import io
import json
import sys
import time
from pathlib import Path

import pytest


class TestRunner:
    """Manages test execution and reporting."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results: dict = {}
        self.start_time = 0.0
        self.end_time = 0.0

    def run_tests(self) -> bool:
        """Run all tests and collect results."""
        print("\n=== Starting Test Suite ===\n")

        self.start_time = time.time()

        # Run pytest with detailed output
        exit_code = pytest.main(
            [
                str(self.test_dir),
                "-v",
                "--tb=short",
                "--cov=src",
                "--cov-report=term-missing",
                f"--junit-xml={self.test_dir / 'test-results.xml'}",
            ]
        )

        self.end_time = time.time()

        return exit_code == 0

    def generate_report(self, success: bool) -> dict:
        """Generate detailed test report."""
        duration = self.end_time - self.start_time

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{duration:.2f}s",
            "test_files": self._collect_test_files(),
            "coverage": self._parse_coverage(),
            "status": "PASSED" if success else "FAILED",
        }

        return report

    def _collect_test_files(self) -> list[str]:
        """Collect all test files."""
        return [str(f.relative_to(self.test_dir)) for f in sorted(self.test_dir.rglob("test_*.py"))]

    def _parse_coverage(self) -> dict:
        """Parse coverage data if available."""
        try:
            from coverage import Coverage

            coverage_path = Path(".coverage")
            if not coverage_path.exists():
                return {}
            coverage = Coverage(data_file=str(coverage_path))
            coverage.load()
            stream = io.StringIO()
            total = coverage.report(file=stream, show_missing=False)
            return {
                "total_percent": round(total, 2),
                "summary": stream.getvalue().strip().splitlines()[-1],
            }
        except Exception:
            return {}

    def save_report(self, report: dict, output_file: Path):
        """Save test report to file."""
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nTest report saved to: {output_file}")


def main():
    """Main entry point."""
    test_dir = Path(__file__).parent

    runner = TestRunner(test_dir)
    success = runner.run_tests()

    report = runner.generate_report(success)
    runner.save_report(report, test_dir / "test_report.json")

    print("\n=== Test Summary ===")
    print(f"Status: {report['status']}")
    print(f"Duration: {report['duration']}")
    print(f"Test Files: {len(report['test_files'])}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
