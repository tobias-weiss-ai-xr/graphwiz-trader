#!/usr/bin/env python3
"""
Comprehensive test suite runner.

Runs all tests for the GraphWiz Trader live trading system.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime


class TestRunner:
    """Test suite runner."""

    def __init__(self):
        """Initialize test runner."""
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.results = {}

    def print_header(self, title):
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)

    def print_test_file(self, name):
        """Print test file header."""
        print(f"\n{'─' * 80}")
        print(f" Running: {name}")
        print(f"{'─' * 80}")

    def run_test_file(self, test_file, description):
        """Run a single test file."""
        self.print_test_file(description)

        test_path = self.tests_dir / test_file

        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            return False

        # Activate venv and run test
        cmd = f"cd {self.project_root} && source venv/bin/activate && python3 {test_path}"

        start_time = time.time()
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True
        )
        duration = time.time() - start_time

        success = result.returncode == 0

        self.results[description] = {
            'success': success,
            'duration': duration
        }

        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status} (Duration: {duration:.2f}s)")

        return success

    def run_all_tests(self):
        """Run all test files."""
        self.print_header("GraphWiz Trader - Comprehensive Test Suite")

        print(f"\nProject Root: {self.project_root}")
        print(f"Tests Directory: {self.tests_dir}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Define all test suites
        test_suites = [
            ('test_rsi.py', 'RSI & Signal Generation Tests'),
            ('test_risk_management.py', 'Risk Management Tests'),
            ('test_order_execution.py', 'Order Execution Tests (Mock)'),
            ('test_api_connections.py', 'API Connection Tests'),
            ('test_config_validation.py', 'Configuration Validation Tests'),
        ]

        # Run each test suite
        total_tests = len(test_suites)
        passed = 0

        for test_file, description in test_suites:
            if self.run_test_file(test_file, description):
                passed += 1

        # Print summary
        self.print_summary(passed, total_tests)

        return passed == total_tests

    def print_summary(self, passed, total):
        """Print test summary."""
        self.print_header("Test Suite Summary")

        print(f"\nTotal Test Suites: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {total - passed} ❌")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        print(f"\nDetailed Results:")
        print(f"{'─' * 80}")
        print(f"{'Test Suite':<40} {'Status':<15} {'Duration':<10}")
        print(f"{'─' * 80}")

        total_duration = 0
        for description, result in self.results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            duration = f"{result['duration']:.2f}s"
            print(f"{description:<40} {status:<15} {duration:<10}")
            total_duration += result['duration']

        print(f"{'─' * 80}")
        print(f"{'Total Duration:':<51} {total_duration:.2f}s")
        print(f"{'=' * 80}")

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if passed == total:
            print("\n🎉 All test suites passed!")
        else:
            print(f"\n⚠️  {total - passed} test suite(s) failed")

    def run_quick_tests(self):
        """Run only quick unit tests (no API calls)."""
        self.print_header("Quick Unit Tests (No API Calls)")

        quick_tests = [
            ('test_rsi.py', 'RSI & Signal Generation Tests'),
            ('test_risk_management.py', 'Risk Management Tests'),
            ('test_order_execution.py', 'Order Execution Tests (Mock)'),
            ('test_config_validation.py', 'Configuration Validation Tests'),
        ]

        total = len(quick_tests)
        passed = 0

        for test_file, description in quick_tests:
            if self.run_test_file(test_file, description):
                passed += 1

        self.print_summary(passed, total)

        return passed == total

    def run_integration_tests(self):
        """Run integration tests (requires API credentials)."""
        self.print_header("Integration Tests (API Required)")

        integration_tests = [
            ('test_api_connections.py', 'API Connection Tests'),
        ]

        total = len(integration_tests)
        passed = 0

        for test_file, description in integration_tests:
            if self.run_test_file(test_file, description):
                passed += 1

        self.print_summary(passed, total)

        return passed == total


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GraphWiz Trader Test Suite Runner"
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'quick', 'integration'],
        default='all',
        help='Test mode: all (default), quick (unit tests only), or integration (API tests only)'
    )

    args = parser.parse_args()

    runner = TestRunner()

    if args.mode == 'quick':
        success = runner.run_quick_tests()
    elif args.mode == 'integration':
        success = runner.run_integration_tests()
    else:
        success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
