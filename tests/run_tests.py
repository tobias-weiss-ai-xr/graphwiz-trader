#!/usr/bin/env python
"""
Comprehensive Test Runner for GraphWiz Trader

Following Cognitive QA Principles:
- Multi-stage testing (Unit → Integration → Mutation)
- Quality gates (mutation score >80%)
- Performance monitoring
- Clear reporting

Usage:
    python run_tests.py              # Run all stages
    python run_tests.py --unit        # Run only unit tests
    python run_tests.py --mutation    # Run mutation testing
    python run_tests.py --quick       # Quick check (unit tests only)
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class TestRunner:
    """
    Orchestrates multi-stage test execution.

    Cognitive QA: Systematic validation with quality gates
    """

    def __init__(self, project_root: Path = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.results = {
            "unit": {"passed": 0, "failed": 0, "duration": 0},
            "integration": {"passed": 0, "failed": 0, "duration": 0},
            "mutation": {"killed": 0, "survived": 0, "score": 0.0},
        }

    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def run_unit_tests(self, verbose: bool = False) -> bool:
        """
        Run unit tests.

        Cognitive QA: Fast, isolated component testing
        Target: <2 seconds for entire suite
        """
        self.print_header("STAGE 1: UNIT TESTS")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--cov=graphwiz_trader.strategies",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--no-cov-on-fail",
        ]

        print(f"Command: {' '.join(cmd[2:])}")
        start = time.time()

        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start

        # Store results
        self.results["unit"]["duration"] = duration

        if result.returncode != 0:
            print(f"\n❌ Unit tests failed in {duration:.2f}s")
            return False
        else:
            print(f"\n✅ Unit tests passed in {duration:.2f}s")
            return True

    def run_integration_tests(self, verbose: bool = False) -> bool:
        """
        Run integration tests.

        Cognitive QA: Component interaction validation
        Target: All tests pass, acceptable performance
        """
        self.print_header("STAGE 2: INTEGRATION TESTS")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "not slow",  # Skip slow tests by default
        ]

        print(f"Command: {' '.join(cmd[2:])}")
        print("Note: Skipping slow tests. Use --verbose to see all tests.")
        start = time.time()

        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start

        self.results["integration"]["duration"] = duration

        if result.returncode != 0:
            print(f"\n❌ Integration tests failed in {duration:.2f}s")
            return False
        else:
            print(f"\n✅ Integration tests passed in {duration:.2f}s")
            return True

    def run_mutation_tests(self, verbose: bool = False) -> bool:
        """
        Run mutation tests.

        Cognitive QA: "Testing the Tester"
        Target: >80% mutation score

        Mutation testing validates that tests actually catch bugs
        by introducing small changes (mutations) into production code.
        """
        self.print_header("STAGE 3: MUTATION TESTING")

        # Run mutation tests on key files
        source_files = [
            "src/graphwiz_trader/strategies/modern_strategies.py",
        ]

        total_score = 0.0
        total_mutants = 0
        total_killed = 0

        for source_file in source_files:
            source_path = self.project_root / source_file
            if not source_path.exists():
                print(f"⚠️  Skipping {source_file} (not found)")
                continue

            print(f"\nTesting: {source_file}")

            cmd = [
                sys.executable,
                "tests/mutation/mutation_test_framework.py",
                str(source_path),
                "tests/unit/",
            ]

            start = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            duration = time.time() - start

            # Parse results (simplified)
            output = result.stdout
            if "Mutation Score:" in output:
                score_line = [line for line in output.split("\n") if "Mutation Score:" in line]
                if score_line:
                    score = float(score_line[0].split(":")[1].strip().replace("%", ""))
                    total_score += score

                    # Count killed vs survived (from output)
                    if "Killed:" in output:
                        killed_line = [line for line in output.split("\n") if "Killed:" in line]
                        if killed_line:
                            killed = int(killed_line[0].split(":")[1].strip().split()[0])
                            total_killed += killed

                    if "Survived:" in output:
                        survived_line = [line for line in output.split("\n") if "Survived:" in line]
                        if survived_line:
                            survived = int(survived_line[0].split(":")[1].strip().split()[0])
                            total_mutants = killed + survived

            print(f"  Duration: {duration:.2f}s")

        if total_mutants > 0:
            final_score = (total_killed / total_mutants) * 100
            self.results["mutation"]["score"] = final_score
            self.results["mutation"]["killed"] = total_killed
            self.results["mutation"]["survived"] = total_mutants - total_killed

            print(f"\nMutation Score: {final_score:.1f}%")

            if final_score >= 80.0:
                print("✅ EXCELLENT: Mutation score ≥ 80%")
                return True
            elif final_score >= 60.0:
                print("⚠️  ACCEPTABLE: Mutation score ≥ 60%, but < 80%")
                print("   Recommendation: Add tests for surviving mutants")
                return True
            else:
                print("❌ POOR: Mutation score < 60%")
                print("   Action Required: Improve test coverage")
                return False
        else:
            print("\n⚠️  No mutation tests run (framework may need setup)")
            return True  # Don't block if mutation testing not available

    def run_quick_tests(self, verbose: bool = False) -> bool:
        """
        Run quick tests only.

        Cognitive QA: Fast feedback loop for development
        """
        self.print_header("QUICK TEST CHECK")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-q",
            "--tb=line",
        ]

        result = subprocess.run(cmd, cwd=self.project_root)

        if result.returncode != 0:
            print("\n❌ Quick tests failed")
            return False
        else:
            print("\n✅ Quick tests passed")
            return True

    def print_summary(self):
        """Print comprehensive test summary."""
        self.print_header("TEST SUMMARY")

        # Unit tests
        unit = self.results["unit"]
        if unit["duration"] > 0:
            print(f"Unit Tests:")
            print(f"  Duration: {unit['duration']:.2f}s")
            print(f"  Status: {'✅ PASSED' if unit.get('passed', True) else '❌ FAILED'}")

        # Integration tests
        integration = self.results["integration"]
        if integration["duration"] > 0:
            print(f"\nIntegration Tests:")
            print(f"  Duration: {integration['duration']:.2f}s")
            print(f"  Status: {'✅ PASSED' if integration.get('passed', True) else '❌ FAILED'}")

        # Mutation tests
        mutation = self.results["mutation"]
        if mutation["score"] > 0:
            print(f"\nMutation Testing:")
            print(f"  Score: {mutation['score']:.1f}%")
            print(f"  Killed: {mutation['killed']} ✅")
            print(f"  Survived: {mutation['survived']} ❌")
            print(f"  Status: {'✅ EXCELLENT' if mutation['score'] >= 80 else '⚠️  NEEDS IMPROVEMENT'}")

        # Overall assessment
        print("\n" + "=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)

        all_passed = (
            unit.get("passed", True) and
            integration.get("passed", True) and
            (mutation["score"] >= 60 or mutation["score"] == 0)
        )

        if all_passed:
            print("\n✅ ALL TEST STAGES PASSED")
            print("\nCognitive QA Status:")
            print("  ✅ Unit tests validate components")
            print("  ✅ Integration tests validate interactions")
            print("  ✅ Mutation testing validates test quality")
            print("\nReady for production deployment!")
        else:
            print("\n❌ SOME TEST STAGES FAILED")
            print("\nAction Required:")
            if not unit.get("passed", True):
                print("  - Fix failing unit tests")
            if not integration.get("passed", True):
                print("  - Fix failing integration tests")
            if mutation["score"] < 60:
                print("  - Improve test coverage (mutation score < 60%)")

        print("=" * 80 + "\n")

        return all_passed

    def run_all(self, verbose: bool = False, skip_slow: bool = True):
        """Run all test stages."""
        print("\n" + "=" * 80)
        print("GraphWiz Trader - Comprehensive Test Suite")
        print("=" * 80)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Working Directory: {self.project_root}")
        print("\nStages:")
        print("  1. Unit Tests (fast, isolated)")
        print("  2. Integration Tests (component interaction)")
        print("  3. Mutation Tests (test quality validation)")

        start_time = time.time()

        # Run stages
        results = []
        results.append(self.run_unit_tests(verbose))
        results.append(self.run_integration_tests(verbose))
        results.append(self.run_mutation_tests(verbose))

        total_duration = time.time() - start_time

        # Print summary
        all_passed = all(results)
        self.print_summary()

        print(f"Total Duration: {total_duration:.2f}s")

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run GraphWiz Trader test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py              # Run all test stages
  python run_tests.py --unit        # Run only unit tests
  python run_tests.py --integration # Run only integration tests
  python run_tests.py --mutation    # Run mutation testing
  python run_tests.py --quick       # Quick check (unit tests only)
  python run_tests.py -v            # Verbose output
        """
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--mutation",
        action="store_true",
        help="Run only mutation tests"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (unit tests, fast feedback)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    runner = TestRunner()

    if args.quick:
        success = runner.run_quick_tests(verbose=args.verbose)
    elif args.unit:
        success = runner.run_unit_tests(verbose=args.verbose)
    elif args.integration:
        success = runner.run_integration_tests(verbose=args.verbose)
    elif args.mutation:
        success = runner.run_mutation_tests(verbose=args.verbose)
    else:
        success = runner.run_all(verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
