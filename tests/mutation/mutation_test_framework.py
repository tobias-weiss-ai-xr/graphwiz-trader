"""
Mutation Testing Framework for GraphWiz Trader

Following Cognitive QA Principles:
1. Mutation Score: Tests should fail when code mutates
2. Kill Mutants: Each test should "kill" specific mutants
3. Coverage: High line coverage ≠ high mutation score
4. Validation: "Testing the Tester"

Mutation Categories:
- Arithmetic: Replace +, -, *, / with alternatives
- Logical: Replace &&, ||, ! with alternatives
- Conditional: Replace >, <, >=, <= with alternatives
- Boundary: Replace array indices, loop conditions

Target: Achieve >80% mutation score (vs >95% line coverage)
"""

import ast
import copy
import operator
from pathlib import Path
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import pytest


@dataclass
class Mutant:
    """Represents a single code mutation."""
    id: str
    original_code: str
    mutated_code: str
    location: str
    description: str
    killed: bool = False
    tests_run: List[str] = None

    def __post_init__(self):
        if self.tests_run is None:
            self.tests_run = []


class MutationTester:
    """
    Mutation testing framework for Python code.

    Core concept: Introduce small changes (mutations) into production code
    and verify that tests fail. If tests pass with mutation, mutant "survived"
    (bad - tests missed the bug). If tests fail, mutant "killed" (good - tests
    detected the bug).

    Cognitive QA: This is "Testing the Tester" - validating test quality.
    """

    def __init__(self, source_file: Path):
        """
        Initialize mutation tester.

        Args:
            source_file: Path to Python source file to mutate
        """
        self.source_file = source_file
        self.mutants: List[Mutant] = []
        self.original_code = source_file.read_text()

    def generate_mutants(self) -> List[Mutant]:
        """
        Generate all possible mutants for the source code.

        Returns:
            List of Mutant objects

        Mutation Types:
        1. Arithmetic mutations: + -> -, * -> /, etc.
        2. Comparison mutations: > -> <, >= -> <=, etc.
        3. Logical mutations: and -> or, not -> (removed)
        4. Constant mutations: 0 -> 1, 1 -> 0, etc.
        """
        tree = ast.parse(self.original_code)
        mutator = Mutator()
        mutator.visit(tree)

        self.mutants = mutator.mutants
        return self.mutants

    def run_mutation_tests(self, test_command: List[str]) -> Dict[str, Any]:
        """
        Run tests against each mutant.

        Args:
            test_command: Command to run tests (e.g., ["pytest", "tests/"])

        Returns:
            Summary with mutation score

        Process:
        1. For each mutant:
           a. Write mutated code to file
           b. Run tests
           c. Record if tests pass (mutant survived) or fail (mutant killed)
        """
        import subprocess
        import tempfile

        results = {
            "total_mutants": len(self.mutants),
            "killed": 0,
            "survived": 0,
            "errors": 0,
            "mutation_score": 0.0,
            "mutants": [],
        }

        for mutant in self.mutants:
            # Write mutant to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Create temporary source file
                temp_file = Path(f.name)
                f.write(mutant.mutated_code)

            try:
                # Run tests against mutant
                result = subprocess.run(
                    test_command,
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 second timeout per mutant
                )

                # Check if tests failed (mutant killed)
                if result.returncode != 0:
                    mutant.killed = True
                    results["killed"] += 1
                else:
                    mutant.killed = False
                    results["survived"] += 1

            except subprocess.TimeoutExpired:
                results["errors"] += 1
            except Exception as e:
                results["errors"] += 1
            finally:
                # Clean up temporary file
                try:
                    temp_file.unlink()
                except:
                    pass

            results["mutants"].append(mutant)

        # Calculate mutation score
        if results["total_mutants"] > 0:
            results["mutation_score"] = (
                results["killed"] / results["total_mutants"] * 100
            )

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable mutation test report.

        Args:
            results: Results from run_mutation_tests

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("MUTATION TESTING REPORT")
        report.append("=" * 80)
        report.append(f"\nSource File: {self.source_file}")
        report.append(f"Total Mutants: {results['total_mutants']}")
        report.append(f"Killed: {results['killed']} ✅")
        report.append(f"Survived: {results['survived']} ❌")
        report.append(f"Errors: {results['errors']} ⚠️")
        report.append(f"\nMutation Score: {results['mutation_score']:.1f}%")
        report.append("")

        # Target score
        if results['mutation_score'] >= 80:
            report.append("✅ EXCELLENT: Mutation score ≥ 80%")
        elif results['mutation_score'] >= 60:
            report.append("⚠️  GOOD: Mutation score ≥ 60%, but < 80%")
        else:
            report.append("❌ NEEDS IMPROVEMENT: Mutation score < 60%")

        report.append("\n" + "=" * 80)
        report.append("SURVIVING MUTANTS (Need Better Tests)")
        report.append("=" * 80)

        survived = [m for m in results['mutants'] if not m.killed]
        for mutant in survived:
            report.append(f"\n{mutant.id}: {mutant.description}")
            report.append(f"  Location: {mutant.location}")
            report.append(f"  Original: {mutant.original_code[:50]}...")
            report.append(f"  Mutated: {mutant.mutated_code[:50]}...")

        return "\n".join(report)


class Mutator(ast.NodeTransformer):
    """
    AST visitor that generates code mutations.

    Implements specific mutation operators targeting common bug patterns.
    """

    def __init__(self):
        super().__init__()
        self.mutants: List[Mutant] = []
        self.mutation_id = 0

    def _create_mutant(
        self,
        original_node: ast.AST,
        mutated_node: ast.AST,
        description: str,
        location: str,
    ) -> ast.AST:
        """Create a mutant and return the mutated node."""
        self.mutation_id += 1

        mutant = Mutant(
            id=f"mut_{self.mutation_id}",
            original_code=ast.unparse(original_node),
            mutated_code=ast.unparse(mutated_node),
            location=location,
            description=description,
        )

        self.mutants.append(mutant)
        return mutated_node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """
        Generate arithmetic and comparison mutations.

        Examples:
        - a + b -> a - b
        - a > b -> a < b
        - a * b -> a / b
        """
        # Arithmetic mutations
        if isinstance(node.op, ast.Add):
            # Replace + with -
            mutated = copy.copy(node)
            mutated.op = ast.Sub()
            self._create_mutant(
                node,
                mutated,
                "Arithmetic: Replaced + with -",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.Sub):
            # Replace - with +
            mutated = copy.copy(node)
            mutated.op = ast.Add()
            self._create_mutant(
                node,
                mutated,
                "Arithmetic: Replaced - with +",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.Mult):
            # Replace * with /
            mutated = copy.copy(node)
            mutated.op = ast.Div()
            self._create_mutant(
                node,
                mutated,
                "Arithmetic: Replaced * with /",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.Div):
            # Replace / with *
            mutated = copy.copy(node)
            mutated.op = ast.Mult()
            self._create_mutant(
                node,
                mutated,
                "Arithmetic: Replaced / with *",
                f"Line {node.lineno}",
            )

        # Comparison mutations
        elif isinstance(node.op, ast.Gt):
            # Replace > with >=
            mutated = copy.copy(node)
            mutated.op = ast.GtE()
            self._create_mutant(
                node,
                mutated,
                "Comparison: Replaced > with >=",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.Lt):
            # Replace < with <=
            mutated = copy.copy(node)
            mutated.op = ast.LtE()
            self._create_mutant(
                node,
                mutated,
                "Comparison: Replaced < with <=",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.GtE):
            # Replace >= with >
            mutated = copy.copy(node)
            mutated.op = ast.Gt()
            self._create_mutant(
                node,
                mutated,
                "Comparison: Replaced >= with >",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.LtE):
            # Replace <= with <
            mutated = copy.copy(node)
            mutated.op = ast.Lt()
            self._create_mutant(
                node,
                mutated,
                "Comparison: Replaced <= with <",
                f"Line {node.lineno}",
            )

        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """
        Generate comparison mutations.

        Examples:
        - a == b -> a != b
        - a != b -> a == b
        """
        # Note: BinOp also handles some comparisons, this handles others
        for i, op in enumerate(node.ops):
            if isinstance(op, ast.Eq):
                # Replace == with !=
                mutated_ops = list(node.ops)
                mutated_ops[i] = ast.NotEq()
                mutated = ast.Compare(left=node.left, ops=mutated_ops, comparators=node.comparators)
                self._create_mutant(
                    node,
                    mutated,
                    "Comparison: Replaced == with !=",
                    f"Line {node.lineno}",
                )

            elif isinstance(op, ast.NotEq):
                # Replace != with ==
                mutated_ops = list(node.ops)
                mutated_ops[i] = ast.Eq()
                mutated = ast.Compare(left=node.left, ops=mutated_ops, comparators=node.comparators)
                self._create_mutant(
                    node,
                    mutated,
                    "Comparison: Replaced != with ==",
                    f"Line {node.lineno}",
                )

        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        """
        Generate unary operator mutations.

        Examples:
        - not a -> (removed)
        - -a -> +a
        """
        if isinstance(node.op, ast.Not):
            # Remove not: if not x -> if x
            self._create_mutant(
                node,
                node.operand,
                "Logical: Removed 'not' operator",
                f"Line {node.lineno}",
            )

        elif isinstance(node.op, ast.USub):
            # Replace - with + (if applicable)
            mutated = ast.UnaryOp(op=ast.UAdd(), operand=node.operand)
            self._create_mutant(
                node,
                mutated,
                "Arithmetic: Replaced unary - with unary +",
                f"Line {node.lineno}",
            )

        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """
        Generate constant mutations.

        Examples:
        - 0 -> 1
        - 1 -> 0
        - True -> False
        """
        if isinstance(node.value, bool):
            # Flip boolean
            mutated = ast.Constant(value=not node.value)
            self._create_mutant(
                node,
                mutated,
                f"Constant: Replaced {node.value} with {not node.value}",
                f"Line {node.lineno}",
            )

        elif isinstance(node.value, int) and node.value == 0:
            # Replace 0 with 1
            mutated = ast.Constant(value=1)
            self._create_mutant(
                node,
                mutated,
                "Constant: Replaced 0 with 1",
                f"Line {node.lineno}",
            )

        elif isinstance(node.value, int) and node.value == 1:
            # Replace 1 with 0
            mutated = ast.Constant(value=0)
            self._create_mutant(
                node,
                mutated,
                "Constant: Replaced 1 with 0",
                f"Line {node.lineno}",
            )

        return node


class MutationScoreValidator:
    """
    Validates mutation scores against quality thresholds.

    Cognitive QA: Define acceptable quality standards
    """

    MINIMUM_SCORE = 60.0  # Minimum acceptable mutation score
    TARGET_SCORE = 80.0   # Target mutation score for high quality

    @classmethod
    def validate(cls, score: float) -> tuple[bool, str]:
        """
        Validate mutation score against thresholds.

        Args:
            score: Mutation score (0-100)

        Returns:
            (is_valid, message) tuple
        """
        if score >= cls.TARGET_SCORE:
            return True, f"✅ Excellent: {score:.1f}% ≥ {cls.TARGET_SCORE}% target"
        elif score >= cls.MINIMUM_SCORE:
            return True, f"⚠️  Acceptable: {score:.1f}% ≥ {cls.MINIMUM_SCORE}% minimum"
        else:
            return False, f"❌ Poor: {score:.1f}% < {cls.MINIMUM_SCORE}% minimum"

    @classmethod
    def get_improvement_suggestions(cls, results: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements based on mutation test results.

        Args:
            results: Results from mutation testing

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if results["mutation_score"] < cls.TARGET_SCORE:
            suggestions.append(
                "Add more edge case tests to kill surviving mutants"
            )

            # Check specific patterns
            survived = [m for m in results["mutants"] if not m.killed]
            arithmetic_survived = sum(1 for m in survived if "Arithmetic" in m.description)
            comparison_survived = sum(1 for m in survived if "Comparison" in m.description)

            if arithmetic_survived > 0:
                suggestions.append(
                    f"Add tests for arithmetic operations ({arithmetic_survived} mutants survived)"
                )

            if comparison_survived > 0:
                suggestions.append(
                    f"Add boundary value tests for comparisons ({comparison_survived} mutants survived)"
                )

        return suggestions


# Pytest fixtures and helpers
@pytest.fixture
def mutation_test_config():
    """
    Fixture providing mutation test configuration.

    Usage:
        def test_my_code(mutation_test_config):
            # This test should kill specific mutants
            assert my_function(2, 3) == 5
    """
    return {
        "target_score": 80.0,
        "enabled_mutations": [
            "arithmetic",
            "comparison",
            "logical",
            "constant",
        ],
    }


def run_mutation_tests_on_file(source_file: str, test_args: List[str] = None):
    """
    Helper function to run mutation tests on a specific file.

    Args:
        source_file: Path to source file (relative to project root)
        test_args: Additional arguments for pytest

    Example:
        results = run_mutation_tests_on_file(
            "src/graphwiz_trader/strategies/modern_strategies.py",
            ["tests/unit/test_grid_trading_strategy.py"]
        )
    """
    source_path = Path(source_file)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    tester = MutationTester(source_path)
    mutants = tester.generate_mutants()

    test_command = ["pytest", "-v"]
    if test_args:
        test_command.extend(test_args)

    results = tester.run_mutation_tests(test_command)
    report = tester.generate_report(results)

    return results, report


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mutation_test_framework.py <source_file.py> [test_args...]")
        print("\nExample:")
        print("  python mutation_test_framework.py src/graphwiz_trader/strategies/modern_strategies.py")
        print("  python mutation_test_framework.py src/graphwiz_trader/strategies/modern_strategies.py tests/unit/")
        sys.exit(1)

    source_file = sys.argv[1]
    test_args = sys.argv[2:] if len(sys.argv) > 2 else ["tests/"]

    results, report = run_mutation_tests_on_file(source_file, test_args)

    print(report)

    # Exit with appropriate code
    is_valid, _ = MutationScoreValidator.validate(results["mutation_score"])
    sys.exit(0 if is_valid else 1)
