#!/usr/bin/env python3
"""
Quick runner script for robustness analysis experiments.

This script provides easy-to-use commands for running the robustness evaluation
described in docs/experiments.md without modifying existing code.
"""

import subprocess
import sys
from pathlib import Path

from paths import PROJECT_ROOT

def run_robustness_quick_test():
    """Run a quick test (30 minutes, coarse degradation factors)."""
    print("Running quick robustness test (30 min, 3 degradation levels)...")
    cmd = [
        sys.executable,
        "-m", "evaluate.experiments.robustness",
        "--duration", "30",
        "--adaptation_interval", "5",
        "--degradation_factors", "0.0,0.10,0.20",
        "--output_dir", "quick",
        "--seed", "42"
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_robustness_paper():
    """Run full paper-quality evaluation (3 hours, 5 degradation levels)."""
    print("Running full paper-quality robustness evaluation (3 hours, 5 degradation levels)...")
    cmd = [
        sys.executable,
        "-m", "evaluate.experiments.robustness",
        "--duration", "180",
        "--adaptation_interval", "5",
        "--degradation_factors", "0.0,0.05,0.10,0.15,0.20",
        "--output_dir", "paper",
        "--seed", "42"
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_robustness_multiple_scenarios():
    """Run across multiple weather scenarios for comprehensive analysis."""
    print("Running robustness across multiple weather scenarios...")
    
    scenarios = [
        (90, 15),   # Spring, moderate temp
        (180, 25),  # Summer, high temp
        (330, 5)    # Winter, low temp
    ]
    
    for day, temp in scenarios:
        print(f"\n{'='*60}")
        print(f"Weather scenario: Day {day}, Temperature {temp}°C")
        print('='*60)
        
        cmd = [
            sys.executable,
            "-m", "evaluate.experiments.robustness",
            "--duration", "180",
            "--adaptation_interval", "5",
            "--degradation_factors", "0.0,0.05,0.10,0.15,0.20",
            "--weather_scenario", str(day),
            "--temperature", str(temp),
            "--output_dir", "multi_scenarios",
            "--seed", "42"
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT)


def print_usage():
    """Print usage instructions."""
    print("""
Robustness Analysis Runner
===========================

Usage: python -m evaluate.experiments.run_robustness_tests [option]

Options:
  quick       - Quick test (30 min, 3 degradation levels)
  paper       - Full paper evaluation (3 hours, 5 degradation levels)  
  multi       - Multiple weather scenarios (comprehensive)
  help        - Show this help message

Examples:
  python -m evaluate.experiments.run_robustness_tests quick
  python -m evaluate.experiments.run_robustness_tests paper
  python -m evaluate.experiments.run_robustness_tests multi

Direct Usage (without this runner):
  python -m evaluate.experiments.robustness --duration 60 --degradation_factors "0.0,0.05,0.10"

For more options:
  python -m evaluate.experiments.robustness --help
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    option = sys.argv[1].lower()
    
    if option == "quick":
        run_robustness_quick_test()
    elif option == "paper":
        run_robustness_paper()
    elif option == "multi":
        run_robustness_multiple_scenarios()
    elif option == "help":
        print_usage()
    else:
        print(f"Unknown option: {option}")
        print_usage()


if __name__ == "__main__":
    main()
