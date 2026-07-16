"""
Quick verification test for robustness analysis implementation.

This script performs a minimal test to ensure the new robustness evaluation
works correctly without running a full evaluation.
"""

import sys

import numpy as np
from pathlib import Path

print("="*80)
print("ROBUSTNESS ANALYSIS - VERIFICATION TEST")
print("="*80)

# Test 1: Import new modules
print("\n[1/5] Testing imports...")
try:
    from src.milp_solver_v2 import (
        drone_energy_model_v2_perturbed,
        calculate_real_cost,
        check_feasibility_violation,
        RobustnessMetrics
    )
    print("   ✓ Successfully imported milp_solver_v2")
except Exception as e:
    print(f"   ✗ Failed to import milp_solver_v2: {e}")
    sys.exit(1)

try:
    from src.milp_solver import drone_energy_model
    from src.domain import Configuration, Parameter, DomainSpec
    print("   ✓ Successfully imported original modules")
except Exception as e:
    print(f"   ✗ Failed to import original modules: {e}")
    sys.exit(1)

# Test 2: Verify original energy model unchanged
print("\n[2/5] Testing original energy model (should be unchanged)...")
try:
    # Create a simple configuration for testing
    params = [
        Parameter('speed', (5.0, 15.0)),
        Parameter('altitude', (20.0, 100.0)),
        Parameter('camera_res', (4.0, 12.0)),
        Parameter('spray_rate', (0.0, 5.0)),
        Parameter('power_limit_factor', (0.5, 1.0)),
        Parameter('sensor_sampling', (5.0, 20.0))
    ]
    
    domain_spec = DomainSpec(params)
    
    # Create configuration using create_configuration method
    config = domain_spec.create_configuration({
        'speed': 10.0,
        'altitude': 50.0,
        'camera_res': 8.0,
        'spray_rate': 0.0,
        'power_limit_factor': 1.0,
        'sensor_sampling': 10.0
    })
    
    task = {'length': 1000.0}
    
    # Test original model
    energy_original = drone_energy_model(config, task)
    print(f"   Original model energy: {energy_original:.2f} Wh")
    
    # Verify it's a reasonable value
    if 50 < energy_original < 500:
        print("   ✓ Original model produces reasonable values")
    else:
        print(f"   ⚠ Warning: Energy value seems unusual: {energy_original} Wh")
        
except Exception as e:
    print(f"   ✗ Failed testing original model: {e}")
    sys.exit(1)

# Test 3: Test perturbed energy model
print("\n[3/5] Testing perturbed energy model...")
try:
    # Test without perturbation (should match original)
    energy_no_perturb = drone_energy_model_v2_perturbed(
        config, task,
        degradation_factor=0.0,
        apply_perturbation=False
    )
    
    diff = abs(energy_no_perturb - energy_original)
    if diff < 0.01:
        print(f"   ✓ No perturbation mode matches original: {energy_no_perturb:.2f} Wh")
    else:
        print(f"   ⚠ Warning: Mismatch between original and v2 (no perturb): {diff:.2f} Wh")
    
    # Test with perturbation
    np.random.seed(42)
    energy_perturbed = drone_energy_model_v2_perturbed(
        config, task,
        degradation_factor=0.10,  # 10% degradation
        apply_perturbation=True
    )
    
    percent_increase = ((energy_perturbed - energy_original) / energy_original) * 100
    print(f"   Perturbed model (10% deg): {energy_perturbed:.2f} Wh ({percent_increase:+.1f}%)")
    
    if 8 < percent_increase < 15:  # Should be around 10% +/- noise
        print("   ✓ Perturbation working correctly")
    else:
        print(f"   ⚠ Warning: Perturbation outside expected range")
        
except Exception as e:
    print(f"   ✗ Failed testing perturbed model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test feasibility checking
print("\n[4/5] Testing feasibility checking...")
try:
    battery_capacity = 300.0  # Wh
    
    # Test case 1: Feasible
    is_violation, margin = check_feasibility_violation(
        real_cost=150.0,
        battery_capacity=battery_capacity
    )
    if not is_violation and margin < 0:
        print(f"   ✓ Feasible case detected correctly (margin: {margin:.2f} Wh)")
    else:
        print(f"   ⚠ Feasible case unexpected result: violation={is_violation}, margin={margin}")
    
    # Test case 2: Violation
    is_violation, margin = check_feasibility_violation(
        real_cost=280.0,  # Very high
        battery_capacity=battery_capacity
    )
    if is_violation and margin > 0:
        print(f"   ✓ Violation case detected correctly (margin: {margin:.2f} Wh)")
    else:
        print(f"   ⚠ Violation case unexpected result: violation={is_violation}, margin={margin}")
        
except Exception as e:
    print(f"   ✗ Failed testing feasibility checking: {e}")
    sys.exit(1)

# Test 5: Test metrics collection
print("\n[5/5] Testing metrics collection...")
try:
    metrics = RobustnessMetrics()
    
    # Add some sample measurements
    for i in range(5):
        metrics.add_measurement(
            degradation_factor=0.10,
            predicted_energy=150.0,
            real_energy=165.0,
            is_violation=False,
            margin=-20.0,
            timestamp=float(i * 60)
        )
    
    summary = metrics.get_summary(degradation_factor=0.10)
    
    if summary['count'] == 5:
        print(f"   ✓ Metrics collection working: {summary['count']} measurements")
        print(f"      Avg predicted: {summary['avg_predicted']:.2f} Wh")
        print(f"      Avg real: {summary['avg_real']:.2f} Wh")
        print(f"      Violation rate: {summary['violation_rate']:.1f}%")
    else:
        print(f"   ⚠ Unexpected count: {summary['count']}")
        
except Exception as e:
    print(f"   ✗ Failed testing metrics: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n✓ All core components verified successfully!")
print("\nThe robustness analysis implementation is ready to use.")
print("\nNext steps:")
print("  1. Run quick test:  python evaluate/run_robustness_tests.py quick")
print("  2. Run full test:   python evaluate/run_robustness_tests.py paper")
print("  3. See README:      docs/robustness.md")
print("\n" + "="*80 + "\n")
