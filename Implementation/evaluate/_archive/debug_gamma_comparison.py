"""
Debug script to compare Gamma V1 vs V2 constraint generation.

This will help us understand why V2 starts at 40% coverage vs 0% for V1.
"""

import numpy as np
from evaluate.implementation_specifics.domain_specific import Fleet, TightDroneODD
from evaluate.implementation_specifics.weather import WeatherEnvironmentFixed
from src.zonotope_ops import Zonotope
from paths import ASSETS_DIR

def analyze_initial_scenario():
    """Analyze the first weather scenario with both gamma versions."""
    
    # Load configuration
    drone_domain = Fleet(file_path=str(ASSETS_DIR / "fleet.json"))
    domain_spec = drone_domain.get_domain_spec()
    
    # Simulate the initial weather scenario manually
    # Based on weather_scenario_330_temp_5_realistic
    initial_weather = {
        'wind': 4.0,  # Typical for this scenario
        'temperature': 8.0,
        'visibility': 13.5,
        'humidity': 72.0,
        'precipitation': 0.0
    }
    print("="*70)
    print("INITIAL WEATHER SCENARIO: weather_scenario_330_temp_5_realistic")
    print("="*70)
    print(f"Wind: {initial_weather.get('wind')} m/s")
    print(f"Temperature: {initial_weather.get('temperature')} °C")
    print(f"Visibility: {initial_weather.get('visibility')} km")
    print(f"Humidity: {initial_weather.get('humidity')} %")
    print(f"Precipitation: {initial_weather.get('precipitation', 0)} mm/h")
    print()
    
    # Calculate CS volume
    cs_bounds = domain_spec.get_bounds()
    cs_volume = np.prod([b[1] - b[0] for b in cs_bounds])
    print(f"Total CS Volume: {cs_volume:.2e}")
    print()
    
    # Create ODD and test both gamma versions
    for version in ['v1', 'v2']:
        print("="*70)
        print(f"TESTING GAMMA {version.upper()}")
        print("="*70)
        
        # Set gamma version
        TightDroneODD.set_gamma_version(version)
        
        # Create ODD
        odd = TightDroneODD(
            timestamp=0,
            conditions={
                'wind': initial_weather.get('wind'),
                'temperature': initial_weather.get('temperature'),
                'visibility': initial_weather.get('visibility'),
                'humidity': initial_weather.get('humidity')
            }
        )
        
        # Get constraints
        constraints = odd.apply_domain_constraints(domain_spec)
        
        print("\nConstraints Applied:")
        print("-" * 50)
        for param, bounds in constraints.items():
            original = None
            for p in domain_spec.parameters:
                if p.name == param:
                    original = p.bounds
                    break
            if original:
                restriction_pct = ((bounds[1] - bounds[0]) / (original[1] - original[0])) * 100
                print(f"  {param:20s}: [{bounds[0]:6.2f}, {bounds[1]:6.2f}]  "
                      f"(was [{original[0]:6.2f}, {original[1]:6.2f}])  "
                      f"→ {restriction_pct:5.1f}% of original")
        
        # Calculate DS volume
        ds_bounds = []
        for p in domain_spec.parameters:
            if p.name in constraints:
                ds_bounds.append(constraints[p.name])
            else:
                ds_bounds.append(p.bounds)
        
        ds_volume = np.prod([b[1] - b[0] for b in ds_bounds])
        coverage_pct = (ds_volume / cs_volume) * 100
        
        print()
        print(f"DS Volume: {ds_volume:.2e}")
        print(f"Coverage: {coverage_pct:.2f}%")
        print()
        
        # Analyze each parameter's contribution
        print("Parameter Contribution to Volume Reduction:")
        print("-" * 50)
        for i, p in enumerate(domain_spec.parameters):
            original_range = p.bounds[1] - p.bounds[0]
            constrained_range = ds_bounds[i][1] - ds_bounds[i][0]
            reduction_factor = constrained_range / original_range
            print(f"  {p.name:20s}: {reduction_factor:6.3f}x  "
                  f"({constrained_range:6.2f} / {original_range:6.2f})")
        print()

def compare_gamma_sensitivity():
    """Test how V1 and V2 respond to small environmental changes."""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: How do constraints change with weather?")
    print("="*70)
    print()
    
    drone_domain = Fleet(file_path=str(ASSETS_DIR / "fleet.json"))
    domain_spec = drone_domain.get_domain_spec()

    # Test different wind speeds
    wind_speeds = [3.0, 5.0, 8.0, 12.0, 15.0]
    
    for version in ['v1', 'v2']:
        print(f"\nGAMMA {version.upper()} - Wind Speed Sensitivity:")
        print("-" * 70)
        TightDroneODD.set_gamma_version(version)
        
        prev_constraints = None
        for wind in wind_speeds:
            odd = TightDroneODD(
                timestamp=0,
                conditions={
                    'wind': wind,
                    'temperature': 15.0,
                    'visibility': 10.0,
                    'humidity': 50.0
                }
            )
            
            constraints = odd.apply_domain_constraints(domain_spec)
            speed_range = constraints.get('speed', (0, 0))
            
            if prev_constraints:
                prev_speed = prev_constraints.get('speed', (0, 0))
                change = abs(speed_range[0] - prev_speed[0]) + abs(speed_range[1] - prev_speed[1])
                print(f"  Wind {wind:4.1f} m/s → speed [{speed_range[0]:5.1f}, {speed_range[1]:5.1f}]  "
                      f"(Δ={change:5.2f} from previous)")
            else:
                print(f"  Wind {wind:4.1f} m/s → speed [{speed_range[0]:5.1f}, {speed_range[1]:5.1f}]")
            
            prev_constraints = constraints

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# GAMMA V1 vs V2 DIAGNOSTIC COMPARISON")
    print("#"*70)
    print()
    
    analyze_initial_scenario()
    compare_gamma_sensitivity()
    
    print("\n" + "#"*70)
    print("# ANALYSIS COMPLETE")
    print("#"*70)
