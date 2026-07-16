

from src.domain import DomainSpec, Parameter,  Configuration, ODD, ConfigurationSpace,ParameterType
import numpy as np
from dataclasses import dataclass
from typing import List
import os

"""
Domain-Specific Implementation for Drone Operations

This module defines drone-specific Operational Design Domains (ODDs) with environmental
constraints. It includes:

1. DroneODD: Base class with continuous physics-based constraints
   - apply_constraints() modifies bounds arrays directly using continuous functions
   - Models gradual performance degradation with environmental changes
   - Uses realistic physical relationships between conditions and parameters
   - Used when you want smooth, continuous constraint application

2. TightDroneODD: Subclass for creating smaller, more constrained domains
   - OVERRIDES apply_constraints() - does NOT call DroneODD's version
   - Two gamma function implementations (V1/V2) that define ABSOLUTE ranges:
     * V1 (discrete): Threshold-based with distinct operational regions
     * V2 (continuous): Smooth physics-based constraints (tighter than DroneODD)
   - Used for cache testing where you want well-defined tight regions
   
IMPORTANT DESIGN DECISION:
  - DroneODD.apply_constraints() is for CONTINUOUS constraint application on bounds
  - TightDroneODD.apply_constraints() REPLACES this with gamma-based ABSOLUTE ranges
  - They are mutually exclusive - only ONE is applied, not both!
  
To switch between gamma functions:
    TightDroneODD.gamma_version = 'v1'  # Discrete thresholds (default)
    TightDroneODD.gamma_version = 'v2'  # Continuous physics-based

Environmental Factors:
    - wind_ms: Wind speed (m/s) - affects speed, altitude, spray accuracy
    - temperature_c: Temperature (°C) - affects battery, sensors
    - visibility_km: Visibility (km) - affects camera, safe altitude
    - humidity_pct: Humidity (%) - affects spray, electronics
    - precipitation_mm: Precipitation (mm/h) - safety constraint
"""

# Example: Drone-specific ODD with constraint logic
@dataclass(eq=False)
class DroneODD(ODD):
    """
    Drone-specific ODD with environmental constraints.
    
    Note: We use eq=False to inherit custom __hash__ from parent ODD class.
    
    Common environmental factors for drone operations:
    - wind_ms: Wind speed (m/s)
    - temperature_c: Temperature (°C)
    - visibility_km: Visibility (km)
    - humidity_pct: Humidity (%)
    - precipitation_mm: Precipitation (mm/h)
    """
    
    @property
    def wind_ms(self) -> float:
        return self.get('wind', 0.0)
    
    @property
    def temperature_c(self) -> float:
        return self.get('temperature', 20.0)
    
    @property
    def visibility_km(self) -> float:
        return self.get('visibility', 10.0)
    
    @property
    def humidity_pct(self) -> float:
        return self.get('humidity', 50.0)
    
    @property
    def precipitation_mm(self) -> float:
        return self.get('precipitation', 0.0)
    
    def apply_constraints(self, bounds: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        """
        Apply drone-specific environmental constraints using continuous physics-based models.
        
        NOTE: This method is OVERRIDDEN by TightDroneODD!
        - If you use TightDroneODD, this method is NOT called
        - TightDroneODD uses absolute ranges from gamma functions instead
        - Use DroneODD directly for continuous constraint application
        - Use TightDroneODD for discrete region-based constraints
        
        Constraints applied:
        - Wind: Reduces max speed and altitude, affects spray accuracy
        - Temperature: Affects battery capacity and sensor performance
        - Visibility: Affects camera requirements and safe altitude
        - Humidity: Affects electronics reliability and spray effectiveness
        - Precipitation: Safety constraint for multiple parameters
        - Combined effects: Cross-factor interactions
        
        Args:
            bounds: Current parameter bounds (n × 2 array)
            domain_spec: Domain specification with parameter info
            
        Returns:
            Modified bounds (n × 2 array) with environmental constraints applied
        """
        constrained = bounds.copy()
        param_names = domain_spec.parameter_names
        
        # === WIND CONSTRAINTS ===
        if 'speed' in param_names:
            speed_idx = param_names.index('speed')
            # Exponential reduction of max speed with wind (quadratic drag)
            # Speed reduction = wind * (1 + wind/15) to model non-linear drag
            wind_reduction = self.wind_ms * (1.0 + self.wind_ms / 15.0)
            constrained[speed_idx, 1] = max(
                constrained[speed_idx, 0],
                constrained[speed_idx, 1] - wind_reduction
            )
        
        if 'altitude' in param_names and self.wind_ms > 5:
            altitude_idx = param_names.index('altitude')
            # High wind at altitude is dangerous - reduce max altitude
            # Each m/s above 5 reduces safe altitude by 5m
            altitude_reduction = (self.wind_ms - 5) * 5.0
            constrained[altitude_idx, 1] = max(
                constrained[altitude_idx, 0],
                constrained[altitude_idx, 1] - altitude_reduction
            )
        
        if 'spray_rate' in param_names and self.wind_ms > 3:
            spray_idx = param_names.index('spray_rate')
            # Wind causes drift - reduce max spray rate for accuracy
            # Above 3 m/s, each additional m/s reduces max spray by 10%
            spray_factor = 1.0 - (self.wind_ms - 3) * 0.1
            spray_factor = max(0.3, spray_factor)  # At least 30% capacity
            constrained[spray_idx, 1] = max(
                constrained[spray_idx, 0],
                constrained[spray_idx, 1] * spray_factor
            )
        
        # === TEMPERATURE CONSTRAINTS ===
        if 'power_limit_factor' in param_names:
            battery_idx = param_names.index('power_limit_factor')
            temp = self.temperature_c
            
            # LiPo battery performance curve: optimal 15-25°C
            # Below 0°C: capacity drops ~50%, Above 40°C: safety concerns
            if temp < 15:
                # Cold reduces capacity: -2% per degree below 15°C
                temp_reduction = (15 - temp) * 0.02
                temp_reduction = min(temp_reduction, 0.5)  # Max 50% reduction
                constrained[battery_idx, 1] = max(
                    constrained[battery_idx, 0],
                    constrained[battery_idx, 1] * (1.0 - temp_reduction)
                )
            elif temp > 30:
                # Heat reduces safe max power: -1.5% per degree above 30°C
                temp_reduction = (temp - 30) * 0.015
                temp_reduction = min(temp_reduction, 0.35)  # Max 35% reduction
                constrained[battery_idx, 1] = max(
                    constrained[battery_idx, 0],
                    constrained[battery_idx, 1] * (1.0 - temp_reduction)
                )
        
        if 'sensor_sampling' in param_names:
            sensor_idx = param_names.index('sensor_sampling')
            temp = self.temperature_c
            # Sensors have optimal temperature range
            if temp < 0 or temp > 40:
                # Extreme temperatures require lower sampling for reliability
                constrained[sensor_idx, 1] = max(
                    constrained[sensor_idx, 0],
                    constrained[sensor_idx, 1] * 0.7
                )
        
        # === VISIBILITY CONSTRAINTS ===
        if 'camera_res' in param_names:
            camera_idx = param_names.index('camera_res')
            vis = self.visibility_km
            
            if vis < 10:
                # Poor visibility requires higher resolution
                # Linear increase: 10% more resolution per km below 10
                visibility_factor = (10.0 - vis) / 10.0
                range_size = constrained[camera_idx, 1] - constrained[camera_idx, 0]
                min_increase = range_size * visibility_factor * 0.4
                constrained[camera_idx, 0] = min(
                    constrained[camera_idx, 1],
                    constrained[camera_idx, 0] + min_increase
                )
        
        if 'altitude' in param_names:
            altitude_idx = param_names.index('altitude')
            vis = self.visibility_km
            if vis < 5:
                # Very poor visibility limits safe altitude
                # Must stay lower to maintain visual line of sight
                altitude_factor = vis / 5.0  # 0.0 to 1.0
                constrained[altitude_idx, 1] = max(
                    constrained[altitude_idx, 0],
                    constrained[altitude_idx, 0] + 
                    (constrained[altitude_idx, 1] - constrained[altitude_idx, 0]) * 
                    (0.3 + 0.7 * altitude_factor)  # 30% to 100% of range
                )
        
        # === HUMIDITY CONSTRAINTS ===
        if 'spray_rate' in param_names:
            spray_idx = param_names.index('spray_rate')
            hum = self.humidity_pct
            
            if hum > 80:
                # Very high humidity affects spray evaporation and electronics
                # Reduce max spray rate by 1% per percentage point above 80
                humidity_reduction = (hum - 80) * 0.01
                humidity_reduction = min(humidity_reduction, 0.20)  # Max 20% reduction
                constrained[spray_idx, 1] = max(
                    constrained[spray_idx, 0],
                    constrained[spray_idx, 1] * (1.0 - humidity_reduction)
                )
            elif hum < 30:
                # Very low humidity causes rapid evaporation - reduce effectiveness
                # Increase minimum spray rate to compensate
                range_size = constrained[spray_idx, 1] - constrained[spray_idx, 0]
                min_increase = range_size * (30 - hum) / 30.0 * 0.2
                constrained[spray_idx, 0] = min(
                    constrained[spray_idx, 1],
                    constrained[spray_idx, 0] + min_increase
                )
        
        if 'sensor_sampling' in param_names and self.humidity_pct > 85:
            sensor_idx = param_names.index('sensor_sampling')
            # Very high humidity risks condensation on sensors
            constrained[sensor_idx, 1] = max(
                constrained[sensor_idx, 0],
                constrained[sensor_idx, 1] * 0.8
            )
        
        # === PRECIPITATION CONSTRAINTS ===
        # Rain significantly limits operations
        precip = self.precipitation_mm
        if precip > 0.5:  # Light rain threshold
            # Reduce all operational parameters for safety
            if 'speed' in param_names:
                speed_idx = param_names.index('speed')
                rain_factor = min(0.5, precip / 5.0)  # Up to 50% reduction
                constrained[speed_idx, 1] = max(
                    constrained[speed_idx, 0],
                    constrained[speed_idx, 1] * (1.0 - rain_factor)
                )
            
            if 'altitude' in param_names:
                altitude_idx = param_names.index('altitude')
                # Rain limits altitude (visibility + weight of water)
                constrained[altitude_idx, 1] = max(
                    constrained[altitude_idx, 0],
                    constrained[altitude_idx, 1] * 0.6
                )
            
            if 'spray_rate' in param_names:
                spray_idx = param_names.index('spray_rate')
                # Rain makes spraying ineffective
                constrained[spray_idx, 1] = max(
                    constrained[spray_idx, 0],
                    constrained[spray_idx, 1] * 0.3
                )
        
        # === COMBINED EFFECTS ===
        # Wind + Humidity combined effect on spray
        if 'spray_rate' in param_names and self.wind_ms > 5 and self.humidity_pct > 70:
            spray_idx = param_names.index('spray_rate')
            # Combined effect is worse than individual effects
            combined_factor = 0.85  # Additional 15% reduction
            constrained[spray_idx, 1] = max(
                constrained[spray_idx, 0],
                constrained[spray_idx, 1] * combined_factor
            )
        
        return constrained



class TightDroneODD(DroneODD):
    """
    DroneODD with MUCH TIGHTER constraints to create smaller DS regions.
    
    This makes each DS only 5-30% of CS instead of 90%+.
    Supports two gamma function implementations for comparison.
    
    IMPORTANT: This class OVERRIDES apply_constraints() from DroneODD.
    - The gamma functions (V1/V2) define ABSOLUTE parameter ranges based on environment
    - These ranges are intersected with the original CS bounds
    - DroneODD.apply_constraints() is NOT called (to avoid double application)
    
    Configuration:
        gamma_version: 'v1' (discrete thresholds) or 'v2' (continuous physics-based)
    
    Design Note:
        V1: Uses discrete environmental thresholds → distinct operational regions
        V2: Uses continuous physics models → smooth degradation (similar philosophy to DroneODD but tighter ranges)
    """
    
    # Class variable to control which gamma function to use
    gamma_version = 'v2'  # Change to 'v2' to use the alternative implementation
    
    @classmethod
    def set_gamma_version(cls, version: str):
        """
        Set which gamma function implementation to use.
        
        Args:
            version: Either 'v1' (discrete thresholds) or 'v2' (continuous physics-based)
        
        Example:
            TightDroneODD.set_gamma_version('v2')
        """
        if version not in ['v1', 'v2']:
            raise ValueError(f"Invalid gamma version '{version}'. Must be 'v1' or 'v2'.")
        cls.gamma_version = version
        print(f"Gamma function set to version {version}")
    
    @classmethod
    def compare_gamma_functions(cls, odd_instance):
        """
        Compare the output of both gamma functions for a given ODD instance.
        
        Args:
            odd_instance: An instance of TightDroneODD with specific environmental conditions
        
        Returns:
            dict with 'v1' and 'v2' keys containing constraint dictionaries
        
        Example:
            odd = TightDroneODD(timestamp=0, conditions={'wind': 10, 'temperature': 25, 'visibility': 8, 'humidity': 60})
            comparison = TightDroneODD.compare_gamma_functions(odd)
            print("V1:", comparison['v1'])
            print("V2:", comparison['v2'])
        """
        v1_constraints = odd_instance.apply_domain_constraints_v1()
        v2_constraints = odd_instance.apply_domain_constraints_v2()
        
        print(f"\nComparison for ODD: {odd_instance}")
        print(f"{'Parameter':<20} {'V1 (Discrete)':<25} {'V2 (Continuous)':<25} {'Difference'}")
        print("-" * 90)
        
        all_params = set(v1_constraints.keys()) | set(v2_constraints.keys())
        for param in sorted(all_params):
            v1_range = v1_constraints.get(param, (None, None))
            v2_range = v2_constraints.get(param, (None, None))
            
            if v1_range[0] is not None and v2_range[0] is not None:
                v1_size = v1_range[1] - v1_range[0]
                v2_size = v2_range[1] - v2_range[0]
                diff_pct = ((v2_size - v1_size) / v1_size * 100) if v1_size > 0 else 0
                print(f"{param:<20} {str(v1_range):<25} {str(v2_range):<25} {diff_pct:+.1f}%")
            else:
                print(f"{param:<20} {str(v1_range):<25} {str(v2_range):<25} N/A")
        
        return {'v1': v1_constraints, 'v2': v2_constraints}

    def apply_domain_constraints(self, domain_spec: DomainSpec = None) -> dict:
        """
        This is the Gamma function from the paper.
        Routes to the appropriate implementation based on gamma_version.
        """
        if self.gamma_version == 'v2':
            constraints = self.apply_domain_constraints_v2(domain_spec)
        else:
            constraints = self.apply_domain_constraints_v1(domain_spec)
        
        # CRITICAL FIX: Clip all constraints to CS bounds
        # V2 can generate bounds outside CS, which breaks coverage calculation
        if domain_spec is not None:
            for param in domain_spec.parameters:
                if param.name in constraints:
                    cs_min, cs_max = param.bounds
                    constraint_min, constraint_max = constraints[param.name]
                    
                    # Clip to CS bounds
                    clipped_min = max(cs_min, constraint_min)
                    clipped_max = min(cs_max, constraint_max)
                    
                    # Ensure min < max after clipping
                    if clipped_min >= clipped_max:
                        clipped_min = cs_min
                        clipped_max = cs_max
                    
                    constraints[param.name] = (clipped_min, clipped_max)
        
        return constraints
    
    def apply_domain_constraints_v1(self, domain_spec: DomainSpec = None) -> dict:
        """
        Gamma function V1: Discrete threshold-based constraints (ORIGINAL).
        
        Uses hard thresholds for environmental conditions to create distinct operational regions.
        Each condition range maps to a specific parameter constraint range.
        
        Pros:
        - Creates clear, distinct operational regions
        - Easy to understand and verify
        - Good for testing cache behavior with clear boundaries
        
        Cons:
        - Discontinuous jumps at thresholds
        - Doesn't capture gradual degradation
        - May miss intermediate conditions
        """
        constraints = {}
        
        wind = self.wind_ms
        temp = self.temperature_c
        vis = self.visibility_km
        hum = self.humidity_pct
        precip = self.precipitation_mm
        
        # Speed: heavily constrained by wind
        if wind > 12:
            constraints['speed'] = (3, 15)  # 26% of range - dangerous winds
        elif wind > 8:
            constraints['speed'] = (5, 25)  # 43% of range - strong winds
        elif wind > 5:
            constraints['speed'] = (10, 35)  # 53% of range - moderate winds
        else:
            constraints['speed'] = (15, 50)  # 74% of range - calm conditions
        
        # Altitude: constrained by visibility and wind
        if vis < 6 or wind > 10:
            constraints['altitude'] = (5, 40)  # 24% of range - stay low
        elif vis < 8 or wind > 7:
            constraints['altitude'] = (10, 80)  # 48% of range - moderate altitude
        else:
            constraints['altitude'] = (20, 150)  # 90% of range - full altitude range
        
        # Camera: constrained by visibility
        if vis < 6:
            constraints['camera_res'] = (2, 8)  # 27% of range - need high res
        elif vis < 8:
            constraints['camera_res'] = (4, 16)  # 55% of range - medium res
        else:
            constraints['camera_res'] = (8, 24)  # 73% of range - any res works
        
        # Spray rate: constrained by wind and humidity
        if wind > 10 or hum > 80 or precip > 1.0:
            constraints['spray_rate'] = (1, 5)  # 29% of range - minimal spraying
        elif wind > 6 or hum > 70 or precip > 0.5:
            constraints['spray_rate'] = (2, 8)  # 50% of range - reduced spraying
        elif wind > 3 or hum < 30:
            constraints['spray_rate'] = (3, 12)  # 64% of range - moderate spraying
        else:
            constraints['spray_rate'] = (4, 15)  # 79% of range - normal operations
        
        # Battery: constrained by temperature extremes
        if temp < -5 or temp > 40:
            constraints['power_limit_factor'] = (0.40, 0.65)  # 59% of range - extreme temps
        elif temp < 5 or temp > 35:
            constraints['power_limit_factor'] = (0.40, 0.75)  # 65% of range - cold/hot
        elif temp < 15 or temp > 30:
            constraints['power_limit_factor'] = (0.55, 0.90)  # 63% of range - suboptimal
        else:
            constraints['power_limit_factor'] = (0.55, 1.0)  # 71% of range - optimal
        
        # Sensors: constrained by wind, humidity, and temperature
        if wind > 10 or hum > 80 or temp < 0 or temp > 40:
            constraints['sensor_sampling'] = (1, 5)  # 36% of range - harsh conditions
        elif wind > 6 or hum > 70 or temp < 10 or temp > 35:
            constraints['sensor_sampling'] = (2, 8)  # 55% of range - challenging
        else:
            constraints['sensor_sampling'] = (4, 12)  # 73% of range - good conditions
        
        return constraints
    
    def apply_domain_constraints_v2(self, domain_spec: DomainSpec = None) -> dict:
        """
        Gamma function V2: More granular discrete threshold-based constraints.
        
        Similar to V1 but with MORE detailed thresholds for finer-grained regions.
        Uses MORE environmental thresholds to create MORE operational regions.
        Still discrete/threshold-based, just less abstracted with more detail.
        
        Pros:
        - More granular operational regions than V1
        - Considers more environmental factors in combination
        - Creates more diverse constraint patterns
        - Better differentiation between similar conditions
        
        Cons:
        - More complex threshold structure
        - More regions to manage in cache
        - Requires more careful tuning
        """
        constraints = {}
        
        wind = self.wind_ms
        temp = self.temperature_c
        vis = self.visibility_km
        hum = self.humidity_pct
        precip = self.precipitation_mm
        
        # === SPEED CONSTRAINTS ===
        # More granular wind thresholds + precipitation consideration
        if precip > 2.0:
            constraints['speed'] = (3, 10)  # 15% of range - heavy rain, minimal ops
        elif precip > 1.0:
            constraints['speed'] = (3, 15)  # 26% of range - moderate rain
        elif wind > 14:
            constraints['speed'] = (3, 12)  # 19% of range - very dangerous winds
        elif wind > 12:
            constraints['speed'] = (3, 18)  # 32% of range - dangerous winds
        elif wind > 10:
            constraints['speed'] = (4, 22)  # 38% of range - very strong winds
        elif wind > 8:
            constraints['speed'] = (5, 28)  # 49% of range - strong winds
        elif wind > 6:
            constraints['speed'] = (8, 33)  # 53% of range - moderate winds
        elif wind > 5:
            constraints['speed'] = (10, 37)  # 57% of range - light-moderate winds
        elif wind > 3:
            constraints['speed'] = (12, 42)  # 64% of range - light winds
        else:
            constraints['speed'] = (15, 50)  # 74% of range - calm conditions
        
        # === ALTITUDE CONSTRAINTS ===
        # Combines visibility, wind, and precipitation
        if precip > 1.5 or (vis < 5 and wind > 8):
            constraints['altitude'] = (5, 30)  # 17% of range - very dangerous combo
        elif precip > 0.5 or vis < 5:
            constraints['altitude'] = (5, 40)  # 24% of range - poor conditions
        elif vis < 6 and wind > 10:
            constraints['altitude'] = (5, 45)  # 28% of range - poor vis + high wind
        elif vis < 6 or wind > 12:
            constraints['altitude'] = (5, 50)  # 31% of range - poor vis OR very high wind
        elif vis < 7 and wind > 9:
            constraints['altitude'] = (8, 65)  # 39% of range - moderate combo
        elif vis < 8 and wind > 7:
            constraints['altitude'] = (10, 75)  # 45% of range - moderate conditions
        elif vis < 8 or wind > 9:
            constraints['altitude'] = (10, 85)  # 52% of range - one moderate factor
        elif vis < 10 or wind > 6:
            constraints['altitude'] = (15, 110)  # 66% of range - slightly suboptimal
        elif vis < 12 and wind > 4:
            constraints['altitude'] = (18, 130)  # 78% of range - good conditions
        else:
            constraints['altitude'] = (20, 150)  # 90% of range - excellent conditions
        
        # === CAMERA RESOLUTION CONSTRAINTS ===
        # More granular visibility thresholds
        if vis < 4:
            constraints['camera_res'] = (16, 24)  # 36% of range - very poor vis
        elif vis < 5:
            constraints['camera_res'] = (12, 24)  # 55% of range - poor vis
        elif vis < 6:
            constraints['camera_res'] = (10, 24)  # 64% of range - low vis
        elif vis < 7:
            constraints['camera_res'] = (8, 22)  # 64% of range - moderate-low vis
        elif vis < 8:
            constraints['camera_res'] = (6, 20)  # 64% of range - moderate vis
        elif vis < 10:
            constraints['camera_res'] = (5, 18)  # 59% of range - decent vis
        elif vis < 12:
            constraints['camera_res'] = (4, 16)  # 55% of range - good vis
        else:
            constraints['camera_res'] = (2, 16)  # 64% of range - excellent vis
        
        # === SPRAY RATE CONSTRAINTS ===
        # Considers wind, humidity, AND precipitation in more detail
        if precip > 2.0:
            constraints['spray_rate'] = (1, 3)  # 14% of range - heavy rain
        elif precip > 1.5 or (wind > 12 and hum > 75):
            constraints['spray_rate'] = (1, 4)  # 21% of range - very poor conditions
        elif precip > 1.0 or wind > 12:
            constraints['spray_rate'] = (1, 5)  # 29% of range - poor conditions
        elif precip > 0.5 or (wind > 10 and hum > 80):
            constraints['spray_rate'] = (1, 6)  # 36% of range - challenging combo
        elif wind > 11 or (wind > 9 and hum > 75):
            constraints['spray_rate'] = (2, 7)  # 36% of range - high wind or combo
        elif wind > 10 or hum > 82:
            constraints['spray_rate'] = (2, 7.5)  # 39% of range - one severe factor
        elif wind > 9 or (wind > 7 and hum > 75):
            constraints['spray_rate'] = (2, 8)  # 43% of range - moderate wind or combo
        elif wind > 8 or hum > 78:
            constraints['spray_rate'] = (2.5, 9)  # 46% of range - moderate factors
        elif wind > 7 or hum > 72:
            constraints['spray_rate'] = (3, 10)  # 50% of range - light-moderate
        elif wind > 6 or hum > 68 or hum < 32:
            constraints['spray_rate'] = (3, 11)  # 57% of range - one suboptimal factor
        elif wind > 4 or hum < 35 or hum > 65:
            constraints['spray_rate'] = (3.5, 12.5)  # 64% of range - slightly suboptimal
        elif wind > 3 or hum < 38:
            constraints['spray_rate'] = (4, 13)  # 64% of range - good with minor issues
        else:
            constraints['spray_rate'] = (4, 15)  # 79% of range - optimal conditions
        
        # === BATTERY/POWER CONSTRAINTS ===
        # More temperature gradations
        if temp < -10 or temp > 45:
            constraints['power_limit_factor'] = (0.40, 0.55)  # 35% of range - extreme danger
        elif temp < -5 or temp > 42:
            constraints['power_limit_factor'] = (0.40, 0.60)  # 47% of range - very extreme
        elif temp < 0 or temp > 40:
            constraints['power_limit_factor'] = (0.40, 0.65)  # 59% of range - extreme
        elif temp < 3 or temp > 38:
            constraints['power_limit_factor'] = (0.45, 0.70)  # 47% of range - very cold/hot
        elif temp < 5 or temp > 36:
            constraints['power_limit_factor'] = (0.45, 0.75)  # 56% of range - cold/hot
        elif temp < 8 or temp > 34:
            constraints['power_limit_factor'] = (0.50, 0.80)  # 56% of range - cool/warm
        elif temp < 12 or temp > 32:
            constraints['power_limit_factor'] = (0.52, 0.85)  # 62% of range - suboptimal
        elif temp < 15 or temp > 30:
            constraints['power_limit_factor'] = (0.55, 0.90)  # 65% of range - acceptable
        elif temp < 18 or temp > 28:
            constraints['power_limit_factor'] = (0.60, 0.95)  # 65% of range - good
        else:
            constraints['power_limit_factor'] = (0.65, 1.0)  # 65% of range - optimal
        
        # === SENSOR SAMPLING CONSTRAINTS ===
        # Combines wind, humidity, and temperature with more thresholds
        if (wind > 12 and hum > 85) or temp < -5 or temp > 42:
            constraints['sensor_sampling'] = (1, 4)  # 27% of range - severe combo or extreme
        elif wind > 12 or hum > 88 or temp < 0 or temp > 40:
            constraints['sensor_sampling'] = (1, 5)  # 36% of range - one severe factor
        elif (wind > 10 and hum > 80) or temp < 2 or temp > 38:
            constraints['sensor_sampling'] = (1, 5.5)  # 41% of range - harsh combo
        elif wind > 11 or hum > 85 or temp < 5 or temp > 36:
            constraints['sensor_sampling'] = (1.5, 6)  # 41% of range - one harsh factor
        elif (wind > 9 and hum > 75) or temp < 8 or temp > 34:
            constraints['sensor_sampling'] = (2, 7)  # 45% of range - moderate combo
        elif wind > 10 or hum > 82 or temp < 10 or temp > 35:
            constraints['sensor_sampling'] = (2, 7.5)  # 50% of range - one moderate factor
        elif (wind > 7 and hum > 72) or temp < 12 or temp > 33:
            constraints['sensor_sampling'] = (2.5, 8.5)  # 55% of range - multiple suboptimal
        elif wind > 8 or hum > 78 or temp < 15 or temp > 32:
            constraints['sensor_sampling'] = (3, 9)  # 55% of range - one suboptimal
        elif wind > 6 or hum > 72 or temp < 18 or temp > 30:
            constraints['sensor_sampling'] = (3.5, 10)  # 59% of range - acceptable
        elif wind > 4 or hum > 65 or temp < 20 or temp > 28:
            constraints['sensor_sampling'] = (4, 11)  # 64% of range - good
        else:
            constraints['sensor_sampling'] = (5, 12)  # 64% of range - optimal
        
        return constraints

    def apply_constraints(self, bounds: np.ndarray, domain_spec: DomainSpec) -> dict:
        """
        Apply tight constraints by intersecting gamma function ranges with CS bounds.
        
        IMPORTANT: This method OVERRIDES DroneODD.apply_constraints()!
        - We do NOT call super().apply_constraints() in the normal flow
        - The gamma functions (V1/V2) define ABSOLUTE ranges, not modifications
        - These absolute ranges are intersected with the incoming bounds
        - Result: Tighter bounds than the original CS
        
        The only time we call super() is when constraints dict is empty (fallback).
        
        Args:
            bounds: Original configuration space bounds (n × 2 array)
            domain_spec: Domain specification with parameter info
            
        Returns:
            Modified bounds array (n × 2) with tighter constraints applied
        """
        # Get gamma function constraints (absolute ranges per parameter)
        constraints = self.apply_domain_constraints(domain_spec)
        
        if constraints:
            # Create tighter bounds by intersecting CS bounds with gamma ranges
            current_bounds = bounds.copy()
            tight_bounds = []

            param_names = [p.name for p in domain_spec.parameters]
            for i, param_name in enumerate(param_names):
                if param_name in constraints:
                    # Intersect current bounds with tight constraint
                    tight_range = constraints[param_name]
                    current_range = current_bounds[i]
                    new_min = max(current_range[0], tight_range[0])
                    new_max = min(current_range[1], tight_range[1])
                    tight_bounds.append([new_min, new_max])
                else:
                    tight_bounds.append(current_bounds[i])
        
            return np.array(tight_bounds)
        else:
            # Fallback: if no constraints, use parent's continuous constraint model
            return super().apply_constraints(bounds, domain_spec)

    def intersects(self, other, threshold=1):
        """
        Check if two objects' parameter domains intersect, considering an optional threshold.
        Expands each interval by `threshold` on both sides.
        """
        # Get parameter bounds: {param_name: (min, max)}
        my_constraints = self.apply_domain_constraints()
        other_constraints = other.apply_domain_constraints()

        # Consider only parameters that exist in both
        common_keys = set(my_constraints.keys()) & set(other_constraints.keys())
        if not common_keys:
            return False

        # Check overlap in each shared dimension
        for key in common_keys:
            my_min, my_max = my_constraints[key]
            other_min, other_max = other_constraints[key]

            # Expand both ranges by threshold in both directions
            my_min_expanded = my_min - threshold
            my_max_expanded = my_max + threshold
            other_min_expanded = other_min - threshold
            other_max_expanded = other_max + threshold

            # If they don't overlap in this dimension, no intersection overall
            if my_max_expanded < other_min_expanded or other_max_expanded < my_min_expanded:
                return False

        # Overlap in all common dimensions
        return True
    


    def contains(self, other, threshold = 0.5):
        my_constraints = self.apply_domain_constraints()
        other_constraints = other.apply_domain_constraints()

        # To contain `other`, all parameters defined in `other` must be present
        # in `self` and `self`'s ranges must fully cover `other`'s ranges (within
        # the given threshold).
        for key, (other_min, other_max) in other_constraints.items():
            if key not in my_constraints:
                return False

            my_min, my_max = my_constraints[key]
            # Allow small tolerance: my_min should be <= other_min + threshold and
            # my_max should be >= other_max - threshold to consider containment.
            if my_min > other_min + threshold or my_max < other_max - threshold:
                return False

        return True



        

import numpy as np
import json
import yaml
from typing import List, Optional, Union

class Drone(ConfigurationSpace):
    """Drone configuration space with 6 parameters."""
    tasks = List[dict] 
    num_rows: int
    row_length: float
    name: str = "Default Name"
    description: str = "Default Description"
    def __init__(self,  
                domain_spec: Optional['DomainSpec'] = None, 
                drone_info: Optional[dict] = None,
                file_path: Optional[str] = None):
        super().__init__(domain_spec=domain_spec, file_path=file_path)
        # Validate inputs
        if file_path is None:
            if drone_info is None or domain_spec is None:
                raise ValueError(
                    "If 'file_path' is not provided, 'drone_info' and 'domain_spec' must be given."
                )
        else:
            if drone_info is not None or domain_spec is not None:
                raise ValueError(
                    "If 'file_path' is provided, 'drone_info' must not be given."
                )
            
        if file_path is not None:
           self._load_from_file(file_path)
        else:
            self.domain_spec = domain_spec
            self.num_rows = drone_info.get("num_rows", 25)
            self.row_length = drone_info.get("row_length", 150.0)
            self.tasks = self.create_field_tasks(self.num_rows, self.row_length)
            self.name = drone_info.get("name", "Default Name")
            self.description = drone_info.get("description", "Default Description")


    def _load_from_file(self, file_path: str) -> 'DomainSpec':
        """Load domain specification from a JSON or YAML file based on extension."""
        import os
        ext = os.path.splitext(file_path)[1].lower()

        match ext:
            case ".json":
                data = self._load_json(file_path)["drone"]
            case ".yaml" | ".yml":
                data = self._load_yaml(file_path)["drone"]
            case _:
                raise ValueError(f"Unsupported file extension: {ext}. Use .json or .yaml/.yml")
        
        self.domain_spec = self._create_domain_spec(data)
        self.num_rows = data['drone_info'].get("num_rows", 25)
        self.row_length = data['drone_info'].get("row_length", 150.0)
        self.tasks = self.create_field_tasks(self.num_rows, self.row_length)
        self.name = data['drone_info'].get("name", "Default Name")
        self.description = data['drone_info'].get("description", "Default Description")
        return self.domain_spec
    def size(self) -> int:
        return 1  # Single drone per configuration space
    def create_field_tasks(self,num_rows: int = 25, row_length: float = 150.0):
        """Create list of field row tasks."""
        tasks = []
        for i in range(num_rows):
            tasks.append({
                'id': i,
                'type': 'spray',
                'length': row_length,
                'priority': 1
            })
        return tasks

class Fleet(ConfigurationSpace):
    """
    Fleet configuration space with multiple drones.
    Loads either from a list of Drone objects or a JSON/YAML file describing them.
    """
    tasks = List[dict] 
    num_rows: int
    row_length: float
    name: str = "Default Name"
    description: str = "Default Description"

    def __init__(self, drones: Optional[List[Drone]] = None, file_path: Optional[str] = None):
        if drones is None and file_path is None:
            raise ValueError("Either 'drones' or 'file_path' must be provided.")
        if drones is not None and file_path is not None:
            raise ValueError("Provide only one of 'drones' or 'file_path', not both.")
        if file_path is not None:
            drones = self._load_fleet_file(file_path)

        self.drones = drones
        self.num_drones = len(drones)
        # Combine configuration spaces
        base_spec = getattr(self.drones[0], 'domain_spec', self.drones[0])
        bounds_stack = np.stack([d.domain_spec.get_bounds_array() for d in self.drones])
        mins = np.max(bounds_stack[:, :, 0], axis=0)
        maxs = np.min(bounds_stack[:, :, 1], axis=0)
        combined_bounds = np.column_stack([mins, maxs])

        combined_params = []
        for i, p in enumerate(base_spec.parameters):
            combined_params.append(
                Parameter(
                    p.name,
                    (float(combined_bounds[i, 0]), float(combined_bounds[i, 1])),
                    getattr(p, 'unit', None)
                )
            )

        combined_spec = DomainSpec(
            parameters=combined_params,
            constraint_functions=getattr(base_spec, 'constraint_functions', [])
        )
        super().__init__(domain_spec=combined_spec)

    def size(self) -> int:
        return self.num_drones
    def create_field_tasks(self,num_rows: int = 25, row_length: float = 150.0):
        """Create list of field row tasks."""
        tasks = []
        for i in range(num_rows):
            tasks.append({
                'id': i,
                'type': 'spray',
                'length': row_length,
                'priority': 1
            })
        return tasks
    def _load_fleet_file(self, file_path: str) -> List[Drone]:
        ext = os.path.splitext(file_path)[1].lower()
        match ext:
            case ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
            case ".yaml" | ".yml":
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
            case _:
                raise ValueError(f"Unsupported fleet file: {ext}")

        # load fleet information
        fleet_info = data.get("fleet_info", {})
        if not fleet_info:
            print("Warning: No fleet_info found in fleet file.")
        else:
            self.num_rows = fleet_info.get("num_rows", 25)
            self.row_length = fleet_info.get("row_length", 150.0)
            self.tasks = self.create_field_tasks(self.num_rows, self.row_length)
            self.name = fleet_info.get("name", "Default Name")
            self.description = fleet_info.get("description", "Default Description")
        

            


        # load drones
        drones = []

        for drone_data in data.get("drones", []):
            if isinstance(drone_data, str):
                # Drone is specified by file path
                drones.append(Drone(file_path=drone_data))
            elif isinstance(drone_data, dict):
                # Inline specification
                from src.domain import DomainSpec, Parameter, ParameterType
                params = [
                    Parameter(
                        name=p["name"],
                        bounds=tuple(p["bounds"]),
                        unit=p.get("unit", ""),
                        param_type=ParameterType(p.get("param_type", "continuous")),
                        description=p.get("description", "")
                    )
                    for p in drone_data.get("parameters", [])
                ]
                drones.append(
                    Drone(domain_spec=DomainSpec(parameters=params),
                          drone_info=drone_data.get("drone_info", {})
                    )
                )
            else:
                raise ValueError("Each drone must be a dict or a file path string.")
        return drones
