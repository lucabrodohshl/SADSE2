

from src.domain import DomainSpec, Parameter,  Configuration, ODD
import numpy as np
from dataclasses import dataclass


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
    
    def apply_constraints(self, bounds: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        """
        Apply drone-specific environmental constraints.
        
        Example constraints:
        - Wind reduces max speed
        - Temperature affects battery capacity
        - Visibility affects camera requirements
        """
        constrained = bounds.copy()
        param_names = domain_spec.parameter_names
        
        # Wind constraints (affects speed if present)
        if 'speed' in param_names and self.wind_ms > 0:
            speed_idx = param_names.index('speed')
            # Reduce max speed in high wind
            wind_factor = 0.5 * self.wind_ms
            constrained[speed_idx, 1] = max(
                constrained[speed_idx, 0],
                constrained[speed_idx, 1] - wind_factor
            )
        
        # Temperature constraints (affects battery if present)
        if 'battery_capacity' in param_names:
            battery_idx = param_names.index('battery_capacity')
            if self.temperature_c < 5:
                # Cold reduces battery capacity
                temp_factor = (5 - self.temperature_c) / 20.0
                constrained[battery_idx, 1] = max(
                    constrained[battery_idx, 0],
                    constrained[battery_idx, 1] * (1.0 - temp_factor)
                )
        
        # Visibility constraints (affects camera if present)
        if 'camera_res' in param_names and self.visibility_km < 5:
            camera_idx = param_names.index('camera_res')
            # Poor visibility requires higher resolution
            visibility_factor = (5.0 - self.visibility_km) / 5.0
            range_size = constrained[camera_idx, 1] - constrained[camera_idx, 0]
            constrained[camera_idx, 0] = min(
                constrained[camera_idx, 1],
                constrained[camera_idx, 0] + range_size * visibility_factor * 0.3
            )
        
        return constrained



class TightDroneODD(DroneODD):
    """
    DroneODD with MUCH TIGHTER constraints to create smaller DS regions.
    
    This makes each DS only 5-30% of CS instead of 90%+.
    """
    
    def apply_domain_constraints(self, domain_spec: DomainSpec) -> dict:
        """Apply TIGHT environmental constraints."""
        constraints = {}
        
        # AGGRESSIVE constraints based on weather
        wind = self.wind_ms
        temp = self.temperature_c
        vis = self.visibility_km
        hum = self.humidity_pct
        
        # Speed: heavily constrained by wind
        if wind > 12:
            constraints['speed'] = (3, 15)  # 26% of range
        elif wind > 8:
            constraints['speed'] = (5, 25)  # 43% of range
        elif wind > 5:
            constraints['speed'] = (10, 35)  # 53% of range
        else:
            constraints['speed'] = (15, 50)  # 74% of range
        
        # Altitude: constrained by visibility
        if vis < 6:
            constraints['altitude'] = (5, 40)  # 24% of range
        elif vis < 8:
            constraints['altitude'] = (10, 80)  # 48% of range
        else:
            constraints['altitude'] = (20, 150)  # 90% of range
        
        # Camera: constrained by visibility
        if vis < 6:
            constraints['camera_res'] = (2, 8)  # 27% of range
        elif vis < 8:
            constraints['camera_res'] = (4, 16)  # 55% of range
        else:
            constraints['camera_res'] = (8, 24)  # 73% of range
        
        # Spray rate: constrained by wind
        if wind > 10:
            constraints['spray_rate'] = (1, 5)  # 29% of range
        elif wind > 6:
            constraints['spray_rate'] = (2, 8)  # 50% of range
        else:
            constraints['spray_rate'] = (4, 15)  # 79% of range
        
        # Battery: constrained by temperature extremes
        if temp < 15 or temp > 30:
            constraints['battery_capacity'] = (10000, 20000)  # 59% of range
        elif temp < 18 or temp > 26:
            constraints['battery_capacity'] = (7000, 18000)  # 65% of range
        else:
            constraints['battery_capacity'] = (3000, 15000)  # 71% of range
        
        # Payload: constrained by wind + humidity
        if wind > 10 or hum > 70:
            constraints['payload_weight'] = (1, 5)  # 36% of range
        elif wind > 6 or hum > 60:
            constraints['payload_weight'] = (2, 8)  # 55% of range
        else:
            constraints['payload_weight'] = (4, 12)  # 73% of range
        
        return constraints
