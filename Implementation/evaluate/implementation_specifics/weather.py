from src.environment import Environment
from .domain_specific import TightDroneODD
import random

import math
import random
from typing import List, Dict, Optional

REPETITION_PROBABILITY = 0.3

def gamma_weather_abstraction(conditions: Dict[str, float], granularity: Dict[str, float] = None) -> Dict[str, float]:
    """
    Gamma function for weather abstraction - quantizes weather conditions
    to reduce small variations that don't meaningfully affect system behavior.
    
    Args:
        conditions: Raw weather conditions dict
        granularity: Dict specifying quantization step for each condition
        
    Returns:
        Quantized weather conditions dict
        
    Example:
        >>> conditions = {'wind': 14.23, 'temperature': 22.1, 'visibility': 9.87, 'humidity': 55.3}
        >>> gamma_weather_abstraction(conditions)
        {'wind': 14.0, 'temperature': 22.0, 'visibility': 10.0, 'humidity': 55.0}
    """
    if granularity is None:
        granularity = get_default_weather_granularity()
    
    abstracted = {}
    for key, value in conditions.items():
        if key in granularity:
            # Quantize: round to nearest multiple of granularity
            step = granularity[key]
            abstracted[key] = round(value / step) * step
        else:
            # Keep original value if not in granularity dict
            abstracted[key] = value
    
    return abstracted


def get_default_weather_granularity() -> Dict[str, float]:
    """Get default granularity values for weather abstraction."""
    return {
        'wind': 0.5,         # Round to nearest 0.5 m/s (14.2 -> 14.0, 14.3 -> 14.5)
        'temperature': 1.0,   # Round to nearest 1.0°C (22.1 -> 22.0, 22.7 -> 23.0)  
        'visibility': 0.5,    # Round to nearest 0.5 km (9.8 -> 10.0, 9.2 -> 9.0)
        'humidity': 2.0,      # Round to nearest 2% (55.3 -> 56.0, 53.8 -> 54.0)
    }


def get_coarse_weather_granularity() -> Dict[str, float]:
    """Get coarse granularity for aggressive weather abstraction."""
    return {
        'wind': 1.0,         # Round to nearest 1.0 m/s (more aggressive grouping)
        'temperature': 2.0,   # Round to nearest 2.0°C (14.1, 14.8 -> 14.0)
        'visibility': 1.0,    # Round to nearest 1.0 km
        'humidity': 5.0,      # Round to nearest 5% (54%, 56%, 58% -> 55%)
    }


def get_fine_weather_granularity() -> Dict[str, float]:
    """Get fine granularity for minimal weather abstraction."""
    return {
        'wind': 0.2,         # Round to nearest 0.2 m/s (more precision)
        'temperature': 0.5,   # Round to nearest 0.5°C
        'visibility': 0.2,    # Round to nearest 0.2 km
        'humidity': 1.0,      # Round to nearest 1%
    }

FIXED_DATA  = [
        # Initial mild conditions (wind 3-5)
        {'wind': 3.0, 'temperature': 22.0, 'visibility': 10.0, 'humidity': 55.0},  # 0
        {'wind': 3.5, 'temperature': 23.0, 'visibility': 9.8, 'humidity': 56.0},   # 1
        {'wind': 4.0, 'temperature': 21.0, 'visibility': 9.5, 'humidity': 57.0},   # 2
        {'wind': 3.0, 'temperature': 22.0, 'visibility': 10.0, 'humidity': 55.0},  # REPEAT 0
        {'wind': 4.5, 'temperature': 24.0, 'visibility': 9.3, 'humidity': 58.0},   # 3
        {'wind': 5.0, 'temperature': 25.0, 'visibility': 9.0, 'humidity': 60.0},   # 4
        {'wind': 3.5, 'temperature': 23.0, 'visibility': 9.8, 'humidity': 56.0},   # REPEAT 1
        
        # Moderate wind (wind 6-9)
        {'wind': 6.0, 'temperature': 20.0, 'visibility': 8.5, 'humidity': 62.0},   # 5
        {'wind': 6.5, 'temperature': 26.0, 'visibility': 8.3, 'humidity': 63.0},   # 6
        {'wind': 4.0, 'temperature': 21.0, 'visibility': 9.5, 'humidity': 57.0},   # REPEAT 2
        {'wind': 7.0, 'temperature': 19.0, 'visibility': 8.2, 'humidity': 64.0},   # 7
        {'wind': 7.5, 'temperature': 27.0, 'visibility': 8.0, 'humidity': 61.0},   # 8
        {'wind': 6.0, 'temperature': 20.0, 'visibility': 8.5, 'humidity': 62.0},   # REPEAT 5
        {'wind': 8.0, 'temperature': 18.0, 'visibility': 7.8, 'humidity': 65.0},   # 9
        {'wind': 8.5, 'temperature': 28.0, 'visibility': 7.7, 'humidity': 66.0},   # 10
        {'wind': 5.0, 'temperature': 25.0, 'visibility': 9.0, 'humidity': 60.0},   # REPEAT 4
        {'wind': 9.0, 'temperature': 17.0, 'visibility': 7.5, 'humidity': 67.0},   # 11
        {'wind': 9.5, 'temperature': 29.0, 'visibility': 7.3, 'humidity': 68.0},   # 12
        {'wind': 7.0, 'temperature': 19.0, 'visibility': 8.2, 'humidity': 64.0},   # REPEAT 7
        
        # High wind (wind 10-13)
        {'wind': 10.0, 'temperature': 16.0, 'visibility': 7.0, 'humidity': 50.0},  # 13
        {'wind': 10.5, 'temperature': 15.0, 'visibility': 6.8, 'humidity': 51.0},  # 14
        {'wind': 8.0, 'temperature': 18.0, 'visibility': 7.8, 'humidity': 65.0},   # REPEAT 9
        {'wind': 11.0, 'temperature': 14.0, 'visibility': 6.7, 'humidity': 52.0},  # 15
        {'wind': 11.5, 'temperature': 13.0, 'visibility': 6.5, 'humidity': 53.0},  # 16
        {'wind': 10.0, 'temperature': 16.0, 'visibility': 7.0, 'humidity': 50.0},  # REPEAT 13
        {'wind': 12.0, 'temperature': 25.0, 'visibility': 6.3, 'humidity': 60.0},  # 17
        {'wind': 12.5, 'temperature': 12.0, 'visibility': 6.2, 'humidity': 54.0},  # 18
        {'wind': 11.0, 'temperature': 14.0, 'visibility': 6.7, 'humidity': 52.0},  # REPEAT 15
        {'wind': 13.0, 'temperature': 19.0, 'visibility': 6.0, 'humidity': 68.0},  # 19
        
        # Strong wind (wind 14-16)
        {'wind': 14.0, 'temperature': 11.0, 'visibility': 5.8, 'humidity': 70.0},  # 20
        {'wind': 12.0, 'temperature': 25.0, 'visibility': 6.3, 'humidity': 60.0},  # REPEAT 17
        {'wind': 14.5, 'temperature': 10.0, 'visibility': 5.5, 'humidity': 71.0},  # 21
        {'wind': 15.0, 'temperature': 20.0, 'visibility': 5.3, 'humidity': 69.0},  # 22
        {'wind': 13.0, 'temperature': 19.0, 'visibility': 6.0, 'humidity': 68.0},  # REPEAT 19
        {'wind': 15.5, 'temperature': 9.0, 'visibility': 5.1, 'humidity': 72.0},   # 23

        # More repeats to show sustained cache effectiveness
        {'wind': 3.0, 'temperature': 22.0, 'visibility': 10.0, 'humidity': 55.0},  # REPEAT 0
        {'wind': 8.0, 'temperature': 18.0, 'visibility': 7.8, 'humidity': 65.0},   # REPEAT 9
        {'wind': 12.0, 'temperature': 25.0, 'visibility': 6.3, 'humidity': 60.0},  # REPEAT 17
        {'wind': 15.0, 'temperature': 20.0, 'visibility': 5.3, 'humidity': 69.0},  # REPEAT 22
        {'wind': 6.0, 'temperature': 20.0, 'visibility': 8.5, 'humidity': 62.0},   # REPEAT 5
        {'wind': 10.0, 'temperature': 16.0, 'visibility': 7.0, 'humidity': 50.0},  # REPEAT 13
        {'wind': 14.0, 'temperature': 11.0, 'visibility': 5.8, 'humidity': 70.0},  # REPEAT 20
        {'wind': 9.0, 'temperature': 17.0, 'visibility': 7.5, 'humidity': 67.0},   # REPEAT 11
        {'wind': 13.0, 'temperature': 19.0, 'visibility': 6.0, 'humidity': 68.0},  # REPEAT 19
        {'wind': 7.0, 'temperature': 19.0, 'visibility': 8.2, 'humidity': 64.0},   # REPEAT 7
        {'wind': 11.0, 'temperature': 14.0, 'visibility': 6.7, 'humidity': 52.0},  # REPEAT 15
        {'wind': 15.5, 'temperature': 9.0, 'visibility': 5.1, 'humidity': 72.0},   # REPEAT 23
        # Additional unique scenarios to populate the 10-hour run
        {'wind': 2.5, 'temperature': 21.0, 'visibility': 10.0, 'humidity': 54.0},  # extra A
        {'wind': 5.5, 'temperature': 26.5, 'visibility': 9.1, 'humidity': 59.0},   # extra B
        {'wind': 16.0, 'temperature': 12.0, 'visibility': 4.8, 'humidity': 74.0},  # extra C
        {'wind': 1.5, 'temperature': 30.0, 'visibility': 11.0, 'humidity': 40.0},  # extra D
        {'wind': 17.0, 'temperature': 10.0, 'visibility': 4.0, 'humidity': 75.0},  # extra E
    ]


class WeatherEnvironmentFixed(Environment):
    """A class representing the environment for weather evaluation."""
    duration_hours = 600  # 10 hours
    dt_minutes = 5        # 5-minute intervals
    seed = 42
    lat_deg = 52.5        # Berlin-ish; affects diurnal amplitude a bit
    day_of_year = 180  # midsummer default
    base_temp_c = 15.0    # seasonal mean near-surface temperature
    data_function : callable
    
    def __init__(self, name: str, data_function:str = None, seed: Optional[int] = None, 
                 weather_granularity: Dict[str, float] = None):
        """Initialize the environment with a name and optional data function."""
        self.seed = seed if seed is not None else 42
        self.name = name
        self.index = 0
        self.weather_granularity = weather_granularity  # Store granularity settings
        if data_function is None:
            self.data_function = self.get_fixed_data
        else:
            match data_function:
                case 'fixed':
                    # Pre-process fixed data with gamma abstraction
                    self.data = [gamma_weather_abstraction(data_point.copy(), self.weather_granularity) 
                                for data_point in FIXED_DATA]
                    self.data_function = self.get_fixed_data
                case 'realistic':
                    self.data = self.generate_realistic_data()
                    self.data_function = self.get_realistic_data
                    self.index = 0

                case _:
                    raise ValueError(f"Unknown data_function: {data_function}")
        
    def set_seed(self, seed: int):
        self.seed = seed
        random.seed(self.seed)
    def get_weather_data(self):
        """Method to retrieve weather data."""
        return self.data

    def get_fixed_data(self):
        # Return pre-processed data (already gamma abstracted during initialization)
        return random.choice(self.data).copy()
    
    def get_realistic_data(self):
        #occasionally repeat the previous data point to simulate persistence
        if random.random() < REPETITION_PROBABILITY: 
            print("Repeating previous data point")
            if self.index == 0:
                data_point = self.data[0]
            else:
                data_point = self.data[self.index - 1]
            return data_point.copy()

        if self.index >= len(self.data):
            self.index = 0  # Loop back to start if we exceed data length
        data_point = self.data[self.index]
        self.index += 1
        return data_point.copy()

    def step(self, timestamp):
        """Advance to the next weather data point."""     
        #random.seed(self.seed)   
        return  TightDroneODD(
            timestamp=timestamp,
            conditions=self.data_function()
        )
            

    def generate_realistic_data(self) -> List[Dict[str, float]]:
        """
        Generate a realistic sequence of weather states for a drone ODD.
        Returns a list of dicts with keys: wind [m/s], temperature [°C], visibility [km], humidity [%].

        Model:
        - Diurnal cycles for temperature, humidity, and wind (afternoons breezier).
        - AR(1) persistence to create smoothness.
        - Sporadic 'front' shocks altering wind/temperature/humidity for a few hours.
        - Visibility reduced by high humidity + low wind + random haze.
        """

        rng = random.Random(self.seed)

        # Time axis
        n_steps = int((self.duration_hours * 60) / self.dt_minutes)
        dt_hours = self.dt_minutes / 60.0

        # --- Helper diurnal functions ---
        # phase shifts for peak timing (local time)
        def diurnal_temp(hour_local: float) -> float:
            # Peak mid-afternoon, trough near sunrise; amplitude lightly varies with latitude/season
            amp = 6.0 * max(0.3, math.cos(math.radians(self.lat_deg))**2)  # 2–6 °C typical
            # sin with peak ~15:00 local
            return amp * math.sin(2 * math.pi * (hour_local - 15.0) / 24.0)

        def diurnal_wind(hour_local: float) -> float:
            # Breezier afternoons due to mixing; minimum early morning
            return 1.5 + 1.0 * (math.sin(2 * math.pi * (hour_local - 14.0) / 24.0) + 1.0)  # 0.5–3.5 m/s bump

        def diurnal_humidity(hour_local: float) -> float:
            # Inverse of temperature cycle (warmer air = lower RH)
            return -12.0 * math.sin(2 * math.pi * (hour_local - 15.0) / 24.0)

        # Simple seasonality tweak (warmer in summer)
        seasonal_temp_offset = 7.0 * math.sin(2 * math.pi * (self.day_of_year - 205) / 365.0)

        # --- State variables (initialize near climatology) ---
        temp = self.base_temp_c + seasonal_temp_offset + diurnal_temp(8.0)  # morning-ish
        hum = 60.0 + diurnal_humidity(8.0)                             # %
        wind = 3.0 + diurnal_wind(8.0)                                 # m/s

        # AR(1) coefficients (persistence)
        phi_t, phi_h, phi_w = 0.90, 0.92, 0.88

        # Shock std devs (process noise)
        sig_t, sig_h, sig_w = 0.8, 3.0, 0.9

        # Front process: when active, nudges wind/temp/humidity for several hours
        front_active = False
        front_time_left = 0
        front_temp_push = 0.0
        front_hum_push = 0.0
        front_wind_push = 0.0

        def maybe_start_front():
            nonlocal front_active, front_time_left, front_temp_push, front_hum_push, front_wind_push
            # Low probability each hour to start a front (e.g., every ~2 days on average)
            if rng.random() < dt_hours * (1.0 / 48.0):  # ~1 every 48h
                front_active = True
                # Duration 3–12 hours
                front_time_left = rng.randint(max(3, int(3/dt_hours)), max(12, int(12/dt_hours)))
                # Cool or warm front
                sign = -1.0 if rng.random() < 0.55 else 1.0
                front_temp_push = sign * rng.uniform(1.5, 4.0)  # °C
                front_hum_push = (-sign) * rng.uniform(5.0, 12.0)  # inverse relation
                front_wind_push = rng.uniform(1.0, 4.0)            # fronts often windier

        data = []

        hour_local = 8.0  # arbitrary start-of-day hour
        for step in range(n_steps):
            # Possibly start a front
            if not front_active:
                maybe_start_front()

            # Diurnal means
            mu_temp = self.base_temp_c + seasonal_temp_offset + diurnal_temp(hour_local)
            mu_hum = 60.0 + diurnal_humidity(hour_local)
            mu_wind = 3.0 + diurnal_wind(hour_local)

            # Apply front offsets if active
            if front_active:
                mu_temp += front_temp_push
                mu_hum += front_hum_push
                mu_wind += front_wind_push
                front_time_left -= 1
                if front_time_left <= 0:
                    front_active = False
                    front_temp_push = front_hum_push = front_wind_push = 0.0

            # AR(1) updates with noise
            temp = mu_temp + phi_t * (temp - mu_temp) + rng.gauss(0.0, sig_t)
            hum  = mu_hum + phi_h * (hum  - mu_hum) + rng.gauss(0.0, sig_h)
            wind = mu_wind + phi_w * (wind - mu_wind) + rng.gauss(0.0, sig_w)

            # Physical/realistic bounds
            temp = max(-20.0, min(40.0, temp))
            hum  = max(5.0, min(100.0, hum))
            wind = max(0.0, min(22.0, wind))  # cap at ~stormy 22 m/s

            # Visibility model [km]:
            # Base clear-air vis, minus losses from high humidity & low wind, plus small noise.
            # Drop sharply when humidity is very high; improve with wind (dispersion).
            haze = max(0.0, 2.5 * math.exp(-max(0.0, wind) / 3.0))      # calm air → more haze
            rh_term = 8.0 * (hum / 100.0) ** 2                           # strong at high RH
            base_clear = 18.0                                            # km
            vis = base_clear - rh_term - haze + rng.gauss(0.0, 0.4)
            vis = max(0.3, min(25.0, vis))  # clamp (dense fog .. very clear)

            # Create raw data point
            raw_data_point = {
                "wind": round(wind, 1),
                "temperature": round(temp, 1),
                "visibility": round(vis, 1),
                "humidity": round(hum, 1),
            }
            
            # Apply gamma abstraction to reduce small variations
            abstracted_data_point = gamma_weather_abstraction(raw_data_point, self.weather_granularity)
            data.append(abstracted_data_point)

            hour_local = (hour_local + dt_hours) % 24.0
        
        return data
