"""Track-specific data for F1 race analysis."""

from typing import Dict, Optional
from dataclasses import dataclass

# ============================================================================
# Safety Car (SC) Probabilities by Track
# ============================================================================

TRACK_SC_PROBABILITIES = {
    "Bahrain": 0.25,  # Relatively safe, modern track
    "Saudi Arabian": 0.35,  # Street circuit, some incidents
    "Australian": 0.30,  # Mixed conditions possible
    "Japanese": 0.28,  # Well-maintained track
    "Chinese": 0.32,  # Wet weather risk
    "Miami": 0.38,  # Street circuit, barrier contact common
    "Emilia Romagna": 0.35,  # Medium incident rate
    "Monaco": 0.75,  # Tight street circuit, very high incident rate
    "Canadian": 0.40,  # Street circuit, weather dependent
    "Spanish": 0.28,  # Modern, safe design
    "Austrian": 0.25,  # High-speed, generally safe
    "British": 0.35,  # Variable weather
    "Hungarian": 0.32,  # Tight circuit
    "Belgian": 0.28,  # High-speed, weather dependent
    "Dutch": 0.26,  # Generally safe
    "Italian": 0.15,  # High-speed, very safe (Monza)
    "Azerbaijan": 0.45,  # Street circuit, narrow sections
    "Singapore": 0.65,  # Night race, difficult visibility
    "United States": 0.33,  # Austin circuit
    "Mexico": 0.35,  # High altitude, tight circuit
    "São Paulo": 0.40,  # Weather dependent, street-like
    "Las Vegas": 0.50,  # Street circuit, many incidents
    "Qatar": 0.28,  # Modern, safe design
    "Abu Dhabi": 0.30,  # Modern, mostly safe
}

# ============================================================================
# Fuel Consumption Factors by Track
# ============================================================================
# Relative to Monaco baseline (1.0)
# Higher values = more fuel consumption per lap
# Depends on: average speed, fuel load, engine usage, etc.

TRACK_FUEL_CONSUMPTION_FACTORS = {
    "Bahrain": 1.15,  # Medium fuel consumption
    "Saudi Arabian": 1.20,  # High-speed street circuit
    "Australian": 1.10,  # Moderate
    "Japanese": 1.08,  # Moderate
    "Chinese": 1.12,  # Moderate-high
    "Miami": 1.15,  # Street circuit, moderate consumption
    "Emilia Romagna": 1.12,  # Medium
    "Monaco": 1.00,  # Baseline (slow, low consumption)
    "Canadian": 1.13,  # Street circuit, high consumption
    "Spanish": 1.18,  # Fast track, high consumption
    "Austrian": 1.22,  # Fast, high consumption
    "British": 1.20,  # Fast, high fuel burn
    "Hungarian": 1.11,  # Tight, medium consumption
    "Belgian": 1.25,  # Very fast (Spa), high consumption
    "Dutch": 1.23,  # Fast track, high consumption
    "Italian": 1.40,  # High-speed Monza, very high consumption
    "Azerbaijan": 1.17,  # Street circuit, mixed speeds
    "Singapore": 1.08,  # Night race, tight, low consumption
    "United States": 1.16,  # Austin, moderate-high
    "Mexico": 1.14,  # High altitude, tight circuit
    "São Paulo": 1.13,  # Medium-high
    "Las Vegas": 1.15,  # Street circuit
    "Qatar": 1.18,  # Fast track
    "Abu Dhabi": 1.19,  # Fast track, high consumption
}

# ============================================================================
# Tire Degradation Profiles by Track
# ============================================================================
# Defines how much pace a driver loses lap-by-lap for different tire compounds
# Values represent pace loss (seconds per lap) relative to lap 1


@dataclass
class TireDegradationProfile:
    """Profile for tire degradation on a specific track."""

    soft_degradation: float  # Pace loss per lap (soft compound)
    medium_degradation: float  # Pace loss per lap (medium compound)
    hard_degradation: float  # Pace loss per lap (hard compound)

    # Degradation increases with stint progress
    degradation_curve_exponent: float = 1.5  # 1.0 = linear, >1.0 = accelerating


TRACK_TIRE_DEGRADATION = {
    "Bahrain": TireDegradationProfile(
        soft_degradation=0.045,
        medium_degradation=0.035,
        hard_degradation=0.025,
    ),
    "Saudi Arabian": TireDegradationProfile(
        soft_degradation=0.050,
        medium_degradation=0.040,
        hard_degradation=0.028,
    ),
    "Australian": TireDegradationProfile(
        soft_degradation=0.048,
        medium_degradation=0.038,
        hard_degradation=0.026,
    ),
    "Japanese": TireDegradationProfile(
        soft_degradation=0.042,
        medium_degradation=0.032,
        hard_degradation=0.022,
    ),
    "Chinese": TireDegradationProfile(
        soft_degradation=0.046,
        medium_degradation=0.036,
        hard_degradation=0.024,
    ),
    "Miami": TireDegradationProfile(
        soft_degradation=0.052,
        medium_degradation=0.042,
        hard_degradation=0.030,
    ),
    "Emilia Romagna": TireDegradationProfile(
        soft_degradation=0.048,
        medium_degradation=0.038,
        hard_degradation=0.026,
    ),
    "Monaco": TireDegradationProfile(
        soft_degradation=0.032,
        medium_degradation=0.025,
        hard_degradation=0.018,
    ),
    "Canadian": TireDegradationProfile(
        soft_degradation=0.050,
        medium_degradation=0.040,
        hard_degradation=0.028,
    ),
    "Spanish": TireDegradationProfile(
        soft_degradation=0.055,
        medium_degradation=0.045,
        hard_degradation=0.032,
    ),
    "Austrian": TireDegradationProfile(
        soft_degradation=0.058,
        medium_degradation=0.048,
        hard_degradation=0.035,
    ),
    "British": TireDegradationProfile(
        soft_degradation=0.056,
        medium_degradation=0.046,
        hard_degradation=0.033,
    ),
    "Hungarian": TireDegradationProfile(
        soft_degradation=0.051,
        medium_degradation=0.041,
        hard_degradation=0.029,
    ),
    "Belgian": TireDegradationProfile(
        soft_degradation=0.060,
        medium_degradation=0.050,
        hard_degradation=0.037,
    ),
    "Dutch": TireDegradationProfile(
        soft_degradation=0.057,
        medium_degradation=0.047,
        hard_degradation=0.034,
    ),
    "Italian": TireDegradationProfile(
        soft_degradation=0.065,
        medium_degradation=0.055,
        hard_degradation=0.040,
    ),
    "Azerbaijan": TireDegradationProfile(
        soft_degradation=0.049,
        medium_degradation=0.039,
        hard_degradation=0.027,
    ),
    "Singapore": TireDegradationProfile(
        soft_degradation=0.044,
        medium_degradation=0.034,
        hard_degradation=0.023,
    ),
    "United States": TireDegradationProfile(
        soft_degradation=0.052,
        medium_degradation=0.042,
        hard_degradation=0.030,
    ),
    "Mexico": TireDegradationProfile(
        soft_degradation=0.047,
        medium_degradation=0.037,
        hard_degradation=0.025,
    ),
    "São Paulo": TireDegradationProfile(
        soft_degradation=0.050,
        medium_degradation=0.040,
        hard_degradation=0.028,
    ),
    "Las Vegas": TireDegradationProfile(
        soft_degradation=0.051,
        medium_degradation=0.041,
        hard_degradation=0.029,
    ),
    "Qatar": TireDegradationProfile(
        soft_degradation=0.053,
        medium_degradation=0.043,
        hard_degradation=0.031,
    ),
    "Abu Dhabi": TireDegradationProfile(
        soft_degradation=0.054,
        medium_degradation=0.044,
        hard_degradation=0.032,
    ),
}

# ============================================================================
# Track Characteristics
# ============================================================================


@dataclass
class TrackCharacteristics:
    """Characteristics of an F1 track."""

    circuit_type: str  # "Road course", "Street circuit", "Permanent"
    lap_length_km: float  # Circuit length in km
    drs_zones: int  # Number of DRS zones
    typical_pit_stop_time_s: float  # Typical pit stop duration
    fuel_tank_capacity_kg: float  # Fuel tank capacity in kg
    avg_fuel_consumption_kg_per_km: float  # Typical consumption


TRACK_CHARACTERISTICS = {
    "Bahrain": TrackCharacteristics(
        circuit_type="Road course",
        lap_length_km=5.412,
        drs_zones=3,
        typical_pit_stop_time_s=20.5,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=1.23,
    ),
    "Saudi Arabian": TrackCharacteristics(
        circuit_type="Street circuit",
        lap_length_km=6.174,
        drs_zones=3,
        typical_pit_stop_time_s=22.0,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=1.35,
    ),
    "Monaco": TrackCharacteristics(
        circuit_type="Street circuit",
        lap_length_km=3.337,
        drs_zones=0,
        typical_pit_stop_time_s=23.0,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=0.85,
    ),
    "Italian": TrackCharacteristics(
        circuit_type="Road course",
        lap_length_km=5.793,
        drs_zones=4,
        typical_pit_stop_time_s=21.0,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=1.60,
    ),
    "Singapore": TrackCharacteristics(
        circuit_type="Street circuit",
        lap_length_km=5.065,
        drs_zones=2,
        typical_pit_stop_time_s=23.5,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=1.15,
    ),
    "British": TrackCharacteristics(
        circuit_type="Road course",
        lap_length_km=5.891,
        drs_zones=3,
        typical_pit_stop_time_s=20.5,
        fuel_tank_capacity_kg=110,
        avg_fuel_consumption_kg_per_km=1.40,
    ),
}


def get_safety_car_probability(track_name: str) -> float:
    """Get Safety Car probability for a track.

    Args:
        track_name: Official track name

    Returns:
        Probability of Safety Car (0.0 to 1.0)
    """
    return TRACK_SC_PROBABILITIES.get(track_name, 0.35)  # Default: 0.35


def get_fuel_consumption_factor(track_name: str) -> float:
    """Get fuel consumption factor for a track.

    Args:
        track_name: Official track name

    Returns:
        Fuel consumption factor (relative to baseline)
    """
    return TRACK_FUEL_CONSUMPTION_FACTORS.get(track_name, 1.15)


def get_tire_degradation(track_name: str) -> Optional[TireDegradationProfile]:
    """Get tire degradation profile for a track.

    Args:
        track_name: Official track name

    Returns:
        TireDegradationProfile or None if not found
    """
    return TRACK_TIRE_DEGRADATION.get(track_name)


def get_track_characteristics(track_name: str) -> Optional[TrackCharacteristics]:
    """Get characteristics for a track.

    Args:
        track_name: Official track name

    Returns:
        TrackCharacteristics or None if not found
    """
    return TRACK_CHARACTERISTICS.get(track_name)
