"""
SHIRO Covariance Quality Assessment Module

Provides object-specific covariance inflation factors based on
orbital classification and propagation time.

Planned upgrade path following:
    Moncayo et al. (2024) - GSOC SP-COV synthetic covariance methodology
    Journal of Space Safety Engineering

Current implementation uses literature-based defaults from:
    Hejduk et al. (2004) - operational covariance underconfidence 3-5x
    Carpenter et al. (2018) - covariance realism for SSA

Future: replace defaults with coefficients fitted from real CDM archives
following the GSOC SP-COV classification scheme.
"""

from __future__ import annotations

import numpy as np


def classify_object(
    perigee_alt_km: float,
    eccentricity: float,
    inclination_deg: float,
    object_size_m2: float = 1.0,
    solar_flux_f107: float = 150.0,
    propagation_time_days: float = 1.0,
) -> str:
    _ = inclination_deg, solar_flux_f107, propagation_time_days

    if perigee_alt_km < 350:
        alt_class = "VLEO"
    elif perigee_alt_km < 550:
        alt_class = "LEO_LOW"
    elif perigee_alt_km < 800:
        alt_class = "LEO_MID"
    elif perigee_alt_km < 2000:
        alt_class = "LEO_HIGH"
    elif perigee_alt_km < 25000:
        alt_class = "MEO"
    else:
        alt_class = "GEO"

    ecc_class = "CIRCULAR" if eccentricity < 0.1 else "ECCENTRIC"

    if object_size_m2 < 0.1:
        size_class = "SMALL"
    elif object_size_m2 < 1.0:
        size_class = "MEDIUM"
    else:
        size_class = "LARGE"

    return f"{alt_class}_{ecc_class}_{size_class}"


def estimate_inflation_factor(
    perigee_alt_km: float,
    eccentricity: float,
    inclination_deg: float,
    object_size_m2: float = 1.0,
    solar_flux_f107: float = 150.0,
    propagation_time_days: float = 1.0,
    is_active_satellite: bool = True,
) -> tuple[float, float, str]:
    class_label = classify_object(
        perigee_alt_km,
        eccentricity,
        inclination_deg,
        object_size_m2,
        solar_flux_f107,
        propagation_time_days,
    )

    base_factors = {
        "VLEO": {"active": 16.0, "debris": 25.0},
        "LEO_LOW": {"active": 9.0, "debris": 16.0},
        "LEO_MID": {"active": 9.0, "debris": 16.0},
        "LEO_HIGH": {"active": 9.0, "debris": 16.0},
        "MEO": {"active": 9.0, "debris": 9.0},
        "GEO": {"active": 9.0, "debris": 9.0},
    }

    alt_key = class_label.split("_")[0]
    if alt_key == "LEO" and "LOW" in class_label:
        alt_key = "LEO_LOW"
    elif alt_key == "LEO" and "MID" in class_label:
        alt_key = "LEO_MID"
    elif alt_key == "LEO" and "HIGH" in class_label:
        alt_key = "LEO_HIGH"

    obj_type = "active" if is_active_satellite else "debris"
    k_squared = base_factors.get(alt_key, {}).get(obj_type, 16.0)

    time_factor = 1.0 + 0.1 * propagation_time_days**2
    k_squared_scaled = min(k_squared * time_factor, 25.0)
    k = float(np.sqrt(k_squared_scaled))

    return float(k_squared_scaled), k, "literature_default"


def assess_covariance_realism(
    cdm_covariance_km2: np.ndarray,
    perigee_alt_km: float,
    eccentricity: float,
    inclination_deg: float,
    object_size_m2: float,
    propagation_time_days: float,
    is_active_satellite: bool = True,
) -> dict:
    k_sq, k, confidence = estimate_inflation_factor(
        perigee_alt_km,
        eccentricity,
        inclination_deg,
        object_size_m2,
        propagation_time_days=propagation_time_days,
        is_active_satellite=is_active_satellite,
    )

    pos_cov = cdm_covariance_km2[:3, :3]
    sigma_reported = float(np.sqrt(np.mean(np.diag(pos_cov))))

    return {
        "sigma_reported_km": sigma_reported,
        "recommended_inflation_k2": k_sq,
        "recommended_k": k,
        "confidence": confidence,
        "orbital_class": classify_object(
            perigee_alt_km,
            eccentricity,
            inclination_deg,
            object_size_m2,
            propagation_time_days=propagation_time_days,
        ),
        "note": "Pending real CDM archive fitting per Moncayo et al. 2024",
    }


def get_inflation_for_event(
    miss_distance_m: float,
    altitude_km: float,
    lead_time_hours: float,
    is_active: bool = True,
) -> float:
    _ = miss_distance_m
    k_sq, _, _ = estimate_inflation_factor(
        perigee_alt_km=altitude_km,
        eccentricity=0.001,
        inclination_deg=53.0,
        object_size_m2=1.0,
        propagation_time_days=lead_time_hours / 24.0,
        is_active_satellite=is_active,
    )
    return float(k_sq)
