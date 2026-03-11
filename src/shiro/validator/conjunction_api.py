from __future__ import annotations

from threading import Lock
from time import monotonic

import numpy as np
from fastapi import FastAPI

from .cdm_from_stresslab import generate_synthetic_cdm


app = FastAPI(title="SHIRO Conjunction API", version="0.1.0")
_CACHE_LOCK = Lock()
_CACHE_TTL_SECONDS = 300.0
_CACHE_VALUE: dict | None = None
_CACHE_AT = 0.0


def _build_orbit_points(data: dict) -> dict[str, list[list[float]]]:
    traj1 = np.asarray(data["traj1_km"], dtype=float)
    traj2 = np.asarray(data["traj2_km"], dtype=float)
    return {
        "primary": (traj1[:, :3] * 1000.0).tolist(),
        "secondary": (traj2[:, :3] * 1000.0).tolist(),
    }


@app.get("/api/conjunction/latest")
def conjunction_latest() -> dict:
    global _CACHE_VALUE, _CACHE_AT
    now = monotonic()
    if _CACHE_VALUE is not None and (now - _CACHE_AT) < _CACHE_TTL_SECONDS:
        return _CACHE_VALUE

    event_data, _, _ = generate_synthetic_cdm()
    miss_point = [float(event_data["miss_2d_m"][0]), float(event_data["miss_2d_m"][1])]
    c_bplane = np.asarray(event_data["C_bplane_m2"], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(c_bplane)
    eigvals = np.clip(eigvals, 1e-18, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ellipse_points: list[list[float]] = []
    for a in theta:
        p = np.array([3.0 * np.sqrt(eigvals[0]) * np.cos(a), 3.0 * np.sqrt(eigvals[1]) * np.sin(a)])
        q = eigvecs @ p + np.asarray(miss_point)
        ellipse_points.append([float(q[0]), float(q[1])])

    hbr = float(event_data["hbr_m"])
    hbr_circle = [[float(hbr * np.cos(a)), float(hbr * np.sin(a))] for a in theta]

    traj1 = np.asarray(event_data["traj1_km"], dtype=float)
    traj2 = np.asarray(event_data["traj2_km"], dtype=float)
    tca_idx = int(event_data["tca_idx"])
    conjunction_position_m = (((traj1[tca_idx, :3] + traj2[tca_idx, :3]) / 2.0) * 1000.0).tolist()
    orbit_points = _build_orbit_points(event_data)

    timeline = event_data["pc_timeline"]
    time_to_tca_h = [float(p["time_to_tca_hours"]) for p in timeline]
    pc_reference = [float(p["pc"]) for p in timeline]
    pc_degraded = [max(v * 1.3, 1e-16) for v in pc_reference]

    payload = {
        "event": event_data,
        "orbit_points": orbit_points,
        "conjunction_point_eci_m": [float(x) for x in conjunction_position_m],
        "b_plane": {
            "miss_point_m": miss_point,
            "covariance_ellipse_m": ellipse_points,
            "hbr_circle_m": hbr_circle,
            "pc": float(event_data["pc"]),
            "miss_distance_m": float(event_data["miss_distance_m"]),
        },
        "pc_timeline": {
            "time_to_tca_hours": time_to_tca_h,
            "pc_reference": pc_reference,
            "pc_degraded": pc_degraded,
            "threshold": 1e-4,
        },
    }

    with _CACHE_LOCK:
        _CACHE_VALUE = payload
        _CACHE_AT = monotonic()
    return payload
