from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


MU = 398600.4418
RE = 6378.137
J2 = 1.08263e-3
J3 = -2.53266e-6
J4 = -1.61963e-6
J5 = -2.27732e-7
J6 = 5.40681e-7
RHO0 = 1.225e-9
H_SCALE = 8.5
CD = 2.2
AREA_MASS = 0.01

_PROPAGATOR_BANNER_PRINTED = False


def atmospheric_density(altitude_km: float) -> float:
    alt_bands = [
        (0.0, 25.0, 1.225e-9, 7.249),
        (25.0, 30.0, 3.899e-11, 6.349),
        (30.0, 40.0, 1.774e-11, 6.682),
        (40.0, 50.0, 3.972e-12, 7.554),
        (50.0, 60.0, 1.057e-12, 8.382),
        (60.0, 70.0, 3.206e-13, 7.714),
        (70.0, 80.0, 8.770e-14, 6.549),
        (80.0, 90.0, 1.905e-14, 5.799),
        (90.0, 100.0, 3.396e-15, 5.382),
        (100.0, 110.0, 5.297e-16, 5.877),
        (110.0, 120.0, 9.661e-17, 7.263),
        (120.0, 130.0, 2.438e-17, 9.473),
        (130.0, 140.0, 8.484e-18, 12.636),
        (140.0, 150.0, 3.845e-18, 16.149),
        (150.0, 180.0, 2.070e-18, 22.523),
        (180.0, 200.0, 5.464e-19, 29.740),
        (200.0, 250.0, 2.789e-19, 37.105),
        (250.0, 300.0, 7.248e-20, 45.546),
        (300.0, 350.0, 2.418e-20, 53.628),
        (350.0, 400.0, 9.518e-21, 53.298),
        (400.0, 450.0, 3.725e-21, 58.515),
        (450.0, 500.0, 1.585e-21, 60.828),
        (500.0, 600.0, 6.967e-22, 63.822),
        (600.0, 700.0, 1.454e-22, 71.835),
        (700.0, 800.0, 3.614e-23, 88.667),
        (800.0, 900.0, 1.170e-23, 124.64),
        (900.0, 1000.0, 5.245e-24, 181.05),
        (1000.0, 1e9, 3.019e-24, 268.0),
    ]
    for h_min, h_max, rho_ref, h_scale in alt_bands:
        if altitude_km <= h_max:
            return float(rho_ref * np.exp(-(altitude_km - h_min) / h_scale))
    return float(3.019e-24 * np.exp(-(altitude_km - 1000.0) / 268.0))


def accel_j2_drag(
    state: np.ndarray,
    area_mass_ratio: float = AREA_MASS,
    cd: float = CD,
    include_drag: bool = True,
) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    r2 = r**2
    r3 = r**3
    r5 = r**5
    r7 = r**7

    ax = -MU * x / r3
    ay = -MU * y / r3
    az = -MU * z / r3

    z2_r2 = z**2 / r2
    j2_factor = 1.5 * J2 * MU * RE**2 / r5
    ax += j2_factor * x * (1.0 - 5.0 * z2_r2)
    ay += j2_factor * y * (1.0 - 5.0 * z2_r2)
    az += j2_factor * z * (3.0 - 5.0 * z2_r2)

    j3_factor = 2.5 * J3 * MU * RE**3 / r7
    ax += j3_factor * x * (3.0 * z - 7.0 * z**3 / r2)
    ay += j3_factor * y * (3.0 * z - 7.0 * z**3 / r2)
    az += j3_factor * (6.0 * z**2 - 7.0 * z**4 / r2 - 3.0 * r2 / 5.0)

    if include_drag:
        altitude_km = r - RE
        if altitude_km < 1000.0:
            rho = atmospheric_density(altitude_km)
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            if v_mag > 0:
                am_km = area_mass_ratio * 1e-6
                a_drag_mag = -0.5 * cd * am_km * rho * v_mag**2
                ax += a_drag_mag * vx / v_mag
                ay += a_drag_mag * vy / v_mag
                az += a_drag_mag * vz / v_mag

    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def propagate_j2_drag(
    state0: np.ndarray,
    dt: float,
    steps: int,
    area_mass_ratio: float = AREA_MASS,
    cd: float = CD,
    include_drag: bool = True,
) -> np.ndarray:
    traj = np.zeros((steps + 1, 6), dtype=float)
    traj[0] = state0.copy()
    state = state0.copy()
    for i in range(steps):
        k1 = accel_j2_drag(state, area_mass_ratio, cd, include_drag)
        k2 = accel_j2_drag(state + 0.5 * dt * k1, area_mass_ratio, cd, include_drag)
        k3 = accel_j2_drag(state + 0.5 * dt * k2, area_mass_ratio, cd, include_drag)
        k4 = accel_j2_drag(state + dt * k3, area_mass_ratio, cd, include_drag)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i + 1] = state
    return traj


def propagate_cov_simple(p0: np.ndarray, t_seconds: float) -> np.ndarray:
    q = 1e-10
    return p0 + np.eye(6, dtype=float) * q * float(t_seconds)


def compute_pc_bplane(
    state1: np.ndarray,
    state2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    hbr_km: float,
    n_samples: int = 50000,
    seed: int = 42,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rel_pos = state2[:3] - state1[:3]
    rel_vel = state2[3:] - state1[3:]

    v_hat = rel_vel / np.linalg.norm(rel_vel)
    h = np.cross(rel_pos, rel_vel)
    h_hat = h / np.linalg.norm(h)
    eta_hat = np.cross(v_hat, h_hat)
    zeta_hat = h_hat

    t_proj = np.array([eta_hat, zeta_hat], dtype=float)
    t6 = np.zeros((2, 6), dtype=float)
    t6[:, :3] = t_proj

    c_combined = cov1[:6, :6] + cov2[:6, :6]
    c_bplane_computed = t6 @ c_combined @ t6.T

    sigma_intrack_m = 200.0
    sigma_crosstrack_m = 50.0
    min_cov_m2 = np.diag([sigma_intrack_m**2, sigma_crosstrack_m**2])
    min_cov_km2 = min_cov_m2 / 1_000_000.0
    c_bplane = np.maximum(c_bplane_computed, min_cov_km2)

    miss_eta = float(np.dot(rel_pos, eta_hat))
    miss_zeta = float(np.dot(rel_pos, zeta_hat))
    miss_2d = np.array([miss_eta, miss_zeta], dtype=float)

    eigvals = np.linalg.eigvalsh(c_bplane)
    if np.min(eigvals) <= 0:
        c_bplane = c_bplane + np.eye(2, dtype=float) * (abs(np.min(eigvals)) + 1e-12)

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(miss_2d, c_bplane, n_samples)
    in_hbr = np.linalg.norm(samples, axis=1) < float(hbr_km)
    pc = float(np.sum(in_hbr)) / float(n_samples)
    return pc, miss_2d, c_bplane, samples


def build_realistic_miss_vector(
    r1_tca: np.ndarray,
    v1_tca: np.ndarray,
    v2_tca: np.ndarray,
    miss_distance_km: float,
    along_track_fraction: float = 0.7,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    _ = rng

    rel_vel = v2_tca - v1_tca
    rel_vel_norm = rel_vel / np.linalg.norm(rel_vel)

    r_norm = r1_tca / np.linalg.norm(r1_tca)
    along_track = np.cross(r_norm, np.cross(r_norm, rel_vel_norm))
    if np.linalg.norm(along_track) > 1e-10:
        along_track = along_track / np.linalg.norm(along_track)
    else:
        along_track = np.cross(rel_vel_norm, np.array([0.0, 0.0, 1.0], dtype=float))
        along_track = along_track / np.linalg.norm(along_track)

    cross_track = np.cross(rel_vel_norm, along_track)
    if np.linalg.norm(cross_track) > 1e-10:
        cross_track = cross_track / np.linalg.norm(cross_track)
    else:
        cross_track = np.cross(rel_vel_norm, np.array([0.0, 1.0, 0.0], dtype=float))
        cross_track = cross_track / np.linalg.norm(cross_track)

    cross_track_fraction = 1.0 - along_track_fraction
    miss_vec = along_track_fraction * along_track + cross_track_fraction * cross_track
    miss_vec = miss_vec / np.linalg.norm(miss_vec)
    return miss_vec * miss_distance_km


def build_realistic_covariance(
    sigma_combined_km: float,
    lead_time_hours: float,
    altitude_km: float,
) -> tuple[np.ndarray, float, float, float]:
    lead_time_days = lead_time_hours / 24.0
    at_scale = min(10.0 + 5.0 * lead_time_days, 40.0)

    if altitude_km < 400.0:
        alt_factor = 1.5
    elif altitude_km < 600.0:
        alt_factor = 1.0
    else:
        alt_factor = 0.7

    sigma_radial = sigma_combined_km * 0.2 * alt_factor
    sigma_intrack = sigma_combined_km * at_scale * 0.1 * alt_factor
    sigma_crosstrack = sigma_combined_km * 0.25 * alt_factor

    sigma_radial = max(sigma_radial, 50.0 / 1000.0)
    sigma_intrack = max(sigma_intrack, 200.0 / 1000.0)
    sigma_crosstrack = max(sigma_crosstrack, 50.0 / 1000.0)

    c = np.diag(
        [
            (sigma_radial / 2.0) ** 2,
            (sigma_intrack / 2.0) ** 2,
            (sigma_crosstrack / 2.0) ** 2,
            1e-8,
            1e-8,
            1e-8,
        ]
    )
    return c, sigma_radial, sigma_intrack, sigma_crosstrack


def generate_conjunction_from_geometry(
    miss_distance_m: float = 150.0,
    rel_velocity_mps: float = 11000.0,
    inclination_primary_deg: float = 51.6,
    inclination_secondary_deg: float = 97.8,
    altitude_km: float = 400.0,
    target_pc: float = 3e-4,
    hbr_m: float = 20.0,
    lead_time_hours: float = 4.0,
    seed: int = 42,
    include_drag: bool | None = None,
) -> dict:
    global _PROPAGATOR_BANNER_PRINTED

    if include_drag is None:
        include_drag = altitude_km < 600.0

    if not _PROPAGATOR_BANNER_PRINTED:
        print("Propagator upgraded: J2 + J3 + drag active")
        _PROPAGATOR_BANNER_PRINTED = True

    r_mag = RE + altitude_km
    v_circ = np.sqrt(MU / r_mag)

    inc1 = np.radians(inclination_primary_deg)
    r1_tca = np.array([r_mag, 0.0, 0.0], dtype=float)
    v1_tca = np.array([0.0, v_circ * np.cos(inc1), v_circ * np.sin(inc1)], dtype=float)

    angle_diff = np.radians(inclination_secondary_deg - inclination_primary_deg)
    v2_mag = np.sqrt(MU / r_mag)
    v2_tca = np.array(
        [
            0.0,
            v2_mag * np.cos(np.pi - angle_diff),
            v2_mag * np.sin(np.pi - angle_diff),
        ],
        dtype=float,
    )

    miss_km = miss_distance_m / 1000.0
    miss_dir = build_realistic_miss_vector(
        r1_tca,
        v1_tca,
        v2_tca,
        miss_km,
        along_track_fraction=0.7,
    )
    r2_tca = r1_tca + miss_dir

    s1_tca = np.concatenate([r1_tca, v1_tca])
    s2_tca = np.concatenate([r2_tca, v2_tca])

    actual_rel_vel = float(np.linalg.norm(v2_tca - v1_tca) * 1000.0)

    lead_time_s = lead_time_hours * 3600.0
    steps = int(lead_time_s / 30.0)
    state1_initial = propagate_j2_drag(s1_tca, dt=-30.0, steps=steps, include_drag=include_drag)[-1]
    state2_initial = propagate_j2_drag(s2_tca, dt=-30.0, steps=steps, include_drag=include_drag)[-1]

    t = 2.0 * np.pi * np.sqrt(r_mag**3 / MU)
    steps_full = int(t / 30.0) + 1

    traj1_vis = propagate_j2_drag(state1_initial, dt=30.0, steps=steps_full, include_drag=include_drag)
    traj2_vis = propagate_j2_drag(state2_initial, dt=30.0, steps=steps_full, include_drag=include_drag)
    traj1 = propagate_j2_drag(state1_initial, dt=30.0, steps=steps, include_drag=include_drag)
    traj2 = propagate_j2_drag(state2_initial, dt=30.0, steps=steps, include_drag=include_drag)

    seps = np.linalg.norm(traj1[:, :3] - traj2[:, :3], axis=1)
    tca_idx = int(np.argmin(seps))
    actual_miss_km = float(seps[tca_idx])

    hbr_km = hbr_m / 1000.0
    sigma_combined_km = hbr_km / np.sqrt(2.0 * target_pc)
    c, sigma_radial, sigma_intrack, sigma_crosstrack = build_realistic_covariance(
        sigma_combined_km,
        lead_time_hours,
        altitude_km,
    )

    h1 = np.cross(state1_initial[:3], state1_initial[3:])
    h2 = np.cross(state2_initial[:3], state2_initial[3:])
    h1_hat = h1 / np.linalg.norm(h1)
    h2_hat = h2 / np.linalg.norm(h2)
    plane_angle = float(np.degrees(np.arccos(np.clip(np.dot(h1_hat, h2_hat), -1.0, 1.0))))

    print(f"Relative velocity at TCA: {actual_rel_vel:.0f} m/s   [must be > 5000]")
    print(f"Actual miss distance: {actual_miss_km * 1000.0:.1f} m          [must be 140-160m]")
    print(f"Angle between planes: {plane_angle:.1f} deg        [must be > 60]")
    print(f"sigma_combined: {sigma_combined_km * 1000.0:.1f} m")
    print(f"Covariance shape - R:{sigma_radial * 1000.0:.0f}m T:{sigma_intrack * 1000.0:.0f}m N:{sigma_crosstrack * 1000.0:.0f}m")
    print(f"Target Pc: {target_pc:.1e}")

    return {
        "state1_initial": state1_initial,
        "state2_initial": state2_initial,
        "traj1_km": traj1,
        "traj2_km": traj2,
        "traj1_vis_km": traj1_vis,
        "traj2_vis_km": traj2_vis,
        "tca_idx": tca_idx,
        "miss_distance_m": float(actual_miss_km * 1000.0),
        "rel_velocity_mps": float(actual_rel_vel),
        "cov1": c,
        "cov2": c,
        "hbr_m": hbr_m,
        "s1_tca": s1_tca,
        "s2_tca": s2_tca,
        "lead_time_hours": lead_time_hours,
    }


def save_cdm(cdm_data: dict, seed: int, output_dir: str = "outputs/cdm_events") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"cdm_{seed}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(convert(cdm_data), f, indent=2)

    print(f"CDM saved to {filepath}")
    return str(filepath)


def load_cdm(filepath: str) -> dict:
    with Path(filepath).open("r", encoding="utf-8") as f:
        data = json.load(f)
    array_fields = [
        "traj1_km",
        "traj2_km",
        "traj1_vis_km",
        "traj2_vis_km",
        "mc_scatter_m",
        "C_bplane_m2",
        "miss_2d_m",
        "cov1",
        "cov2",
    ]
    for field in array_fields:
        if field in data:
            data[field] = np.array(data[field])
    return data


def _build_pc_timeline(traj1: np.ndarray, traj2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray, hbr_km: float, tca_idx: int, dt: float) -> list[dict]:
    timeline: list[dict] = []
    start_idx = max(0, tca_idx - 50)
    for i in range(start_idx, tca_idx, 10):
        t_sec = float(i * dt)
        cov1_i = propagate_cov_simple(cov1, t_sec)
        cov2_i = propagate_cov_simple(cov2, t_sec)
        cov_combined_i = cov1_i[:3, :3] + cov2_i[:3, :3]
        sigma_km = float(np.sqrt(np.mean(np.diag(cov_combined_i))))
        miss_km = float(np.linalg.norm(traj1[i, :3] - traj2[i, :3]))
        pc_i = float((hbr_km**2) / (miss_km**2 + sigma_km**2))
        timeline.append({"time_to_tca_hours": float((tca_idx - i) * dt / 3600.0), "pc": pc_i})
    return timeline


def generate_synthetic_cdm(seed: int = 42):
    cdm_raw = generate_conjunction_from_geometry(
        miss_distance_m=150.0,
        rel_velocity_mps=11000.0,
        inclination_primary_deg=51.6,
        inclination_secondary_deg=97.8,
        altitude_km=400.0,
        target_pc=3e-4,
        hbr_m=20.0,
        lead_time_hours=4.0,
        seed=seed,
    )

    traj1 = np.asarray(cdm_raw["traj1_km"], dtype=float)
    traj2 = np.asarray(cdm_raw["traj2_km"], dtype=float)
    tca_idx = int(cdm_raw["tca_idx"])
    dt = 30.0
    tca_seconds = float(tca_idx * dt)
    cov1_tca = propagate_cov_simple(np.asarray(cdm_raw["cov1"], dtype=float), tca_seconds)
    cov2_tca = propagate_cov_simple(np.asarray(cdm_raw["cov2"], dtype=float), tca_seconds)
    hbr_km = float(cdm_raw["hbr_m"]) / 1000.0

    pc, miss_2d_km, c_bplane_km2, samples_km = compute_pc_bplane(
        traj1[tca_idx], traj2[tca_idx], cov1_tca, cov2_tca, hbr_km, n_samples=50000, seed=42
    )

    timeline = _build_pc_timeline(traj1, traj2, np.asarray(cdm_raw["cov1"], dtype=float), np.asarray(cdm_raw["cov2"], dtype=float), hbr_km, tca_idx, dt)
    timeline.append({"time_to_tca_hours": 0.0, "pc": float(pc)})
    timeline = list(reversed(timeline))

    samples_m = samples_km * 1000.0
    keep_n = min(2000, samples_m.shape[0])
    keep_idx = np.random.default_rng(seed).choice(samples_m.shape[0], size=keep_n, replace=False)

    miss_distance_m = float(np.linalg.norm(traj1[tca_idx, :3] - traj2[tca_idx, :3]) * 1000.0)
    rel_velocity_mps = float(np.linalg.norm(traj2[tca_idx, 3:] - traj1[tca_idx, 3:]) * 1000.0)

    print(f"Miss distance:    {miss_distance_m:.3f} m        [realistic: 10-500m]")
    print(f"Pc:               {pc:.6e}          [realistic: 1e-4 to 1e-2]")
    print(f"Rel velocity:     {rel_velocity_mps:.3f} m/s      [realistic: 5000-15000 m/s]")
    print(f"HBR:              {cdm_raw['hbr_m']:.3f} m")

    cdm_data = {
        "seed": int(seed),
        "saved_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "miss_distance_m": miss_distance_m,
        "pc": float(pc),
        "miss_2d_m": (miss_2d_km * 1000.0).tolist(),
        "C_bplane_m2": (c_bplane_km2 * 1_000_000.0).tolist(),
        "hbr_m": float(cdm_raw["hbr_m"]),
        "mc_scatter_m": samples_m[keep_idx].tolist(),
        "traj1_km": traj1.tolist(),
        "traj2_km": traj2.tolist(),
        "traj1_vis_km": np.asarray(cdm_raw["traj1_vis_km"]).tolist(),
        "traj2_vis_km": np.asarray(cdm_raw["traj2_vis_km"]).tolist(),
        "tca_idx": tca_idx,
        "rel_velocity_mps": rel_velocity_mps,
        "pc_timeline": timeline,
        "secondary_nu_deg": 178.5,
        "secondary_raan_deg": 90.0,
    }

    saved_path = save_cdm(cdm_data, seed=int(seed))
    cdm_data["saved_filepath"] = saved_path
    return cdm_data, None, None


def build_orbits_for_visualization(secondary_nu_deg: float = 178.5, secondary_raan_deg: float = 90.0):
    _ = secondary_nu_deg, secondary_raan_deg
    return None, None
