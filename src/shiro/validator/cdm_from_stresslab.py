from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


MU = 398600.4418
RE = 6371.0
J2 = 1.08263e-3


def propagate_j2(state: np.ndarray, dt: float, steps: int) -> np.ndarray:
    def dyn(s: np.ndarray) -> np.ndarray:
        x, y, z = s[0], s[1], s[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r2 = r**2
        factor = -MU / r**3
        j2_factor = 1.5 * J2 * MU * RE**2 / r**5
        ax = factor * x + j2_factor * x * (1.0 - 5.0 * z**2 / r2)
        ay = factor * y + j2_factor * y * (1.0 - 5.0 * z**2 / r2)
        az = factor * z + j2_factor * z * (3.0 - 5.0 * z**2 / r2)
        return np.array([s[3], s[4], s[5], ax, ay, az], dtype=float)

    def rk4_step(s: np.ndarray, h: float) -> np.ndarray:
        k1 = dyn(s)
        k2 = dyn(s + 0.5 * h * k1)
        k3 = dyn(s + 0.5 * h * k2)
        k4 = dyn(s + h * k3)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    traj = np.zeros((steps + 1, 6), dtype=float)
    traj[0] = state.astype(float)
    for i in range(steps):
        traj[i + 1] = rk4_step(traj[i], dt)
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
) -> dict:
    _ = seed
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

    rel_vel = v2_tca - v1_tca
    rel_vel_norm = rel_vel / np.linalg.norm(rel_vel)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    miss_dir = np.cross(rel_vel_norm, z_hat)
    if np.linalg.norm(miss_dir) < 1e-10:
        miss_dir = np.cross(rel_vel_norm, np.array([0.0, 1.0, 0.0], dtype=float))
    miss_dir = miss_dir / np.linalg.norm(miss_dir)

    miss_km = miss_distance_m / 1000.0
    r2_tca = r1_tca + miss_dir * miss_km

    s1_tca = np.concatenate([r1_tca, v1_tca])
    s2_tca = np.concatenate([r2_tca, v2_tca])

    actual_rel_vel = float(np.linalg.norm(v2_tca - v1_tca) * 1000.0)

    lead_time_s = lead_time_hours * 3600.0
    steps = int(lead_time_s / 30.0)
    state1_initial = propagate_j2(s1_tca, dt=-30.0, steps=steps)[-1]
    state2_initial = propagate_j2(s2_tca, dt=-30.0, steps=steps)[-1]

    t = 2.0 * np.pi * np.sqrt(r_mag**3 / MU)
    steps_full = int(t / 30.0) + 1

    traj1_vis = propagate_j2(state1_initial, dt=30.0, steps=steps_full)
    traj2_vis = propagate_j2(state2_initial, dt=30.0, steps=steps_full)
    traj1 = propagate_j2(state1_initial, dt=30.0, steps=steps)
    traj2 = propagate_j2(state2_initial, dt=30.0, steps=steps)

    seps = np.linalg.norm(traj1[:, :3] - traj2[:, :3], axis=1)
    tca_idx = int(np.argmin(seps))
    actual_miss_km = float(seps[tca_idx])

    hbr_km = hbr_m / 1000.0
    sigma_combined_km = hbr_km / np.sqrt(2.0 * target_pc)
    sigma_radial = sigma_combined_km * 0.3
    sigma_intrack = sigma_combined_km * 0.9
    sigma_crosstrack = sigma_combined_km * 0.3
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

    h1 = np.cross(state1_initial[:3], state1_initial[3:])
    h2 = np.cross(state2_initial[:3], state2_initial[3:])
    h1_hat = h1 / np.linalg.norm(h1)
    h2_hat = h2 / np.linalg.norm(h2)
    plane_angle = float(np.degrees(np.arccos(np.clip(np.dot(h1_hat, h2_hat), -1.0, 1.0))))

    print(f"Relative velocity at TCA: {actual_rel_vel:.0f} m/s   [must be > 5000]")
    print(f"Actual miss distance: {actual_miss_km * 1000.0:.1f} m          [must be 140-160m]")
    print(f"Angle between planes: {plane_angle:.1f} deg        [must be > 60]")
    print(f"sigma_combined: {sigma_combined_km * 1000.0:.1f} m")
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
