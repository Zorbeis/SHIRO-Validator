from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shiro.validator.cdm_from_stresslab import (  # noqa: E402
    build_orbits_for_visualization,
    compute_pc_bplane,
    generate_conjunction_from_geometry,
    load_cdm,
    propagate_cov_simple,
)
from shiro.validator.render_3d import render_3d  # noqa: E402


def _ellipse_from_cov(c_bplane_m2: np.ndarray, miss_2d_m: np.ndarray, n_points: int = 240) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(c_bplane_m2)
    eigvals = np.clip(eigvals, 1e-18, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    pts = []
    sigma_scale = 3.0
    for a in theta:
        p = np.array(
            [
                sigma_scale * np.sqrt(eigvals[0]) * np.cos(a),
                sigma_scale * np.sqrt(eigvals[1]) * np.sin(a),
            ]
        )
        q = eigvecs @ p + miss_2d_m
        pts.append(q)
    return np.asarray(pts)


def _earth_sphere_points(radius_km: float = 6371.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 24)
    lon = np.linspace(0.0, 2.0 * np.pi, 48)
    lat_grid, lon_grid = np.meshgrid(lat, lon)

    x = radius_km * np.cos(lat_grid) * np.cos(lon_grid)
    y = radius_km * np.cos(lat_grid) * np.sin(lon_grid)
    z = radius_km * np.sin(lat_grid)
    return x.flatten(), y.flatten(), z.flatten()


def _mc_pc_from_cov(
    miss_2d_m: np.ndarray,
    c_bplane_m2: np.ndarray,
    hbr_m: float,
    n_samples: int = 50000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(miss_2d_m, c_bplane_m2, n_samples)
    in_hbr = np.linalg.norm(samples, axis=1) < hbr_m
    pc = float(np.sum(in_hbr)) / float(n_samples)
    return pc, samples


def _build_cdm_data_from_geometry(cdm_raw: dict, seed: int) -> dict:
    traj1 = np.asarray(cdm_raw["traj1_km"], dtype=float)
    traj2 = np.asarray(cdm_raw["traj2_km"], dtype=float)
    tca_idx = int(cdm_raw["tca_idx"])
    dt = 30.0
    tca_seconds = float(tca_idx * dt)
    cov1 = np.asarray(cdm_raw["cov1"], dtype=float)
    cov2 = np.asarray(cdm_raw["cov2"], dtype=float)
    cov1_tca = propagate_cov_simple(cov1, tca_seconds)
    cov2_tca = propagate_cov_simple(cov2, tca_seconds)
    hbr_km = float(cdm_raw["hbr_m"]) / 1000.0

    pc, miss_2d_km, c_bplane_km2, samples_km = compute_pc_bplane(
        traj1[tca_idx],
        traj2[tca_idx],
        cov1_tca,
        cov2_tca,
        hbr_km,
        n_samples=50000,
        seed=seed,
    )

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
    timeline.append({"time_to_tca_hours": 0.0, "pc": float(pc)})
    timeline = list(reversed(timeline))

    samples_m = samples_km * 1000.0
    keep_n = min(2000, samples_m.shape[0])
    keep_idx = np.random.default_rng(seed).choice(samples_m.shape[0], size=keep_n, replace=False)

    miss_distance_m = float(np.linalg.norm(traj1[tca_idx, :3] - traj2[tca_idx, :3]) * 1000.0)
    rel_velocity_mps = float(np.linalg.norm(traj2[tca_idx, 3:] - traj1[tca_idx, 3:]) * 1000.0)

    return {
        "seed": int(seed),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Path to saved CDM JSON file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="disagree",
        choices=["disagree", "agree_act"],
        help="Scenario search mode when not loading a CDM",
    )
    args = parser.parse_args()

    output_plotly_name = "conjunction_viz.html"
    output_three_name = "conjunction_3d.html"

    if args.load:
        data = load_cdm(args.load)
        orbit1, orbit2 = build_orbits_for_visualization(
            float(data.get("secondary_nu_deg", 178.5)),
            float(data.get("secondary_raan_deg", 90.0)),
        )
        print(f"Loaded CDM from {args.load}")
        print("Loaded CDM event:")
        print(f"  Seed:             {data.get('seed', 'unknown')}")
        print(f"  Miss distance:    {float(data.get('miss_distance_m', 0.0)):.1f} m")
        print(f"  Pc (industry):    {float(data.get('pc', 0.0)):.3e}")
        print(f"  Rel velocity:     {float(data.get('rel_velocity_mps', 0.0)):.0f} m/s")
        print(f"  Saved at:         {data.get('saved_at_utc', 'unknown')}")
    else:
        data = None
        seed = 99
        miss_distance_m = 480.0
        threshold = 1e-4
        max_iterations = 300

        if args.mode == "disagree":
            desired_industry_action = "ACT"
            desired_shiro_action = "PASS"
            output_plotly_name = "conjunction_disagree.html"
            output_three_name = "conjunction_3d_disagree.html"
            miss_step = 50.0
        else:
            desired_industry_action = "ACT"
            desired_shiro_action = "ACT"
            output_plotly_name = "conjunction_agree_act.html"
            output_three_name = "conjunction_3d_agree_act.html"
            miss_step = -50.0

        found = False
        for _ in range(max_iterations):
            cdm_raw = generate_conjunction_from_geometry(
                miss_distance_m=miss_distance_m,
                rel_velocity_mps=11000.0,
                inclination_primary_deg=51.6,
                inclination_secondary_deg=97.8,
                altitude_km=400.0,
                target_pc=5e-5,
                hbr_m=20.0,
                lead_time_hours=4.0,
                seed=seed,
            )
            data = _build_cdm_data_from_geometry(cdm_raw, seed=seed)

            miss_2d_m_try = np.asarray(data["miss_2d_m"], dtype=float)
            c_bplane_m2_try = np.asarray(data["C_bplane_m2"], dtype=float)
            hbr_try = float(data["hbr_m"])
            c_bplane_shiro_try = c_bplane_m2_try * 25.0
            shiro_pc_try, _ = _mc_pc_from_cov(
                miss_2d_m_try,
                c_bplane_shiro_try,
                hbr_try,
                n_samples=50000,
                seed=43,
            )
            industry_pc_try = float(data["pc"])

            industry_action_try = "ACT" if industry_pc_try >= threshold else "PASS"
            shiro_action_try = "ACT" if shiro_pc_try >= threshold else "PASS"

            print(
                f"Trying miss_distance_m={miss_distance_m:.1f} "
                f"Industry Pc={industry_pc_try:.6e} "
                f"SHIRO Pc={shiro_pc_try:.6e} "
                f"-> {industry_action_try}/{shiro_action_try}"
            )

            if industry_action_try == desired_industry_action and shiro_action_try == desired_shiro_action:
                print(f"\n{args.mode.upper()} case found")
                print(f"  miss_distance_m: {float(data['miss_distance_m']):.1f}")
                print(f"  industry_pc:     {industry_pc_try:.6e}")
                print(f"  shiro_pc:        {shiro_pc_try:.6e}")
                print(f"  decision:        {industry_action_try}/{shiro_action_try}")
                found = True
                break

            miss_distance_m += miss_step
            miss_distance_m = max(10.0, miss_distance_m)

        if not found:
            raise RuntimeError(
                f"No {args.mode} case found after {max_iterations} attempts. "
                "Try adjusting geometry/search parameters."
            )

        assert data is not None

        orbit1, orbit2 = None, None

    miss_2d_m = np.asarray(data["miss_2d_m"], dtype=float)
    c_bplane_m2 = np.asarray(data["C_bplane_m2"], dtype=float)
    mc = np.asarray(data["mc_scatter_m"], dtype=float)
    ellipse = _ellipse_from_cov(c_bplane_m2, miss_2d_m)
    hbr = float(data["hbr_m"])

    # Covariance realism correction: operational catalogs are often overconfident
    # by roughly 3-5x in position, i.e. 9-25x in covariance. Use 25x here.
    covariance_inflation = 25.0
    c_bplane_shiro_m2 = c_bplane_m2 * covariance_inflation
    ellipse_shiro = _ellipse_from_cov(c_bplane_shiro_m2, miss_2d_m)
    pc_shiro_bplane, shiro_samples_full = _mc_pc_from_cov(miss_2d_m, c_bplane_shiro_m2, hbr, n_samples=50000, seed=43)
    shiro_keep = min(2000, shiro_samples_full.shape[0])
    shiro_idx = np.random.default_rng(43).choice(shiro_samples_full.shape[0], size=shiro_keep, replace=False)
    mc_shiro = shiro_samples_full[shiro_idx]

    ang = np.linspace(0.0, 2.0 * np.pi, 240)
    hbr_circle = np.column_stack((hbr * np.cos(ang), hbr * np.sin(ang)))

    timeline = data["pc_timeline"]
    t_to_tca_h = [float(p["time_to_tca_hours"]) for p in timeline]
    pc_ref = [float(p["pc"]) for p in timeline]
    pc_deg = [max(v * 1.3, 1e-16) for v in pc_ref]
    max_time_to_tca = max(t_to_tca_h) if t_to_tca_h else 0.7

    traj1 = np.asarray(data["traj1_km"], dtype=float)
    traj2 = np.asarray(data["traj2_km"], dtype=float)
    traj1_vis = np.asarray(data.get("traj1_vis_km", data["traj1_km"]), dtype=float)
    traj2_vis = np.asarray(data.get("traj2_vis_km", data["traj2_km"]), dtype=float)
    tca_idx = int(data["tca_idx"])
    conj = (traj1[tca_idx, :3] + traj2[tca_idx, :3]) / 2.0

    sigma2_industry_km2 = float(np.mean(np.diag(c_bplane_m2))) / 1_000_000.0
    sigma2_shiro_km2 = sigma2_industry_km2 * covariance_inflation
    pc_shiro_timeline: list[float] = []
    for t_h in t_to_tca_h:
        i = int(round(tca_idx - (t_h * 3600.0 / 30.0)))
        i = max(0, min(i, len(traj1) - 1))
        miss_km = float(np.linalg.norm(traj1[i, :3] - traj2[i, :3]))
        pc_i = float((hbr / 1000.0) ** 2 / (miss_km**2 + sigma2_shiro_km2))
        pc_shiro_timeline.append(max(pc_i, 1e-16))

    threshold = 1e-4
    shiro_action = "ACT" if pc_shiro_bplane >= threshold else "PASS"
    industry_action = "ACT" if float(data["pc"]) >= threshold else "PASS"

    earth_radius_km = 6371.0
    ex, ey, ez = _earth_sphere_points(radius_km=earth_radius_km)

    eigvals_m2 = np.linalg.eigvals(c_bplane_m2)
    industry_pc_after_floor = float(data["pc"])
    shiro_pc_after_floor = float(pc_shiro_bplane)
    data["pc_shiro"] = shiro_pc_after_floor
    print(f"Earth sphere radius in viz: {earth_radius_km} km")
    print(f"C_bplane eigenvalues: [{eigvals_m2[0]:.6f}, {eigvals_m2[1]:.6f}] m^2")
    print(f"Industry Pc after floor: {industry_pc_after_floor:.6e}")
    print(f"SHIRO Pc after floor: {shiro_pc_after_floor:.6e}")
    print(f"Industry Pc must be > SHIRO Pc - confirm: {industry_pc_after_floor > shiro_pc_after_floor}")

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "scene", "colspan": 2}, None],
        ],
        subplot_titles=("Industry View", "SHIRO View", "Pc Timeline", "3D Orbit View"),
        row_heights=[400 / 1300, 300 / 1300, 600 / 1300],
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scattergl(
            x=mc[:, 0],
            y=mc[:, 1],
            mode="markers",
            marker={"size": 3, "opacity": 0.35, "color": "#60a5fa"},
            name="Monte Carlo TCA",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ellipse[:, 0],
            y=ellipse[:, 1],
            mode="lines",
            line={"color": "#2563eb", "width": 3},
            name="Covariance ellipse (3-sigma)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hbr_circle[:, 0],
            y=hbr_circle[:, 1],
            mode="lines",
            line={"color": "#ef4444", "width": 2},
            name="HBR circle",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[miss_2d_m[0]],
            y=[miss_2d_m[1]],
            mode="markers",
            marker={"size": 10, "color": "#22c55e", "symbol": "x"},
            name="Miss vector",
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        text=f"Pc = {float(data['pc']):.1e} -> {industry_action}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="#94a3b8",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=mc_shiro[:, 0],
            y=mc_shiro[:, 1],
            mode="markers",
            marker={"size": 3, "opacity": 0.35, "color": "#fb7185"},
            name="Monte Carlo TCA (SHIRO)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ellipse_shiro[:, 0],
            y=ellipse_shiro[:, 1],
            mode="lines",
            line={"color": "#dc2626", "width": 3},
            name="Covariance ellipse (SHIRO)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=hbr_circle[:, 0],
            y=hbr_circle[:, 1],
            mode="lines",
            line={"color": "#ef4444", "width": 2},
            name="HBR circle",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[miss_2d_m[0]],
            y=[miss_2d_m[1]],
            mode="markers",
            marker={"size": 10, "color": "#22c55e", "symbol": "x"},
            name="Miss vector",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_annotation(
        xref="x2 domain",
        yref="y2 domain",
        x=0.02,
        y=0.98,
        text=f"Pc = {pc_shiro_bplane:.1e} -> {shiro_action}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="#94a3b8",
    )

    fig.add_trace(
        go.Scatter(
            x=t_to_tca_h,
            y=pc_ref,
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#3b82f6", "width": 3},
            name="Pc reference",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_to_tca_h,
            y=pc_deg,
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#f59e0b", "width": 2},
            name="Pc degraded",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_to_tca_h,
            y=pc_shiro_timeline,
            mode="lines+markers",
            marker={"size": 4},
            line={"color": "#ef4444", "width": 3},
            name="Pc SHIRO (covariance-corrected)",
        ),
        row=2,
        col=1,
    )

    y_upper = np.maximum(np.asarray(pc_ref), np.asarray(pc_shiro_timeline))
    y_lower = np.minimum(np.asarray(pc_ref), np.asarray(pc_shiro_timeline))
    fig.add_trace(
        go.Scatter(
            x=t_to_tca_h,
            y=y_upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_to_tca_h,
            y=y_lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(239,68,68,0.15)",
            name="Unnecessary maneuver zone",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=1e-4, line_dash="dash", line_color="#ef4444", row=2, col=1)

    mid_idx = len(t_to_tca_h) // 2
    fig.add_annotation(
        x=t_to_tca_h[mid_idx],
        y=float(max(y_upper[mid_idx], 1e-16)),
        text="Unnecessary maneuver zone",
        showarrow=False,
        bgcolor="rgba(254,202,202,0.8)",
        bordercolor="#fca5a5",
        row=2,
        col=1,
    )

    crossing_x = None
    crossing_y = None
    pairs = sorted(zip(t_to_tca_h, pc_ref), key=lambda p: p[0], reverse=True)
    prev = pairs[0]
    for cur in pairs[1:]:
        if prev[1] < threshold <= cur[1]:
            crossing_x = cur[0]
            crossing_y = cur[1]
            break
        prev = cur
    if crossing_x is not None and crossing_y is not None:
        fig.add_annotation(
            x=crossing_x,
            y=crossing_y,
            text="Industry: ACT",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30,
            bgcolor="rgba(191,219,254,0.85)",
            bordercolor="#60a5fa",
            row=2,
            col=1,
        )
    if all(p < threshold for p in pc_shiro_timeline):
        fig.add_annotation(
            x=min(t_to_tca_h),
            y=max(pc_shiro_timeline[-1], 2e-5),
            text="SHIRO: PASS",
            showarrow=True,
            arrowhead=2,
            ax=-35,
            ay=-25,
            bgcolor="rgba(254,202,202,0.85)",
            bordercolor="#f87171",
            row=2,
            col=1,
        )

    def split_orbit_front_back(traj_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        front_mask = traj_km[:, 2] >= 0
        back_mask = ~front_mask
        return traj_km[front_mask], traj_km[back_mask]

    front1, back1 = split_orbit_front_back(traj1[:, :3])
    front2, back2 = split_orbit_front_back(traj2[:, :3])

    fig.add_trace(
        go.Scatter3d(
            x=traj1_vis[:, 0],
            y=traj1_vis[:, 1],
            z=traj1_vis[:, 2],
            mode="lines",
            line={"color": "#00ff88", "width": 3},
            name="Orbit 1 (Primary)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=traj2_vis[:, 0],
            y=traj2_vis[:, 1],
            z=traj2_vis[:, 2],
            mode="lines",
            line={"color": "#ff6600", "width": 3},
            name="Orbit 2 (Secondary)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(conj[0])],
            y=[float(conj[1])],
            z=[float(conj[2])],
            mode="markers",
            marker={"size": 8, "color": "red", "symbol": "diamond"},
            name="Conjunction point",
        ),
        row=3,
        col=1,
    )

    n = len(timeline)
    cdm_indices = [int(n * 0.25), int(n * 0.50), int(n * 0.75), int(n * 0.95)]
    dt = 30.0
    for i, idx in enumerate(cdm_indices):
        idx = max(0, min(idx, n - 1))
        entry = timeline[idx]
        time_to_tca_hours = float(entry["time_to_tca_hours"])
        pc_at_cdm = float(entry["pc"])

        t_seconds_before_tca = time_to_tca_hours * 3600.0
        traj_idx = max(0, tca_idx - int(t_seconds_before_tca / dt))
        pos = traj1[traj_idx, :3]

        fig.add_trace(
            go.Scatter3d(
                x=[float(pos[0])],
                y=[float(pos[1])],
                z=[float(pos[2])],
                mode="markers+text",
                marker={"size": 6, "color": "yellow", "symbol": "circle"},
                text=[f"CDM-{i+1}<br>T-{time_to_tca_hours:.1f}h<br>Pc={pc_at_cdm:.2e}"],
                textposition="top center",
                name=f"CDM-{i+1}",
                showlegend=True,
            ),
            row=3,
            col=1,
        )

    fig.update_xaxes(title_text="Eta (m)", row=1, col=1)
    fig.update_yaxes(title_text="Zeta (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="Eta (m)", row=1, col=2)
    fig.update_yaxes(title_text="Zeta (m)", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(
        title_text="Time to TCA (hours)",
        autorange="reversed",
        range=[0.7, 0.0],
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Pc", type="log", row=2, col=1)
    fig.update_scenes(
        bgcolor="#0a0a1a",
        xaxis_backgroundcolor="#0a0a1a",
        yaxis_backgroundcolor="#0a0a1a",
        zaxis_backgroundcolor="#0a0a1a",
        xaxis=dict(backgroundcolor="#0a0a1a", gridcolor="#1a1a3a", zerolinecolor="#1a1a3a", color="#444466", title="X (km)"),
        yaxis=dict(backgroundcolor="#0a0a1a", gridcolor="#1a1a3a", zerolinecolor="#1a1a3a", color="#444466", title="Y (km)"),
        zaxis=dict(backgroundcolor="#0a0a1a", gridcolor="#1a1a3a", zerolinecolor="#1a1a3a", color="#444466", title="Z (km)"),
        aspectmode="data",
        row=3,
        col=1,
    )
    fig.update_layout(
        height=1300,
        width=1400,
        title="SHIRO Conjunction Visualization",
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        scene=dict(
            bgcolor="#0a0a1a",
            xaxis=dict(
                backgroundcolor="#0a0a1a",
                gridcolor="#1a1a3a",
                zerolinecolor="#1a1a3a",
                color="#444466",
                title="X (km)",
            ),
            yaxis=dict(
                backgroundcolor="#0a0a1a",
                gridcolor="#1a1a3a",
                zerolinecolor="#1a1a3a",
                color="#444466",
                title="Y (km)",
            ),
            zaxis=dict(
                backgroundcolor="#0a0a1a",
                gridcolor="#1a1a3a",
                zerolinecolor="#1a1a3a",
                color="#444466",
                title="Z (km)",
            ),
        ),
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_plotly_name
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    webbrowser.open(out_path.resolve().as_uri())
    print(f"Saved visualization to {out_path}")
    render_3d(data, orbit1, orbit2, output_filename=output_three_name)


if __name__ == "__main__":
    main()
