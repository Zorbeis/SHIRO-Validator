from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from cdm_from_stresslab import CDMEvent


def _conjunction_plane_projection(event: CDMEvent) -> tuple[np.ndarray, np.ndarray]:
    cov1 = np.array(event.cov1_at_tca, dtype=float)
    cov2 = np.array(event.cov2_at_tca, dtype=float)
    cov_rel_pos = cov1[:3, :3] + cov2[:3, :3]

    eta = np.array(event.b_plane_eta, dtype=float)
    zeta = np.array(event.b_plane_zeta, dtype=float)
    basis = np.vstack([eta, zeta])
    cov_2d = basis @ cov_rel_pos @ basis.T

    rel_position = np.array(event.rel_position_km, dtype=float)
    miss_vec_2d = np.array([
        float(np.dot(rel_position, eta)),
        float(np.dot(rel_position, zeta)),
    ])
    return miss_vec_2d, cov_2d


def visualize_cdm(event: CDMEvent):
    miss_vec_km, cov_2d_km2 = _conjunction_plane_projection(event)

    eigvals, eigvecs = np.linalg.eigh(cov_2d_km2)
    eigvals = np.clip(eigvals, a_min=1e-18, a_max=None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_scale = 3.0
    width_m = 2.0 * sigma_scale * np.sqrt(eigvals[0]) * 1000.0
    height_m = 2.0 * sigma_scale * np.sqrt(eigvals[1]) * 1000.0
    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

    miss_vec_m = miss_vec_km * 1000.0
    times_h = (np.array(event.timeline_t_s, dtype=float) - event.t_at_min_miss_s) / 3600.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax0 = axes[0]
    ax0.axhline(0.0, color="#dddddd", linewidth=0.8)
    ax0.axvline(0.0, color="#dddddd", linewidth=0.8)
    ellipse = Ellipse(
        xy=(0.0, 0.0),
        width=width_m,
        height=height_m,
        angle=angle_deg,
        edgecolor="#1f77b4",
        facecolor="#1f77b4",
        alpha=0.15,
        linewidth=2.0,
        label="Combined covariance ellipse (3-sigma)",
    )
    ax0.add_patch(ellipse)
    hbr_circle = Circle(
        (0.0, 0.0),
        radius=event.combined_hbr_m,
        edgecolor="#d62728",
        facecolor="none",
        linewidth=2.0,
        label="Combined HBR",
    )
    ax0.add_patch(hbr_circle)
    ax0.quiver(
        0.0,
        0.0,
        miss_vec_m[0],
        miss_vec_m[1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#2ca02c",
        width=0.006,
        label="Miss distance vector",
    )
    ax0.scatter([miss_vec_m[0]], [miss_vec_m[1]], color="#2ca02c", s=40)
    ax0.set_title("Conjunction Plane")
    ax0.set_xlabel("Eta axis (m)")
    ax0.set_ylabel("Zeta axis (m)")
    ax0.set_aspect("equal", adjustable="box")
    max_extent = max(
        abs(miss_vec_m[0]),
        abs(miss_vec_m[1]),
        width_m / 2.0,
        height_m / 2.0,
        event.combined_hbr_m,
    )
    ax0.set_xlim(-1.4 * max_extent, 1.4 * max_extent)
    ax0.set_ylim(-1.4 * max_extent, 1.4 * max_extent)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(
        0.02,
        0.02,
        f"Miss: {event.miss_distance_m:.2f} m\nPc(ref): {event.pc:.3e}",
        transform=ax0.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    ax1 = axes[1]
    ax1.semilogy(times_h, event.timeline_pc_reference, label="Pc reference", color="#1f77b4")
    ax1.semilogy(times_h, event.timeline_pc_degraded, label="Pc degraded", color="#ff7f0e")
    ax1.axhline(1e-4, color="#d62728", linestyle="--", linewidth=1.5, label="Decision threshold 1e-4")
    ax1.set_title("Pc Timeline")
    ax1.set_xlabel("Time to TCA (hours)")
    ax1.set_ylabel("Pc")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = axes[2]
    ax2.plot(times_h, event.timeline_cov_obj1_reference_trace, label="Obj1 ref", color="#1f77b4")
    ax2.plot(times_h, event.timeline_cov_obj1_degraded_trace, label="Obj1 degraded", color="#1f77b4", linestyle="--")
    ax2.plot(times_h, event.timeline_cov_obj2_reference_trace, label="Obj2 ref", color="#ff7f0e")
    ax2.plot(times_h, event.timeline_cov_obj2_degraded_trace, label="Obj2 degraded", color="#ff7f0e", linestyle="--")
    ax2.set_title("Covariance Growth")
    ax2.set_xlabel("Time to TCA (hours)")
    ax2.set_ylabel("Trace (km^2)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"Synthetic CDM Visualization: {event.event_id}", fontsize=14)
    fig.tight_layout()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cdm_visualization_{event.event_id}.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved visualization: {output_path}")
    return fig
