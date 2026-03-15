from __future__ import annotations

import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shiro.validator.cdm_from_stresslab import compute_pc_bplane, generate_conjunction_from_geometry  # noqa: E402
from shiro.validator.covariance_model import get_inflation_for_event  # noqa: E402


INFLATION_FACTORS = [9, 16, 25]
INDUSTRY_THRESHOLD = 1e-4
SHIRO_THRESHOLD = 1e-4


def generate_batch(n_events: int = 50, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    events: list[dict] = []

    for i in range(n_events):
        miss_distance_m = float(np.exp(rng.uniform(np.log(50.0), np.log(800.0))))
        rel_velocity_mps = float(rng.uniform(5000.0, 15000.0))
        altitude_km = float(rng.uniform(350.0, 600.0))
        lead_time_hours = float(rng.uniform(2.0, 72.0))
        inclination_primary = float(rng.uniform(28.0, 98.0))
        inclination_secondary = float(rng.uniform(28.0, 98.0))
        while abs(inclination_secondary - inclination_primary) < 20.0:
            inclination_secondary = float(rng.uniform(28.0, 98.0))
        hbr_m = float(rng.uniform(5.0, 25.0))
        event_seed = int(rng.integers(0, 10000))

        events.append(
            {
                "event_id": f"SYN_{i + 1:03d}",
                "miss_distance_m": miss_distance_m,
                "rel_velocity_mps": rel_velocity_mps,
                "altitude_km": altitude_km,
                "lead_time_hours": lead_time_hours,
                "inclination_primary_deg": inclination_primary,
                "inclination_secondary_deg": inclination_secondary,
                "hbr_m": hbr_m,
                "seed": event_seed,
            }
        )

    return events


def compute_pc_from_cov(
    miss_2d_km: np.ndarray,
    c_bplane_km2: np.ndarray,
    hbr_km: float,
    n_samples: int = 50000,
) -> float:
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(miss_2d_km, c_bplane_km2, n_samples)
    in_hbr = np.linalg.norm(samples, axis=1) < hbr_km
    return float(np.sum(in_hbr)) / float(n_samples)


def run_event(params: dict, inflation_factor: int) -> dict:
    try:
        include_drag = bool(params["altitude_km"] < 600.0)
        cdm = generate_conjunction_from_geometry(
            miss_distance_m=params["miss_distance_m"],
            rel_velocity_mps=params["rel_velocity_mps"],
            inclination_primary_deg=params["inclination_primary_deg"],
            inclination_secondary_deg=params["inclination_secondary_deg"],
            altitude_km=params["altitude_km"],
            target_pc=1e-4,
            hbr_m=params["hbr_m"],
            lead_time_hours=params["lead_time_hours"],
            seed=params["seed"],
            include_drag=include_drag,
        )

        recommended_k2 = get_inflation_for_event(
            miss_distance_m=params["miss_distance_m"],
            altitude_km=params["altitude_km"],
            lead_time_hours=params["lead_time_hours"],
            is_active=True,
        )

        pc_industry, miss_2d, c_bplane, _ = compute_pc_bplane(
            cdm["s1_tca"],
            cdm["s2_tca"],
            cdm["cov1"],
            cdm["cov2"],
            cdm["hbr_m"] / 1000.0,
        )

        c_shiro = c_bplane * float(inflation_factor)
        pc_shiro = compute_pc_from_cov(miss_2d, c_shiro, cdm["hbr_m"] / 1000.0)

        industry_acts = pc_industry >= INDUSTRY_THRESHOLD
        shiro_acts = pc_shiro >= SHIRO_THRESHOLD

        if industry_acts and shiro_acts:
            outcome = "AGREE_ACT"
        elif industry_acts and not shiro_acts:
            outcome = "UNNECESSARY_MANEUVER"
        elif not industry_acts and shiro_acts:
            outcome = "SHIRO_ONLY_ACT"
        else:
            outcome = "AGREE_PASS"

        return {
            "event_id": params["event_id"],
            "miss_distance_m": cdm["miss_distance_m"],
            "rel_velocity_mps": cdm["rel_velocity_mps"],
            "hbr_m": params["hbr_m"],
            "lead_time_hours": params["lead_time_hours"],
            "pc_industry": float(pc_industry),
            "pc_shiro": float(pc_shiro),
            "industry_acts": bool(industry_acts),
            "shiro_acts": bool(shiro_acts),
            "outcome": outcome,
            "inflation_factor": int(inflation_factor),
            "recommended_inflation_k2": float(recommended_k2),
            "error": None,
        }

    except Exception as exc:
        return {
            "event_id": params["event_id"],
            "outcome": "ERROR",
            "inflation_factor": int(inflation_factor),
            "error": str(exc),
        }


def run_sub_batch(events: list[dict], inflation_factor: int) -> list[dict]:
    n_events = len(events)
    print(f"\nSUB-BATCH - covariance inflation {inflation_factor}x")
    print("-" * 60)
    results: list[dict] = []

    for i, params in enumerate(events):
        result = run_event(params, inflation_factor)
        results.append(result)

        if result.get("error"):
            print(f"  [{i + 1:3d}/{n_events}] {params['event_id']} -> ERROR: {result['error'][:50]}")
        else:
            print(
                f"  [{i + 1:3d}/{n_events}] {params['event_id']} "
                f"miss={result['miss_distance_m']:6.0f}m "
                f"Pc_ind={result['pc_industry']:.1e} "
                f"Pc_shi={result['pc_shiro']:.1e} "
                f"-> {result['outcome']}"
            )

    return results


def summarize_sub_batch(results: list[dict], inflation_factor: int) -> dict:
    valid = [r for r in results if r["outcome"] != "ERROR"]
    counts = {
        "AGREE_ACT": 0,
        "UNNECESSARY_MANEUVER": 0,
        "SHIRO_ONLY_ACT": 0,
        "AGREE_PASS": 0,
        "ERROR": 0,
    }
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    industry_acts_total = sum(1 for r in valid if r.get("industry_acts"))
    unnecessary = counts["UNNECESSARY_MANEUVER"]
    umr = unnecessary / industry_acts_total if industry_acts_total > 0 else 0.0

    print(f"\nSub-batch summary ({inflation_factor}x)")
    print(f"AGREE_ACT:        {counts['AGREE_ACT']:3d}  (both act - genuine risk)")
    print(f"UNNECESSARY_MAN:  {counts['UNNECESSARY_MANEUVER']:3d}  (industry acts, SHIRO passes)")
    print(f"SHIRO_ONLY_ACT:   {counts['SHIRO_ONLY_ACT']:3d}  (SHIRO acts, industry passes)")
    print(f"AGREE_PASS:       {counts['AGREE_PASS']:3d}  (both pass - safe event)")
    print(f"  UMR:                  {umr:.1%}")

    return {
        "inflation_factor": inflation_factor,
        "n_events": len(results),
        "valid_runs": len(valid),
        "summary": counts,
        "industry_maneuvers_total": industry_acts_total,
        "unnecessary_maneuvers": unnecessary,
        "unnecessary_maneuver_rate": umr,
        "events": valid,
    }


def print_sensitivity_table(summaries: list[dict]) -> None:
    print("\nSENSITIVITY ANALYSIS - Covariance Inflation Factor")
    print("-" * 52)
    print("Inflation   AGREE_ACT   UNNEC_MAN   SHI_ONLY   AGREE_PASS   UMR")
    for s in summaries:
        counts = s["summary"]
        print(
            f"{s['inflation_factor']}x"
            f"{'':10}"
            f"{counts['AGREE_ACT']:<11}"
            f"{counts['UNNECESSARY_MANEUVER']:<12}"
            f"{counts['SHIRO_ONLY_ACT']:<10}"
            f"{counts['AGREE_PASS']:<13}"
            f"{s['unnecessary_maneuver_rate']:.1%}"
        )
    print("-" * 52)


def run_sensitivity(n_events: int = 50, seed: int = 0) -> dict:
    print(f"\nSHIRO BATCH VALIDATOR - {n_events} synthetic events")
    print("Sensitivity factors: 9x, 16x, 25x")
    print(f"Decision threshold: {INDUSTRY_THRESHOLD:.0e}")
    print("-" * 60)

    events = generate_batch(n_events=n_events, seed=seed)
    summaries: list[dict] = []

    for factor in INFLATION_FACTORS:
        results = run_sub_batch(events, inflation_factor=factor)
        summaries.append(summarize_sub_batch(results, inflation_factor=factor))

    print_sensitivity_table(summaries)

    output = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "n_events": n_events,
        "seed": seed,
        "inflation_factors": INFLATION_FACTORS,
        "industry_threshold": INDUSTRY_THRESHOLD,
        "shiro_threshold": SHIRO_THRESHOLD,
        "sub_batch_summaries": summaries,
    }

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path(f"outputs/batch_results_sensitivity_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return output


def plot_summary(summary: dict) -> None:
    keys = ["AGREE_ACT", "UNNECESSARY_MANEUVER", "SHIRO_ONLY_ACT", "AGREE_PASS"]
    labels = ["AGREE_ACT", "UNNECESSARY\\nMANEUVER", "SHIRO_ONLY\\nACT", "AGREE_PASS"]
    colors = {9: "#60a5fa", 16: "#f59e0b", 25: "#ef4444"}

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
    )

    for sub in summary["sub_batch_summaries"]:
        values = [sub["summary"].get(k, 0) for k in keys]
        factor = int(sub["inflation_factor"])
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=f"{factor}x",
                marker_color=colors.get(factor, "#94a3b8"),
                text=values,
                textposition="outside",
            ),
            row=1,
            col=1,
        )

    table_headers = ["Inflation", "AGREE_ACT", "UNNEC_MAN", "SHI_ONLY", "AGREE_PASS", "UMR"]
    table_rows = []
    for sub in summary["sub_batch_summaries"]:
        table_rows.append(
            [
                f"{sub['inflation_factor']}x",
                sub["summary"]["AGREE_ACT"],
                sub["summary"]["UNNECESSARY_MANEUVER"],
                sub["summary"]["SHIRO_ONLY_ACT"],
                sub["summary"]["AGREE_PASS"],
                f"{sub['unnecessary_maneuver_rate']:.1%}",
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(values=table_headers, fill_color="#1a1a3a", font=dict(color="white"), align="center"),
            cells=dict(
                values=list(map(list, zip(*table_rows))),
                fill_color="#0d0d2b",
                font=dict(color="white"),
                align="center",
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"SHIRO Sensitivity Results - {summary['n_events']} Synthetic Events",
            font=dict(size=16, color="white"),
        ),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0d0d2b",
        font=dict(color="white", family="Courier New"),
        yaxis=dict(title="Number of Events", gridcolor="#1a1a3a"),
        xaxis=dict(gridcolor="#1a1a3a"),
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    out_path = Path("outputs/batch_summary.html")
    fig.write_html(str(out_path))
    webbrowser.open(out_path.resolve().as_uri())
    print("Batch summary chart saved to outputs/batch_summary.html")


if __name__ == "__main__":
    summary = run_sensitivity(n_events=50, seed=0)
    plot_summary(summary)
