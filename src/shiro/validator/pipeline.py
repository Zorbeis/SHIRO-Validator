from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import spacetrack.operators as op
from spacetrack import SpaceTrackClient


# SpaceTrack API limits: 30 req/min, 300 req/hour
# This pipeline uses batched queries - total requests per run: ~2
# Do NOT add loops that make per-object API calls
POLICY_VERSION = "v0.1.0"
CACHE_MAX_AGE_HOURS = 24
ALLOWED_PRIMARY_NAME_TOKENS = {
    "STARLINK",
    "ONEWEB",
    "PLANET",
    "FLOCK",
    "LEMUR",
    "IRIDIUM",
    "ORBCOMM",
    "SPIRE",
    "ISS",
    "GPS",
    "GOES",
    "TERRA",
    "AQUA",
}


@dataclass
class CDMEvent:
    cdm_id: str
    created_at: datetime
    tca: datetime
    lead_time_hours: float
    miss_distance_m: float
    pc: float
    sat1_id: str
    sat1_name: str
    sat1_type: str
    sat2_id: str
    sat2_name: str
    sat2_type: str
    combined_hbr_m: float
    emergency_flag: bool


@dataclass
class ManeuverDetection:
    maneuver_detected: bool
    confidence: str
    delta_mean_motion: float | None
    delta_eccentricity: float | None
    tle_gap_hours: float | None


@dataclass
class PolicyDecision:
    recommends_act: bool
    rationale: str


@dataclass
class ValidatorResult:
    cdm_event: CDMEvent
    maneuver_detection: ManeuverDetection
    industry_acted: bool
    policy_decision: PolicyDecision
    outcome: str
    unnecessary: bool


class SpaceTrackDataSource:
    def __init__(self, user: str, password: str, cache_dir: str) -> None:
        self.client = SpaceTrackClient(identity=user, password=password)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_count = 0

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return age <= timedelta(hours=CACHE_MAX_AGE_HOURS)

    def _normalize_api_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, str):
            parsed = json.loads(payload)
        elif isinstance(payload, list):
            parsed = payload
        else:
            parsed = list(payload)

        if not isinstance(parsed, list):
            raise RuntimeError("Unexpected SpaceTrack response format")
        if parsed and all(isinstance(item, dict) and "error" in item for item in parsed):
            raise RuntimeError(f"SpaceTrack API error: {parsed[0].get('error')}")
        return [item for item in parsed if isinstance(item, dict)]

    def _load_or_fetch(
        self,
        cache_path: Path,
        fetch_fn: Callable[[], Any],
        cache_label: str,
    ) -> list[dict[str, Any]]:
        if self._cache_is_fresh(cache_path):
            print(f"Loading {cache_label} from cache: {cache_path}")
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._normalize_api_payload(cached)

        payload = fetch_fn()
        self.request_count += 1
        rows = self._normalize_api_payload(payload)
        cache_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return rows

    def fetch_cdm_rows(self, days_back: int, pc_min: float, limit: int) -> list[dict[str, Any]]:
        cache_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        cache_path = self.cache_dir / f"cdm_events_{cache_date}.json"
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        return self._load_or_fetch(
            cache_path,
            lambda: self.client.cdm_public(
                tca=op.greater_than(since),
                pc=op.greater_than(pc_min),
                limit=limit,
                orderby="pc desc",
                format="json",
            ),
            "CDM events",
        )

    def fetch_batched_tle_rows(
        self,
        norad_ids: list[str],
        earliest_tca: datetime,
        latest_tca: datetime,
    ) -> list[dict[str, Any]]:
        if not norad_ids:
            return []

        cache_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        cache_path = self.cache_dir / f"tle_history_{cache_date}.json"
        date_range = op.inclusive_range(
            earliest_tca - timedelta(days=3),
            latest_tca + timedelta(days=3),
        )

        return self._load_or_fetch(
            cache_path,
            lambda: self.client.gp_history(
                norad_cat_id=norad_ids,
                epoch=date_range,
                orderby="epoch asc",
                format="json",
            ),
            "TLE history",
        )


def _parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    dt: datetime
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            dt = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().upper() in {"Y", "YES", "TRUE", "1", "T"}


def _hbr_from_rcs(rcs: str) -> float:
    mapping = {"SMALL": 2.0, "MEDIUM": 4.0, "LARGE": 6.0}
    return mapping.get((rcs or "").strip().upper(), 4.0)


def _combined_hbr_m(raw: dict[str, Any]) -> float:
    sat1_excl_vol = _to_float(raw.get("SAT_1_EXCL_VOL"), default=0.0)
    sat2_excl_vol = _to_float(raw.get("SAT_2_EXCL_VOL"), default=0.0)

    if sat1_excl_vol > 0.5 and sat2_excl_vol > 0.5:
        return sat1_excl_vol + sat2_excl_vol

    sat1_rcs = str(raw.get("SAT1_RCS") or "")
    sat2_rcs = str(raw.get("SAT2_RCS") or "")
    return _hbr_from_rcs(sat1_rcs) + _hbr_from_rcs(sat2_rcs)


def _normalize_cdm_record(raw: dict[str, Any]) -> CDMEvent:
    created_at = _parse_utc_datetime(str(raw["CREATED"]))
    tca = _parse_utc_datetime(str(raw["TCA"]))
    lead_time_hours = (tca - created_at).total_seconds() / 3600.0

    return CDMEvent(
        cdm_id=str(raw.get("CDM_ID", "")),
        created_at=created_at,
        tca=tca,
        lead_time_hours=lead_time_hours,
        miss_distance_m=_to_float(raw.get("MIN_RNG"), default=0.0),
        pc=_to_float(raw.get("PC"), default=0.0),
        sat1_id=str(raw.get("SAT_1_ID", "")),
        sat1_name=str(raw.get("SAT_1_NAME", "")),
        sat1_type=str(raw.get("SAT1_OBJECT_TYPE", "UNKNOWN")).strip().upper() or "UNKNOWN",
        sat2_id=str(raw.get("SAT_2_ID", "")),
        sat2_name=str(raw.get("SAT_2_NAME", "")),
        sat2_type=str(raw.get("SAT2_OBJECT_TYPE", "UNKNOWN")).strip().upper() or "UNKNOWN",
        combined_hbr_m=_combined_hbr_m(raw),
        emergency_flag=_to_bool(raw.get("EMERGENCY_REPORTABLE")),
    )


def _fetch_cdm_events(
    data_source: SpaceTrackDataSource,
    days_back: int,
    pc_min: float,
    limit: int,
) -> list[CDMEvent]:
    rows = data_source.fetch_cdm_rows(days_back=days_back, pc_min=pc_min, limit=limit)

    prefixes = ["STARLINK", "ONEWEB", "FLOCK", "LEMUR", "IRIDIUM", "SPIRE", "ORBCOMM"]
    counts = {prefix: 0 for prefix in prefixes}
    for row in rows:
        sat1_name = str(row.get("SAT_1_NAME", "")).strip().upper()
        for prefix in prefixes:
            if sat1_name.startswith(prefix):
                counts[prefix] += 1
                break

    print("\n=== RAW CDM PREFIX BREAKDOWN (BEFORE FILTERS) ===")
    print(f"Raw CDM records fetched: {len(rows)}")
    for prefix in prefixes:
        print(f"{prefix}: {counts[prefix]}")

    rows = [
        row
        for row in rows
        if str(row.get("SAT_1_NAME", "")).strip().upper().startswith(("IRIDIUM", "ORBCOMM"))
    ]
    print(f"Subset for this run (IRIDIUM/ORBCOMM): {len(rows)}")

    events: list[CDMEvent] = []
    for row in rows:
        try:
            events.append(_normalize_cdm_record(row))
        except (KeyError, ValueError, TypeError):
            continue

    filtered_events: list[CDMEvent] = []
    allowed_secondary_types = {"DEBRIS", "ROCKET BODY", "UNKNOWN"}
    for event in events:
        if event.sat1_type != "PAYLOAD":
            continue
        sat1_name_upper = event.sat1_name.upper()
        if not any(token in sat1_name_upper for token in ALLOWED_PRIMARY_NAME_TOKENS):
            continue
        if not (1e-4 <= event.pc <= 1e-3):
            continue
        if event.sat2_type not in allowed_secondary_types:
            continue
        filtered_events.append(event)
    return filtered_events


def _build_tle_index(tle_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tle_by_norad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tle_rows:
        norad = str(row.get("NORAD_CAT_ID", "")).strip()
        if not norad:
            continue
        tle_by_norad[norad].append(row)

    for norad, rows in tle_by_norad.items():
        rows.sort(key=lambda item: _parse_utc_datetime(str(item.get("EPOCH", "1970-01-01T00:00:00"))))
        tle_by_norad[norad] = rows
    return tle_by_norad


def _detect_maneuver(event: CDMEvent, tle_by_norad: dict[str, list[dict[str, Any]]]) -> ManeuverDetection:
    start = event.tca - timedelta(hours=48)
    end = event.tca + timedelta(hours=48)
    rows = tle_by_norad.get(event.sat1_id, [])

    before_row: dict[str, Any] | None = None
    after_row: dict[str, Any] | None = None
    before_epoch: datetime | None = None
    after_epoch: datetime | None = None

    for row in rows:
        epoch_raw = row.get("EPOCH")
        if not epoch_raw:
            continue
        epoch = _parse_utc_datetime(str(epoch_raw))
        if epoch < start or epoch > end:
            continue
        if epoch < event.tca:
            before_row = row
            before_epoch = epoch
        elif epoch > event.tca and after_row is None:
            after_row = row
            after_epoch = epoch
            break

    if before_row is None or after_row is None or before_epoch is None or after_epoch is None:
        return ManeuverDetection(
            maneuver_detected=False,
            confidence="INSUFFICIENT_DATA",
            delta_mean_motion=None,
            delta_eccentricity=None,
            tle_gap_hours=None,
        )

    delta_mean_motion = abs(
        _to_float(after_row.get("MEAN_MOTION"), default=0.0)
        - _to_float(before_row.get("MEAN_MOTION"), default=0.0)
    )
    delta_eccentricity = abs(
        _to_float(after_row.get("ECCENTRICITY"), default=0.0)
        - _to_float(before_row.get("ECCENTRICITY"), default=0.0)
    )

    mean_motion_threshold = 2e-4
    eccentricity_threshold = 1e-4

    detected = (
        delta_mean_motion > mean_motion_threshold
        or delta_eccentricity > eccentricity_threshold
    )
    tle_gap_hours = (after_epoch - before_epoch).total_seconds() / 3600.0

    if not detected:
        confidence = "LOW"
    else:
        high_delta = (
            delta_mean_motion > (2 * mean_motion_threshold)
            or delta_eccentricity > (2 * eccentricity_threshold)
        )
        confidence = "HIGH" if tle_gap_hours < 12 and high_delta else "MEDIUM"

    return ManeuverDetection(
        maneuver_detected=detected,
        confidence=confidence,
        delta_mean_motion=delta_mean_motion,
        delta_eccentricity=delta_eccentricity,
        tle_gap_hours=tle_gap_hours,
    )


def _shiro_policy(event: CDMEvent) -> PolicyDecision:
    if event.pc >= 1e-3:
        return PolicyDecision(True, "EMERGENCY threshold exceeded.")
    if event.pc < 1e-4:
        return PolicyDecision(False, "below maneuver threshold.")
    if event.sat1_type == "DEBRIS" and event.sat2_type == "DEBRIS":
        return PolicyDecision(False, "no actionable object.")
    if event.sat1_type == "UNKNOWN":
        return PolicyDecision(False, "primary not maneuver-capable.")
    if event.combined_hbr_m > 0 and (event.miss_distance_m / event.combined_hbr_m) > 30:
        return PolicyDecision(False, "geometry does not support maneuver.")
    if event.lead_time_hours > 24 and event.pc < 3e-4:
        return PolicyDecision(False, "marginal Pc with time to await updated CDM.")
    return PolicyDecision(True, "default policy action.")


def _classify_outcome(
    industry_acted: bool,
    recommends_act: bool,
    confidence: str,
) -> tuple[str, bool]:
    if confidence == "INSUFFICIENT_DATA":
        return "UNKNOWN", False
    if industry_acted and recommends_act:
        return "AGREE_ACT", False
    if industry_acted and not recommends_act:
        return "UNNECESSARY", True
    if not industry_acted and recommends_act:
        return "SHIRO_ONLY_ACT", False
    return "AGREE_PASS", False


def _result_to_json_dict(result: ValidatorResult) -> dict[str, Any]:
    event_data = asdict(result.cdm_event)
    event_data["created_at"] = result.cdm_event.created_at.isoformat()
    event_data["tca"] = result.cdm_event.tca.isoformat()

    return {
        **event_data,
        "maneuver_detected": result.maneuver_detection.maneuver_detected,
        "maneuver_confidence": result.maneuver_detection.confidence,
        "delta_mean_motion": result.maneuver_detection.delta_mean_motion,
        "delta_eccentricity": result.maneuver_detection.delta_eccentricity,
        "tle_gap_hours": result.maneuver_detection.tle_gap_hours,
        "industry_acted": result.industry_acted,
        "shiro_recommends_act": result.policy_decision.recommends_act,
        "shiro_rationale": result.policy_decision.rationale,
        "outcome": result.outcome,
        "unnecessary": result.unnecessary,
    }


def _build_summary(results: list[ValidatorResult]) -> dict[str, Any]:
    total_events = len(results)
    industry_acted = sum(1 for r in results if r.industry_acted)
    unnecessary_maneuvers = sum(1 for r in results if r.unnecessary)
    agree_act = sum(1 for r in results if r.outcome == "AGREE_ACT")
    shiro_only_act = sum(1 for r in results if r.outcome == "SHIRO_ONLY_ACT")
    unknown = sum(1 for r in results if r.outcome == "UNKNOWN")

    unnecessary_rate = (
        unnecessary_maneuvers / industry_acted if industry_acted > 0 else 0.0
    )

    return {
        "total_events": total_events,
        "industry_acted": industry_acted,
        "unnecessary_maneuvers": unnecessary_maneuvers,
        "unnecessary_maneuver_rate": unnecessary_rate,
        "agree_act": agree_act,
        "shiro_only_act": shiro_only_act,
        "unknown": unknown,
        "policy_version": POLICY_VERSION,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_outputs(results: list[ValidatorResult], output_dir: str) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    full_path = output_path / f"shiro_validator_{timestamp}.json"
    summary_path = output_path / f"shiro_validator_summary_{timestamp}.json"

    full_payload = [_result_to_json_dict(r) for r in results]
    summary_payload = _build_summary(results)

    full_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return full_path, summary_path


def _print_results(results: list[ValidatorResult], request_count: int) -> None:
    summary = _build_summary(results)
    unnecessary_rows = [r for r in results if r.outcome == "UNNECESSARY"]

    print("\n=== SHIRO:VALIDATOR RESULTS ===")
    print(f"Total events processed: {summary['total_events']}")
    print(f"Industry acted count: {summary['industry_acted']}")
    print(
        "Unnecessary maneuver count and rate: "
        f"{summary['unnecessary_maneuvers']} "
        f"({summary['unnecessary_maneuver_rate']:.3f})"
    )
    print(f"SpaceTrack API requests this run: {request_count}")

    print("\nUNNECESSARY events")
    if not unnecessary_rows:
        print("(none)")
        return

    print(
        "CDM_ID | SAT1_NAME | SAT2_NAME | PC | MISS_DISTANCE_M | "
        "LEAD_TIME_HOURS | SHIRO_RATIONALE"
    )
    for row in unnecessary_rows:
        event = row.cdm_event
        print(
            f"{event.cdm_id} | "
            f"{event.sat1_name} | "
            f"{event.sat2_name} | "
            f"{event.pc:.6g} | "
            f"{event.miss_distance_m:.3f} | "
            f"{event.lead_time_hours:.2f} | "
            f"{row.policy_decision.rationale}"
        )


def run_validator(
    days_back: int,
    pc_min: float,
    limit: int,
    industry_acted_sample_size: int,
    output_dir: str,
    cache_dir: str = ".cache",
) -> list[ValidatorResult]:
    user = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")
    if not user or not password:
        raise RuntimeError(
            "Missing SpaceTrack credentials: set SPACETRACK_USER and SPACETRACK_PASS"
        )

    data_source = SpaceTrackDataSource(user=user, password=password, cache_dir=cache_dir)
    events = _fetch_cdm_events(
        data_source=data_source,
        days_back=days_back,
        pc_min=pc_min,
        limit=limit,
    )

    tle_by_norad: dict[str, list[dict[str, Any]]] = {}
    unique_sat1_ids = sorted({event.sat1_id for event in events if event.sat1_id})
    if unique_sat1_ids and events:
        earliest_tca = min(event.tca for event in events)
        latest_tca = max(event.tca for event in events)
        tle_rows = data_source.fetch_batched_tle_rows(
            norad_ids=unique_sat1_ids,
            earliest_tca=earliest_tca,
            latest_tca=latest_tca,
        )
        tle_by_norad = _build_tle_index(tle_rows)

    results: list[ValidatorResult] = []
    acted_count = 0

    for event in events:
        maneuver = _detect_maneuver(event, tle_by_norad)
        industry_acted = maneuver.maneuver_detected and maneuver.confidence in {"HIGH", "MEDIUM"}
        if industry_acted:
            acted_count += 1

        policy = _shiro_policy(event)
        outcome, unnecessary = _classify_outcome(
            industry_acted=industry_acted,
            recommends_act=policy.recommends_act,
            confidence=maneuver.confidence,
        )

        results.append(
            ValidatorResult(
                cdm_event=event,
                maneuver_detection=maneuver,
                industry_acted=industry_acted,
                policy_decision=policy,
                outcome=outcome,
                unnecessary=unnecessary,
            )
        )

        if acted_count >= industry_acted_sample_size:
            break

    _write_outputs(results, output_dir=output_dir)
    _print_results(results, request_count=data_source.request_count)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHIRO validator pipeline")
    parser.add_argument("--days", type=int, default=365, help="Days back for CDM query")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum CDM records")
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Stop after this many industry_acted events",
    )
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="SpaceTrack cache directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_validator(
        days_back=args.days,
        pc_min=1e-4,
        limit=args.limit,
        industry_acted_sample_size=args.sample,
        output_dir=args.output,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
