from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import spacetrack.operators as op
from spacetrack import SpaceTrackClient


# SpaceTrack API limits: 30 req/min, 300 req/hour
# This pipeline uses batched queries - total requests per run: ~2
# Do NOT add loops that make per-object API calls
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


def parse_utc_datetime(value: str) -> datetime:
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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().upper() in {"Y", "YES", "TRUE", "1", "T"}


def hbr_from_rcs(rcs: str) -> float:
    mapping = {"SMALL": 2.0, "MEDIUM": 4.0, "LARGE": 6.0}
    return mapping.get((rcs or "").strip().upper(), 4.0)


def combined_hbr_m(raw: dict[str, Any]) -> float:
    sat1_excl_vol = to_float(raw.get("SAT_1_EXCL_VOL"), default=0.0)
    sat2_excl_vol = to_float(raw.get("SAT_2_EXCL_VOL"), default=0.0)

    if sat1_excl_vol > 0.5 and sat2_excl_vol > 0.5:
        return sat1_excl_vol + sat2_excl_vol

    sat1_rcs = str(raw.get("SAT1_RCS") or "")
    sat2_rcs = str(raw.get("SAT2_RCS") or "")
    return hbr_from_rcs(sat1_rcs) + hbr_from_rcs(sat2_rcs)


def normalize_cdm_record(raw: dict[str, Any]) -> CDMEvent:
    created_at = parse_utc_datetime(str(raw["CREATED"]))
    tca = parse_utc_datetime(str(raw["TCA"]))
    lead_time_hours = (tca - created_at).total_seconds() / 3600.0

    return CDMEvent(
        cdm_id=str(raw.get("CDM_ID", "")),
        created_at=created_at,
        tca=tca,
        lead_time_hours=lead_time_hours,
        miss_distance_m=to_float(raw.get("MIN_RNG"), default=0.0),
        pc=to_float(raw.get("PC"), default=0.0),
        sat1_id=str(raw.get("SAT_1_ID", "")),
        sat1_name=str(raw.get("SAT_1_NAME", "")),
        sat1_type=str(raw.get("SAT1_OBJECT_TYPE", "UNKNOWN")).strip().upper() or "UNKNOWN",
        sat2_id=str(raw.get("SAT_2_ID", "")),
        sat2_name=str(raw.get("SAT_2_NAME", "")),
        sat2_type=str(raw.get("SAT2_OBJECT_TYPE", "UNKNOWN")).strip().upper() or "UNKNOWN",
        combined_hbr_m=combined_hbr_m(raw),
        emergency_flag=to_bool(raw.get("EMERGENCY_REPORTABLE")),
    )


def fetch_cdm_events(
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
            events.append(normalize_cdm_record(row))
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


def build_tle_index(tle_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    from collections import defaultdict

    tle_by_norad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tle_rows:
        norad = str(row.get("NORAD_CAT_ID", "")).strip()
        if not norad:
            continue
        tle_by_norad[norad].append(row)

    for norad, rows in tle_by_norad.items():
        rows.sort(key=lambda item: parse_utc_datetime(str(item.get("EPOCH", "1970-01-01T00:00:00"))))
        tle_by_norad[norad] = rows
    return tle_by_norad
