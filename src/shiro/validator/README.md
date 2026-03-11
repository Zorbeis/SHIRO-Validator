# SHIRO Validator Pipeline

This module runs the SHIRO:VALIDATOR pipeline to estimate how many industry maneuvers were unnecessary.

## Requirements

- Python 3.10+
- `requests`
- SpaceTrack credentials in environment variables:
  - `SPACETRACK_USER`
  - `SPACETRACK_PASS`

## Run

```bash
export SPACETRACK_USER="x"
export SPACETRACK_PASS="y"
python src/shiro/validator/pipeline.py --days 90 --limit 200 --sample 50 --output outputs
```

## CLI Args

- `--days`: days back for CDM query (default: `90`)
- `--limit`: max CDM records to fetch (default: `200`)
- `--sample`: stop after this many `industry_acted=True` events (default: `50`)
- `--output`: output directory for JSON files (default: `outputs`)

## Outputs

- `shiro_validator_<timestamp>.json`: full event-by-event audit log
- `shiro_validator_summary_<timestamp>.json`: aggregate metrics summary

## Import Usage

```python
from src.shiro.validator.pipeline import run_validator

results = run_validator(
    days_back=90,
    pc_min=1e-4,
    limit=200,
    industry_acted_sample_size=50,
    output_dir="outputs",
)
```
