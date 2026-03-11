# SHIRO:VALIDATOR

**Covariance-realism-corrected collision avoidance decision framework for LEO constellation operators.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## The Problem

Operational SSA covariances are systematically underconfident by 3-5x in position
(Hejduk et al. 2004). This inflates Pc estimates and causes operators to maneuver
when genuine risk does not warrant it.

SpaceX responded by dropping their internal maneuver threshold to 1x10^-6 -
100x below the NASA CARA standard - and executed over 144,000 avoidance maneuvers
in a single six-month period. Not because space got more dangerous.
Because the Pc estimates couldn't be trusted.

**SHIRO proposes a different fix: correct the measurement, not the threshold.**

---

## The Approach

SHIRO applies a literature-grounded covariance realism correction before
evaluating the standard 1x10^-4 maneuver threshold:

```text
Pc_industry = MonteCarlo(miss_2d, C_reported, HBR)
Pc_shiro    = MonteCarlo(miss_2d, k^2 x C_reported, HBR)

k = 3 to 5  (position underconfidence factor, Hejduk et al. 2004)
k^2 = 9 to 25 (matrix inflation factor)
```

Same threshold. More accurate Pc. Fewer unnecessary maneuvers.

---

## Results

Across 50 synthetic LEO conjunction events with randomized parameters:

| Inflation Factor | Unnecessary Maneuver Rate | Missed Collisions |
|---|---|---|
| 9x (k=3, conservative) | 40.0% | 0 |
| 16x (k=4, median) | 71.1% | 0 |
| 25x (k=5, pessimistic) | 86.7% | 0 |

SHIRO never passes on an event where the covariance-corrected Pc exceeds 1x10^-4.
The expected collisions from all SHIRO passes across the 50-event batch: **3.6x10^-4**.

> Current validation is synthetic only. Real CDM validation is in progress
> pending Space-Track data access approval.

---

## Visualizations

### B-Plane Comparison - Industry vs SHIRO
![B-Plane](docs/figures/bplane_disagree.png)
*Left: Industry view - tight covariance, Pc = 7.0x10^-4 -> ACT*
*Right: SHIRO view - corrected covariance, Pc = 2.8x10^-5 -> PASS*

### 3D Orbital Geometry
![3D Orbit](docs/figures/3d_orbit.png)
*Cross-track conjunction between 51.6 deg and 97.8 deg inclined LEO orbits*

### Batch Sensitivity Results
![Batch Results](docs/figures/batch_sensitivity.png)

---

## Quickstart

```bash
git clone https://github.com/Zorbeis/SHIRO-Validator
cd SHIRO-Validator
pip install -r requirements.txt

# Run a single conjunction scenario
python src/shiro/validator/visualize.py

# Run batch validation (50 events, ~5 mins)
python src/shiro/validator/batch_runner.py
```

Output files open automatically in your browser.

---

## Repository Structure

```text
src/shiro/validator/
|- cdm_from_stresslab.py   # Synthetic CDM generator + J2 propagator
|- visualize.py            # B-plane + Pc timeline (Plotly)
|- render_3d.py            # 3D orbital visualization (Three.js)
`- batch_runner.py         # Multi-event batch policy engine

METHODOLOGY.md             # Full technical documentation
outputs/                   # HTML visualizations + batch results
```

---

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for full technical documentation including:

- J2-perturbed RK4 orbital propagator
- B-plane projection and Monte Carlo Pc estimation
- Covariance realism correction derivation
- Safety invariant verification
- Batch validation results and sensitivity analysis

---

## References

- Hejduk et al. (2004) - Conjunction assessment risk analysis, AAS 04-170
- Carpenter et al. (2018) - Covariance realism for SSA, JGCD 41(3)
- Hejduk and Snow (2019) - Satellite conjunction assessment for dilution of precision covariances
- NASA CARA (2014) - Recommended Practices for Satellite Operators

---

## Status

| Component | Status |
|---|---|
| Synthetic CDM engine | Complete |
| B-plane visualization | Complete |
| 3D orbital visualization | Complete |
| Batch runner + sensitivity analysis | Complete |
| Real CDM validation | Pending data access |
| SHIRO:CONSTELLATION (live fleet) | Planned |

---

## License

MIT - see [LICENSE](LICENSE)

*SHIRO is research software. Not for operational use without real CDM validation.*
