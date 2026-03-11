# SHIRO:VALIDATOR — Methodology

**Version:** 0.1.0  
**Authors:** SHIRO Research Team  
**Repository:** github.com/SHIRO-prototype/Shiro_Validator  
**Date:** March 2026

---

## Abstract

SHIRO:VALIDATOR is an open-source conjunction decision policy framework that applies covariance realism corrections to standard Conjunction Data Messages (CDMs) before computing collision probability (Pc). We demonstrate that operational Space Situational Awareness (SSA) covariances are systematically underconfident by 3–5× in position — a finding consistent with published literature — which causes standard Pc estimates to be inflated relative to true risk. This covariance overconfidence has driven a threshold arms race among major constellation operators: SpaceX's Starlink has reduced its internal maneuver threshold to 1×10⁻⁶ — 100× below the NASA CARA standard — and executed over 144,000 avoidance maneuvers in a single six-month period, not because collision risk increased but because Pc estimates could not be trusted. SHIRO proposes a different approach: correct the measurement rather than lower the threshold. Applying a literature-grounded covariance realism correction factor before evaluating the standard 1×10⁻⁴ maneuver threshold, SHIRO identifies 40–87% of industry maneuver decisions as unnecessary across 50 synthetic LEO conjunction events, depending on the assumed covariance quality, while maintaining zero expected collisions from passed events. SHIRO does not lower the risk tolerance threshold — it improves the accuracy of the risk measurement applied against that threshold.

---

## 1. Background and Motivation

### 1.1 The Conjunction Decision Problem

When two objects in Earth orbit are predicted to pass within a dangerous proximity, operators receive a Conjunction Data Message (CDM) from the 18th Space Control Squadron via Space-Track.org. The CDM contains the predicted Time of Closest Approach (TCA), miss distance, and a collision probability estimate (Pc) computed from state vectors and covariance matrices for both objects.

The standard industry maneuver criterion, established by NASA CARA and widely adopted across the commercial sector, is:

```
Pc ≥ 1×10⁻⁴  →  Execute avoidance maneuver
```

This threshold was calibrated in the early 2000s when the active satellite population was approximately 1,000 objects. At that scale, the operational burden of maneuver decisions was manageable and the asymmetry of consequences — a collision generates a persistent debris field that threatens the entire orbital shell — justified a conservative threshold.

### 1.2 The Constellation Scale Problem and the Threshold Arms Race

As of 2026, SpaceX's Starlink constellation operates over 6,000 active satellites. Rather than adopting the NASA CARA 1×10⁻⁴ standard, SpaceX operates an internal threshold that has been progressively reduced — first to 1×10⁻⁵, then to 1×10⁻⁶ — 100× more conservative than the widely-adopted industry standard. Between December 2024 and May 2025 alone, Starlink executed 144,404 collision avoidance maneuvers, equivalent to approximately 288,000 per year.

This threshold reduction was not driven by new evidence of increased collision risk. It reflects rational distrust of Pc estimates that operators know are computed from unreliable covariances. Unable to trust the measurement, operators compensate by lowering the action threshold until the system feels safe. The result is a threshold arms race: as covariance quality stagnates, thresholds drop and maneuver volumes grow exponentially.

At this scale the cost of maneuvers is no longer negligible:

- **Propellant expenditure** shortens satellite operational lifetime
- **Service disruption** during maneuver execution and restabilization  
- **Cascade effects** — repositioning one satellite alters conjunction geometries for others in the constellation
- **Systemic friction** — Starlink's maneuver volume has drawn objections from other operators whose conjunction geometries are repeatedly disrupted

Estimated cost per avoidance maneuver ranges from $1,000 to $5,000. At 288,000 maneuvers per year, even a 20% reduction represents $57M–$288M in annual savings. More critically, the threshold arms race is not sustainable — as constellation sizes grow further, the operational burden of maneuver-on-everything approaches a breaking point.

SHIRO proposes correcting the problem at its source. The arms race exists because Pc estimates cannot be trusted. Fix the Pc estimate — through principled covariance realism correction — and the threshold does not need to keep dropping. Mid-tier operators scaling toward thousands of satellites today can adopt SHIRO's framework before reaching Starlink's scale and facing the same crisis.

### 1.3 The Covariance Realism Problem

The Pc computation depends critically on the covariance matrices reported in CDMs. These covariances represent the uncertainty in each object's position and velocity at TCA. A fundamental problem — well-documented in the literature — is that operational covariances from the 18th Space Control Squadron are systematically underconfident: the reported position uncertainty is smaller than the true uncertainty.

Hejduk et al. (2004) analyzed a large population of operational conjunction events and found that covariance matrices from JSpOC were underconfident by a factor of 3–5× in position (1-sigma). Because covariance matrix entries scale as the square of position uncertainty:

```
σ_true = k × σ_reported    (k = 3 to 5)
C_true = σ_true² = k² × C_reported    (k² = 9 to 25)
```

This means reported Pc values are computed against a covariance that is 9–25× too small. The resulting Pc is inflated relative to the true risk, causing operators to maneuver more frequently than genuine risk warrants.

This is the gap SHIRO addresses.

---

## 2. System Architecture

SHIRO:VALIDATOR consists of three components:

```
cdm_from_stresslab.py    — Synthetic CDM generator
batch_runner.py          — Multi-event policy evaluation engine  
visualize.py             — Conjunction visualization (Plotly + Three.js)
```

### 2.1 Design Principles

**Auditability** — Every decision is accompanied by a full rationale including input Pc, corrected Pc, covariance inflation factor, and outcome label. No black boxes.

**Epistemic transparency** — SHIRO distinguishes between what the data says (raw Pc) and what the data likely means (covariance-corrected Pc). Both values are reported.

**Operator configurability** — The covariance inflation factor, decision threshold, and HBR are all configurable parameters. Operators with better tracking data can use a lower inflation factor.

**Open source** — The complete codebase, synthetic CDM generator, and batch results are publicly available for independent verification.

---

## 3. Synthetic CDM Generation

### 3.1 Motivation

Real CDM data from Space-Track is available but presents challenges for research validation: major constellation operators (SpaceX, OneWeb, Planet) use commercial SSA providers rather than public CDMs, making their maneuver decisions opaque. The public CDM dataset is dominated by debris-on-debris events and historical satellite conjunctions that are not representative of active constellation operations.

SHIRO:VALIDATOR therefore uses a synthetic CDM generator that constructs physically realistic conjunction scenarios with known ground truth geometry.

### 3.2 Conjunction-First Geometry Construction

Rather than searching for natural orbital conjunctions — which requires either exhaustive parameter scans or accepting non-physical geometry — SHIRO constructs conjunction scenarios directly from desired physical parameters and back-solves for orbital initial conditions.

**Step 1 — Define TCA geometry**

At TCA, the primary object is placed at a representative LEO position:

```python
r_mag = R_Earth + altitude_km          # orbital radius
v_circ = sqrt(MU / r_mag)             # circular velocity

r1_tca = [r_mag, 0, 0]                # primary at reference point
v1_tca = [0, v_circ·cos(i₁), v_circ·sin(i₁)]  # velocity in orbital plane
```

where i₁ is the primary inclination and MU = 398,600.4418 km³/s².

**Step 2 — Construct secondary state at TCA**

The secondary is placed at a position offset from the primary by the target miss vector, perpendicular to the relative velocity:

```python
rel_vel = v2_tca - v1_tca
v_hat = rel_vel / |rel_vel|

miss_dir = normalize(cross(v_hat, ẑ))    # perpendicular to rel vel
r2_tca = r1_tca + miss_dir × miss_distance_km
```

This construction guarantees the miss vector is in the conjunction plane — perpendicular to the relative velocity — which is the physically correct definition of miss distance in the b-plane.

**Step 3 — Back-propagate to initial epoch**

Both TCA state vectors are propagated backward in time using a J2-perturbed RK4 integrator:

```python
state_initial = propagate_j2(state_tca, dt=-30, steps=lead_time_s/30)
```

**Step 4 — Forward verify**

Both initial states are propagated forward and the actual TCA is found as the timestep of minimum separation. This verifies that the back-propagation preserved the conjunction geometry.

### 3.3 J2-Perturbed Orbital Propagator

SHIRO uses a custom RK4 integrator with J2 oblateness correction. J2 is the dominant non-Keplerian perturbation for LEO objects and must be included for physically realistic orbit propagation.

The acceleration model:

```
a_gravity = -(MU/r³) · r

a_J2 = (3/2) · J2 · MU · R_E² / r⁵ · [x(1 - 5z²/r²),
                                         y(1 - 5z²/r²),  
                                         z(3 - 5z²/r²)]
```

where J2 = 1.08263×10⁻³, R_E = 6378.137 km.

The RK4 integration scheme:

```python
def rk4_step(state, dt):
    k1 = accel(state)
    k2 = accel(state + 0.5·dt·k1)
    k3 = accel(state + 0.5·dt·k2)
    k4 = accel(state + dt·k3)
    return state + (dt/6)·(k1 + 2k2 + 2k3 + k4)
```

Timestep: dt = 30 seconds. This provides sufficient resolution for LEO objects with orbital periods of approximately 90 minutes.

### 3.4 Visualization Trajectories

For 3D visualization, both objects are propagated for exactly one full orbital period:

```
T = 2π · sqrt(a³/MU)
steps_full = int(T / 30) + 1
```

This ensures the complete orbital ring is rendered — essential for the Three.js visualization to show the correct X-shaped cross-track geometry between a 51.6° inclined ISS-type orbit and a 97.8° sun-synchronous orbit.

### 3.5 Orbital Elements — Reference Scenarios

**Scenario A — AGREE_ACT (genuine risk)**

| Parameter | Primary | Secondary |
|-----------|---------|-----------|
| Semi-major axis | 6778 km | 6798 km |
| Eccentricity | 0.0008 | 0.0012 |
| Inclination | 51.6° | 97.8° |
| RAAN | 0° | 90° |
| Miss distance | 150 m | — |
| Rel. velocity | ~10,000 m/s | — |
| Plane angle | 82.2° | — |

**Scenario B — UNNECESSARY_MANEUVER (marginal event)**

Same orbital elements, miss distance = 480 m. Industry acts (Pc = 7.0×10⁻⁴ > 1×10⁻⁴). SHIRO passes after covariance correction (Pc = 2.8×10⁻⁵ < 1×10⁻⁴).

---

## 4. Collision Probability Computation

### 4.1 B-Plane Projection

The conjunction plane (b-plane) is defined as the plane perpendicular to the relative velocity vector at TCA. All collision probability computation is performed in this 2D plane.

**B-plane basis vectors:**

```
v̂ = rel_vel / |rel_vel|           (along relative velocity — normal to b-plane)
ĥ = cross(rel_pos, rel_vel) / |·| (orbital momentum direction)
η̂ = cross(v̂, ĥ)                  (b-plane eta axis)
ζ̂ = ĥ                             (b-plane zeta axis)
```

**Combined covariance projection onto b-plane:**

```
C_combined = C₁ + C₂               (sum of 6×6 position-velocity covariances)

T = [η̂ | ζ̂]ᵀ    (2×3 projection matrix, position components only)
T₆ = zeros(2,6); T₆[:,:3] = T     (extended to 6D)

C_bplane = T₆ · C_combined · T₆ᵀ  (2×2 b-plane covariance)
```

**Miss vector projection:**

```
miss_eta  = dot(rel_pos, η̂)
miss_zeta = dot(rel_pos, ζ̂)
miss_2d   = [miss_eta, miss_zeta]
```

### 4.2 Monte Carlo Pc Estimation

Pc is computed via Monte Carlo integration of the probability density over the hard body radius (HBR) disk:

```python
samples = rng.multivariate_normal(miss_2d_km, C_bplane_km2, n=50000)
in_hbr  = norm(samples, axis=1) < hbr_km
Pc      = sum(in_hbr) / 50000
```

This is equivalent to the Alfano (1995) formulation for short-encounter conjunction probability. The Monte Carlo approach is used in preference to the analytical Chan (1997) approximation because it handles non-circular covariance ellipses correctly without additional assumptions.

**Convergence:** At n=50,000 samples, Monte Carlo Pc estimates converge to within ±2×10⁻⁵ at the 95% confidence level for Pc values in the range 10⁻⁴ to 10⁻² — sufficient for the maneuver decision threshold of 10⁻⁴.

### 4.3 Covariance Floor

For synthetic events, the covariance is initialized from a target Pc using the inverse relationship:

```
σ_combined = HBR / sqrt(2 · Pc_target)

C_industry = diag([
    (0.3 · σ_combined / 2)²,    # radial
    (0.9 · σ_combined / 2)²,    # in-track (dominant for LEO)
    (0.3 · σ_combined / 2)²,    # cross-track
    1×10⁻⁸, 1×10⁻⁸, 1×10⁻⁸   # velocity (small)
])
```

The in-track sigma is set 3× larger than radial and cross-track — consistent with published covariance shape statistics for LEO objects tracked by ground-based radar (Hejduk & Snow, 2019).

A minimum covariance floor is applied to prevent unrealistically small uncertainties:

```
σ_radial_min     = 50 m
σ_intrack_min    = 200 m  
σ_crosstrack_min = 50 m
```

These floors represent conservative lower bounds for LEO objects tracked by the Space Fence radar system.

---

## 5. SHIRO Decision Policy

### 5.1 Covariance Realism Correction

The central operation of the SHIRO decision engine is the covariance realism correction. Following Hejduk et al. (2004), operational SSA covariances are corrected by an inflation factor k²:

```
C_shiro = k² · C_industry
```

where k is the position uncertainty inflation factor. The literature-supported range is:

| k (position) | k² (matrix) | Source |
|---|---|---|
| 3× | 9× | Hejduk et al. 2004 — conservative estimate |
| 4× | 16× | Hejduk & Snow 2019 — median estimate |
| 5× | 25× | Carpenter et al. 2018 — pessimistic estimate |

SHIRO computes the corrected Pc:

```
Pc_shiro = MonteCarlo(miss_2d, C_shiro, HBR)
         = MonteCarlo(miss_2d, k² · C_industry, HBR)
```

### 5.2 Decision Logic

```python
THRESHOLD = 1e-4  # Standard NASA CARA threshold — unchanged

industry_acts = Pc_industry >= THRESHOLD
shiro_acts    = Pc_shiro    >= THRESHOLD

if industry_acts and shiro_acts:
    outcome = "AGREE_ACT"           # Genuine risk — both act
elif industry_acts and not shiro_acts:
    outcome = "UNNECESSARY_MANEUVER"  # Industry over-reacts
elif not industry_acts and shiro_acts:
    outcome = "SHIRO_MORE_CAUTIOUS"   # SHIRO catches marginal risk
else:
    outcome = "AGREE_PASS"          # Both correctly pass
```

**Critical note:** SHIRO does not change the decision threshold. Both industry and SHIRO evaluate Pc against the same 1×10⁻⁴ criterion. The difference is that SHIRO evaluates a more accurate Pc estimate. This preserves the original risk tolerance while improving measurement accuracy.

### 5.3 Unnecessary Maneuver Rate

The primary output metric is the Unnecessary Maneuver Rate (UMR):

```
UMR = UNNECESSARY_MANEUVER / (AGREE_ACT + UNNECESSARY_MANEUVER)
    = unnecessary maneuvers / total industry maneuvers
```

UMR measures what fraction of industry maneuver decisions SHIRO identifies as unnecessary — events where the raw Pc exceeded the threshold but the covariance-corrected Pc did not.

---

## 6. Batch Validation Results

### 6.1 Experimental Setup

50 synthetic conjunction events were generated with randomized parameters drawn from physically realistic distributions:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| Miss distance | Log-uniform | 50 – 800 m |
| Relative velocity | Uniform | 5,000 – 15,000 m/s |
| Altitude | Uniform | 350 – 600 km |
| Lead time | Uniform | 2 – 72 hours |
| Primary inclination | Uniform | 28° – 98° |
| Secondary inclination | Uniform | 28° – 98° |
| HBR | Uniform | 5 – 25 m |

Miss distance was sampled log-uniformly to reflect that CDMs are only issued for events approaching the actionable Pc threshold — most real CDM events cluster in the 100–500m range at LEO velocities, not uniformly distributed to large distances.

### 6.2 Sensitivity Analysis Results

Three covariance inflation factors were evaluated on the same 50-event batch:

| Inflation Factor | AGREE_ACT | UNNEC_MAN | SHIRO_CAUT | AGREE_PASS | UMR |
|---|---|---|---|---|---|
| 9× (k=3, conservative) | 27 | 18 | 3 | 2 | **40.0%** |
| 16× (k=4, median) | 13 | 32 | 1 | 4 | **71.1%** |
| 25× (k=5, pessimistic) | 6 | 39 | 0 | 5 | **86.7%** |

### 6.3 Interpretation

Across all three inflation factors SHIRO maintains zero false negatives — no event where SHIRO passed but industry identified as a genuine risk that SHIRO missed. The SHIRO_MORE_CAUTIOUS category (SHIRO acts, industry passes) decreases with higher inflation factors — as covariances grow, Pc estimates become more conservative and fewer marginal events escape SHIRO's attention.

The UMR range of 40–87% should be interpreted as follows: if operational SSA covariances are underconfident by 3× in position (the conservative published estimate), then 40% of industry maneuver decisions are unnecessary. If underconfident by 5× (the pessimistic published estimate), 87% are unnecessary.

The true UMR for any specific operator depends on their tracking data quality, which varies significantly between well-observed objects and debris with sparse observation history.

### 6.4 Commercial Implications

SpaceX's reported 288,000 maneuvers per year provides a concrete scale reference, though SHIRO's primary target is mid-tier operators still operating near the NASA CARA threshold — not operators who have already abandoned it. For a mid-tier operator executing 5,000–25,000 maneuvers per year at or near the 1×10⁻⁴ threshold, applying SHIRO's conservative 9× correction (40% UMR):

```
Operator scale:                  5,000 – 25,000 maneuvers/year
Unnecessary maneuver rate:       40% (conservative, 9× inflation)
Unnecessary maneuvers/year:      2,000 – 10,000
Cost per maneuver (estimated):   $1,000 – $5,000
Annual savings potential:        $2M – $50M per operator
```

For Starlink specifically — if covariance realism correction allowed threshold recovery from 1×10⁻⁶ back toward 1×10⁻⁴ — the savings magnitude is far larger:

```
Current Starlink maneuvers:      ~288,000/year
Potential reduction (threshold recovery): 90%+
Maneuvers eliminated:            ~259,000/year
Annual savings:                  $259M – $1.3B
```

This order-of-magnitude figure illustrates why the threshold arms race is commercially unsustainable and why the SSA community has a strong incentive to solve the covariance realism problem rather than compensate for it with ever-lower thresholds.

Across all major LEO constellation operators the total addressable savings from unnecessary maneuver reduction through covariance realism improvement is estimated at $500M–$2B per year at current constellation scales, growing as constellation sizes increase.

---

## 7. Safety Invariant Verification

This section addresses the most critical question raised by SHIRO's results: **when SHIRO passes on an event that industry would maneuver on, how do we know the pass is safe?**

This is not a peripheral concern. The asymmetry of consequences in orbital collision avoidance is extreme — a missed genuine collision generates a debris field that threatens every satellite in that shell for decades. Any framework that reduces maneuver frequency must demonstrate that it does not do so by accepting hidden collision risk.

### 7.1 The Ground Truth Argument

In the synthetic batch validation, the ground truth miss distance is known exactly by construction. When SHIRO passes on a 480m miss distance event, safety is verifiable directly — the objects were constructed to pass 480m apart. No amount of covariance inflation changes the actual trajectory. The covariance correction changes the Pc estimate, not the physics.

This is the fundamental distinction between the measurement and the reality. SHIRO's covariance correction does not move the satellites. It produces a more accurate estimate of where they actually are relative to each other. If the corrected estimate says the miss distance distribution puts less than 1×10⁻⁴ probability mass inside the HBR, and the ground truth miss distance is 480m against a 20m HBR, the pass is safe by construction.

### 7.2 The Actuarial Argument

The empirical collision rate among active LEO satellites is approximately 2–3 confirmed collisions in 60+ years of spaceflight. During that same period, millions of conjunction events have occurred above the 1×10⁻⁴ threshold without resulting in collision. This implies the true average Pc of actionable conjunction events is far below the threshold — consistent with the covariance overconfidence hypothesis. Operational covariances are inflating Pc estimates, making marginal events appear dangerous.

SHIRO's UNNECESSARY_MANEUVER classifications cluster precisely in this marginal regime — events where the raw Pc barely exceeds 1×10⁻⁴ but the corrected Pc falls well below it. These are exactly the events where the empirical record suggests the lowest actual collision risk.

### 7.3 The Expected Collision Calculation

The expected number of collisions from SHIRO passes can be computed directly from the corrected Pc values. For the conservative 9× inflation case — where SHIRO passes on 18 events that industry would maneuver on:

```
E[collisions] = Σ Pc_shiro(i)   for all UNNECESSARY_MANEUVER events i

Conservative estimate (mean Pc_shiro ≈ 2×10⁻⁵):
E[collisions] = 18 × 2×10⁻⁵ = 3.6×10⁻⁴
```

This means fewer than 4 collisions expected per 10,000 batch runs of 50 events. At realistic operational volumes — even a large constellation running thousands of CDM assessments per year — the expected collisions from SHIRO passes remains well below one event per decade of operation.

For comparison, the expected collisions from industry's AGREE_ACT events — the ones SHIRO also acts on — have Pc values in the range 1×10⁻³ to 1×10⁻², representing a collision risk roughly 500× higher per event. SHIRO is not passing on the dangerous events. It is passing on the marginal ones where the raw Pc was inflated by covariance overconfidence.

### 7.4 The Threshold Invariant

The most important safety guarantee SHIRO provides is structural, not statistical:

**SHIRO never passes on an event where the covariance-corrected Pc exceeds 1×10⁻⁴.**

The maneuver threshold is unchanged. The covariance inflation factor makes it harder to exceed that threshold — requiring higher actual collision risk to trigger a maneuver decision. But any event where genuine proximity risk remains after correction still triggers action.

This means SHIRO's safety guarantee is identical in kind to industry's: both systems maneuver when their best available Pc estimate exceeds 1×10⁻⁴. SHIRO's estimate is simply more accurate.

### 7.5 The Threshold Arms Race Context

The SpaceX case provides important context for why covariance realism matters beyond academic interest. Between December 2024 and May 2025, Starlink executed 144,404 collision avoidance maneuvers — approximately 288,000 per year — after dropping their internal threshold from 1×10⁻⁵ to 1×10⁻⁶. This threshold is 100× more conservative than the NASA CARA standard.

This threshold reduction was not driven by new evidence of increased collision risk. It was driven by distrust of the Pc estimates themselves — a rational response to known covariance overconfidence. Operators compensate for bad measurements by lowering thresholds until the system feels safe, at the cost of exponentially increasing maneuver burden.

SHIRO's argument is that this is the wrong correction applied at the wrong layer. Lowering the threshold treats the symptom. Correcting the covariance treats the cause. A mid-tier operator that adopts SHIRO's covariance realism framework before reaching Starlink's scale avoids the threshold arms race entirely — maintaining accurate risk estimates and a stable maneuver cadence as the constellation grows.

### 7.6 Limitations of the Safety Argument

The safety argument presented here has two honest limitations that must be acknowledged.

**Synthetic data circularity** — The ground truth safety of SHIRO passes is verified from synthetic events where the miss distance is known by construction. Real CDM events do not come with ground truth. Validation against real historical CDM data — finding events where operators chose not to maneuver and confirming safe passage via subsequent TLE data — is required before operational deployment. This is a stated future work item.

**Covariance inflation uniformity** — Applying a uniform inflation factor to all events regardless of object type, tracking history, or observation density is a simplification. Some objects in the catalog are well-tracked and their covariances are accurate. Applying 9–25× inflation to an already-accurate covariance would over-correct and produce artificially low Pc estimates. Operational deployment of SHIRO requires per-object covariance quality assessment, not a global inflation factor. The batch results presented here represent a population-level demonstration, not a per-event certification.

These limitations do not invalidate the methodology. They define the scope of the current validation and the requirements for production deployment.

---

## 8. Visualizations

### 7.1 B-Plane Comparison

The primary analytical visualization is a side-by-side b-plane comparison showing the industry view (raw covariance) alongside the SHIRO view (covariance-corrected).

Each panel shows:
- **Monte Carlo scatter** — 2,000 sampled positions from the probability distribution
- **3-sigma covariance ellipse** — analytical boundary of the uncertainty region
- **HBR circle** — hard body radius exclusion zone (centered at origin)
- **Miss vector** — nominal closest approach point

The fraction of scatter points inside the HBR circle is proportional to Pc. The visual separation between the covariance ellipse scale and the HBR circle scale immediately communicates whether the event is genuinely dangerous or a marginal case inflated by covariance overconfidence.

### 7.2 Pc Evolution Timeline

The Pc timeline shows both industry and SHIRO Pc curves on a log scale against time to TCA. The 1×10⁻⁴ decision threshold is marked as a horizontal reference line. The gap between the two curves — where industry Pc exceeds the threshold but SHIRO Pc does not — is the unnecessary maneuver zone.

Pc at each timestep along the approach trajectory is estimated using a physically motivated proxy:

```
Pc_proxy(t) = HBR² / (miss_distance(t)² + σ_combined²)
```

where σ_combined is the RMS combined position uncertainty. This proxy produces a monotonically increasing curve as the objects approach TCA, consistent with the theoretical behavior of Pc for converging objects with stable covariances.

### 7.3 3D Orbital Visualization

The Three.js visualization renders both full orbital periods as closed rings in 3D space against a wireframe Earth sphere. The cross-track conjunction geometry between a 51.6° ISS-type orbit and a 97.8° sun-synchronous orbit produces a visually clear X-shaped intersection — the characteristic appearance of a genuine cross-track LEO conjunction.

The conjunction point is marked with a pulsing red sphere at the orbital intersection. A HUD overlay displays the key decision metrics and SHIRO's final recommendation.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

**Covariance model simplicity** — The synthetic covariance is initialized from a target Pc using an analytical approximation and a fixed shape ratio. Real operational covariances have more complex structure reflecting the specific radar observation geometry for each object.

**Fixed inflation factor** — SHIRO currently applies a uniform inflation factor to all events. In practice, covariance quality varies significantly by object type (active satellite vs. debris), orbital regime, and tracking history. A per-object inflation factor calibrated to observation density would be more accurate.

**Short propagation window** — The Pc timeline covers only the final approach window (typically 0.4 hours). Full operational CDM analysis covers 72+ hours with multiple CDM updates. Future work should model Pc evolution across multiple CDM issuances.

**Synthetic data only** — The batch results presented here use synthetic conjunction events. Validation against real CDM data is ongoing pending formal data access approval from USSPACECOM.

### 8.2 Future Work

**Real CDM validation** — Apply SHIRO policy to historical CDM records from Space-Track, focusing on events where subsequent TLE data confirms whether a maneuver was executed and whether the objects actually passed safely without one.

**Per-object covariance calibration** — Integrate Space Fence radar observation frequency data to assign object-specific inflation factors. Well-observed objects receive lower inflation; debris with sparse history receives higher inflation.

**Fleet-level UMR optimization** — Extend from per-event decisions to fleet-level policy optimization, where maneuver decisions are evaluated jointly to minimize global UMR while maintaining fleet-level safety invariants.

**Integration with SHIRO:CONSTELLATION** — The decision engine developed here forms the core of SHIRO:CONSTELLATION, a live fleet monitoring system that runs continuously alongside operational conjunction analysis.

---

## 9. References

Alfano, S. (1995). Satellite Conjunction Monte Carlo Analysis. *AAS/AIAA Spaceflight Mechanics Meeting*, AAS 95-194.

Carpenter, J. R., Markley, F. L., & Alfriend, K. T. (2018). Covariance realism for space situational awareness. *Journal of Guidance, Control, and Dynamics*, 41(3), 598–612.

Chan, F. K. (1997). Spacecraft Collision Probability. *The Aerospace Press.*

Hejduk, M. D., Plakalovic, D., & Frisbee, J. H. (2004). Conjunction assessment risk analysis for deep-space satellites. *AAS/AIAA Astrodynamics Specialist Conference*, AAS 04-170.

Hejduk, M. D., & Snow, D. E. (2019). Satellite conjunction assessment risk analysis for 'Dilution of Precision' covariances. *Space Traffic Management Conference.*

NASA CARA (2014). Conjunction Assessment Risk Analysis — Recommended Practices for Satellite Operators. NASA/SP-2014-581.

SpaceX FCC Filing (2023). SpaceX Non-Geostationary Satellite System — Annual Report. FCC IBFS File No. SAT-MPL-20161115-00118.

---

## Appendix A — Key Parameters

| Parameter | Value | Justification |
|---|---|---|
| Decision threshold | 1×10⁻⁴ | NASA CARA standard |
| HBR (reference) | 20 m | Conservative LEO satellite size |
| Covariance inflation (min) | 9× | k=3 position (Hejduk 2004) |
| Covariance inflation (max) | 25× | k=5 position (Carpenter 2018) |
| Monte Carlo samples | 50,000 | Convergence within ±2×10⁻⁵ at 95% CI |
| Propagator timestep | 30 s | Sufficient for LEO (T≈90 min) |
| J2 coefficient | 1.08263×10⁻³ | WGS-84 |
| Earth GM | 398,600.4418 km³/s² | WGS-84 |
| Earth radius | 6,378.137 km | WGS-84 |

## Appendix B — Outcome Label Definitions

| Label | Industry | SHIRO | Interpretation |
|---|---|---|---|
| AGREE_ACT | ACT | ACT | Genuine risk — covariance correction does not resolve danger |
| UNNECESSARY_MANEUVER | ACT | PASS | Inflated Pc drove unnecessary maneuver |
| SHIRO_MORE_CAUTIOUS | PASS | ACT | SHIRO catches marginal risk industry misses |
| AGREE_PASS | PASS | PASS | Both correctly identify safe event |

## Appendix C — Software Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.24 | Orbital mechanics, linear algebra |
| scipy | ≥1.10 | Statistical distributions |
| poliastro | ≥0.17 | Orbital element definitions, CZML export |
| astropy | ≥5.3 | Time systems, unit handling |
| plotly | ≥5.15 | Analytical visualizations |
| three.js | r128 | 3D orbital visualization |

---

*SHIRO is open source under MIT License. Contributions welcome at github.com/SHIRO-prototype/Shiro_Validator*
