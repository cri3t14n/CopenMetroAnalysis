# Models Overview — Copenhagen Airport Metro Station

**Station:** Københavns Lufthavn (station ID 8603328, line M1/M2)  
**Data period:** 2024-04-01 → 2026-03-31  
**Author:** Cristian (`IndividualWork/Cristian/`)

---

## Problem Framing

Instead of independently predicting `ins` and `outs`, the work decomposes passenger flow into two separate questions:

1. **How busy is the station?** → total demand `N = ins + outs`
2. **Which direction dominates?** → directional split `p = outs / N`

`outs` at the airport station = passengers heading *to* the airport (departures).  
`ins` at the airport station = passengers *from* the airport heading to the city (arrivals).

---

## Data Preparation (`data_preparation.ipynb`)

Raw data is 1-minute interval sensor readings (~1M rows). The pipeline:

1. Load station 8603328 data via `MetroDataQuery`.
2. Clip negative values, remove zero-flow rows (13% of data).
3. **Resample to 15-minute bins** by summing `ins` and `outs` → ~69K rows.
4. Compute features:
   - `total = ins + outs`
   - `outs_share = outs / total` (directional split, between 0 and 1)
   - `slot` — 15-minute slot of day (0–95, one per 15 min across 24h)
   - `weekday` — 0 (Monday) to 6 (Sunday)
   - `month` — 0–11
   - `is_weekend`
5. Save to `airport_15min_model_data.parquet`.

**Train/test split:** 80/20 by time → ~55K train rows (up to Nov 2025), ~13K test rows.

---

## Baselines (`baselines.ipynb`)

Three simple non-Bayesian baselines, used as reference for the Pyro models:

| Baseline | How it works |
|---|---|
| **Global** | Predict single global average for `total` and `outs_share` |
| **Slot** | Predict per-slot (15-min of day) average — captures the daily cycle |
| **Slot + Weekday** | Predict per (slot, weekday) average — captures both daily and weekly patterns |

Each baseline predicts `total` and `outs_share` separately, then derives `outs = total × outs_share` and `ins = total − outs`.

### Baseline Results (on test set)

| Model | total MAE | total RMSE | outs MAE | direction accuracy |
|---|---|---|---|---|
| Global | 131.5 | 160.1 | 81.8 | 59.1% |
| Slot | 75.8 | 97.7 | 41.8 | 80.8% |
| **Slot + Weekday** | **64.1** | **83.3** | **35.8** | **80.8%** |

---

## Bayesian Models (`pyro_model.ipynb`)

Both models are implemented in [Pyro](https://pyro.ai/) using **Stochastic Variational Inference (SVI)**.

### Generative Structure (shared by both models)

The model has two stacked components:

**Step 1 — Total demand:**

```
log(λ_t) = μ + temporal_effects
total_t  ~ NegativeBinomial(mean = λ_t, dispersion = φ)
```

`NegativeBinomial` is chosen because passenger count data is overdispersed — variance is higher than a Poisson would predict. `φ` controls that extra variance.

**Step 2 — Directional split:**

```
logit(p_t) = η + temporal_effects
outs_t     ~ Binomial(total_t, p_t)
```

`Binomial` is the natural choice because `outs` is a count bounded by `total`. `p_t > 0.5` means departure-dominant flow; `p_t < 0.5` means arrival-dominant.

Both components share the same temporal covariates but have **independent effect parameters** — being busy does not imply any particular direction, and vice versa.

### Non-Centered Parameterization

All random effects use the NCP trick to improve SVI convergence:

```python
# Instead of:  effect ~ Normal(0, sigma)
# We do:
raw   ~ Normal(0, 1)        # sampled in standard space
effect = sigma * raw         # scaled outside the sample statement
```

This keeps gradient flow well-conditioned during optimization.

---

### Model 1: Slot-only

**File:** `pyro_model.ipynb` (function `airport_model`)

**Priors:**

| Parameter | Prior | Meaning |
|---|---|---|
| `μ` | Normal(5.5, 1.0) | Global log-demand baseline (~exp(5.5) ≈ 245 passengers) |
| `η` | Normal(0.0, 1.0) | Global logit-direction baseline |
| `sigma_slot_total` | HalfNormal(1.0) | Scale of slot effects on total demand |
| `sigma_slot_dir` | HalfNormal(1.0) | Scale of slot effects on direction |
| `phi` | HalfNormal(10.0) | NegativeBinomial dispersion |
| `slot_total_raw[96]` | Normal(0, 1) | Per-slot raw effects for total (NCP) |
| `slot_dir_raw[96]` | Normal(0, 1) | Per-slot raw effects for direction (NCP) |

**Forward pass:**

```python
log_lambda  = mu + sigma_slot_total * slot_total_raw[slot]
logits_p    = eta + sigma_slot_dir  * slot_dir_raw[slot]

total ~ NegativeBinomial(phi, logits = log(lambda) - log(phi))
outs  ~ Binomial(total, logits = logits_p)
```

**Training:** SVI with `AutoNormal` guide (mean-field Gaussian over all parameters), `ClippedAdam` optimizer (lr=0.01, clip_norm=10), 3000 steps. Converges at ELBO ≈ 12.78 per observation.

---

### Model 2: Slot + Weekday

**File:** `pyro_model.ipynb` (function `airport_model_slot_weekday`)

Extends Model 1 by adding weekday random effects for both total demand and direction:

```
log(λ_t) = μ + slot_total_effect[slot_t] + weekday_total_effect[weekday_t]
logit(p_t) = η + slot_dir_effect[slot_t]  + weekday_dir_effect[weekday_t]
```

Additional priors:

| Parameter | Prior | Meaning |
|---|---|---|
| `sigma_weekday_total` | HalfNormal(0.5) | Scale of weekday effects on total demand |
| `sigma_weekday_dir` | HalfNormal(0.5) | Scale of weekday effects on direction |
| `weekday_total_raw[7]` | Normal(0, 1) | Per-weekday raw effects for total (NCP) |
| `weekday_dir_raw[7]` | Normal(0, 1) | Per-weekday raw effects for direction (NCP) |

Narrower prior (HalfNormal(0.5) vs 1.0) reflects the prior belief that weekday variation is smaller than within-day variation.

**Training:** Same SVI setup, converges at ELBO ≈ 12.68 per observation (slightly better than Model 1).

---

## Results Comparison

| Model | total MAE | total RMSE | outs MAE | outs RMSE | direction accuracy |
|---|---|---|---|---|---|
| Global baseline | 131.5 | 160.1 | 81.8 | 96.9 | 59.1% |
| Slot baseline | 75.8 | 97.7 | 41.8 | 55.6 | 80.8% |
| Slot+Weekday baseline | 64.1 | 83.3 | 35.8 | 48.1 | 80.8% |
| **Pyro: slot-only** | 77.7 | 100.1 | 42.3 | 56.4 | 80.8% |
| **Pyro: slot+weekday** | 74.9 | 96.7 | 41.2 | 55.0 | 80.9% |

Point-estimate accuracy of the Pyro models is competitive with the equivalent baselines, but does not clearly surpass them.

### Key advantage: Uncertainty Quantification

The Bayesian models produce full posterior predictive distributions via 500 posterior samples. The 90% predictive intervals show well-calibrated coverage:

| Target | 90% interval empirical coverage |
|---|---|
| Total passengers | 89% (Model 1), 88.6% (Model 2) |
| Outs count | 90.7% (both models) |
| Outs share | 44% (both models) — undercoverage |

The low coverage on `outs_share` indicates the model does not capture all sources of variability in the directional split. This is a known limitation and a direction for future work (e.g., adding month effects or interaction terms).

---

## Summary

| Aspect | Detail |
|---|---|
| Data granularity | 15-minute bins |
| Train period | Apr 2024 – Nov 2025 (~55K observations) |
| Test period | Nov 2025 – Mar 2026 (~13K observations) |
| Inference method | SVI (Variational Bayes) with AutoNormal guide |
| Optimizer | ClippedAdam, lr=0.01 |
| Training steps | 3000 |
| Posterior samples at test time | 500 |
| Framework | Pyro (PyTorch backend) |
