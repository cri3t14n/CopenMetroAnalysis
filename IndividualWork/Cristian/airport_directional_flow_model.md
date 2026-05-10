# Bayesian Directional Flow Model for Copenhagen Airport Metro Station

## 1. Project Idea

The goal of this project is to model passenger flow at **Københavns Lufthavn**, the Copenhagen Airport metro station, using a **Bayesian generative model**.

Instead of simply predicting `ins` and `outs` separately, we decompose the passenger flow into two interpretable components:

1. **Total demand**: how many passengers passed through the station in total.
2. **Directional split**: what proportion of that total flow corresponds to passengers going toward the airport versus leaving the airport toward the city.

This gives us a clearer interpretation of the airport station's behavior.

---

## 2. Dataset Interpretation

For the airport station, the variables have the following meaning:

| Variable | Meaning at Copenhagen Airport station |
|---|---|
| `ins` | Passengers entering the metro station from the airport, likely after arriving by plane and heading toward the city |
| `outs` | Passengers exiting the metro station at the airport, likely heading to catch a departing flight |
| `timestamp` | Time of the measurement |
| `station` | Station name |
| `stationId` | Unique station identifier |
| `system` | Metro line system, e.g. M1/M2 |

The variable `passengerNo` should not be used as an independent source of information if it is simply derived from `ins` and `outs`.

---

## 3. Main Research Question

> Can we probabilistically decompose passenger flow at Copenhagen Airport metro station into total demand and directional tendency, and explain how these vary by hour of day, weekday, and month/season?

In simpler terms:

> When is the airport metro station mainly serving passengers going to the airport, and when is it mainly serving passengers arriving from the airport and going to the city?

---

## 4. Why This Is Not Just Forecasting

A simple forecasting model would try to predict:

```text
ins_t
outs_t
```

separately.

However, this does not clearly separate two different phenomena:

1. Whether the station is generally busy.
2. Whether the flow is mainly toward the airport or away from the airport.

Our model separates these two questions.

For each time period `t`, we define:

```text
I_t = ins_t
O_t = outs_t
N_t = I_t + O_t
```

where:

```text
N_t = total passenger flow at time t
```

Then we model:

1. `N_t`: total demand.
2. `O_t / N_t`: directional split toward the airport.

---

## 5. Intuition with Examples

### Example 1: Morning departures

Suppose that at 06:00 we observe:

```text
ins = 100
outs = 900
total = 1000
```

Then:

```text
out_share = outs / total = 900 / 1000 = 0.90
```

This means that 90% of the passenger movements are `outs`.

Interpretation:

> The station is strongly departure-oriented at this time. Most passengers are arriving by metro and exiting at the airport to catch flights.

---

### Example 2: Afternoon arrivals

Suppose that at 17:00 we observe:

```text
ins = 800
outs = 500
total = 1300
```

Then:

```text
out_share = outs / total = 500 / 1300 ≈ 0.38
```

This means that only 38% of the passenger movements are `outs`, while the majority are `ins`.

Interpretation:

> The station is more arrival-oriented at this time. Many passengers are entering the metro station after arriving at the airport and are likely traveling toward the city.

---

## 6. Generative Model

The model has two main components.

---

## 6.1 Total Demand Model

First, we model the total number of passenger movements:

```text
N_t = ins_t + outs_t
```

Because this is count data, we can use a Negative Binomial distribution:

```text
N_t ~ NegativeBinomial(mean = λ_t, dispersion = φ)
```

The expected total demand depends on temporal features:

```text
log(λ_t) = μ
         + hour_effect[hour_t]
         + weekday_effect[weekday_t]
         + month_effect[month_t]
```

where:

| Term | Meaning |
|---|---|
| `μ` | Global baseline demand |
| `hour_effect` | Effect of hour of day |
| `weekday_effect` | Effect of day of week |
| `month_effect` | Effect of month or season |
| `φ` | Dispersion parameter |

This part answers:

> How busy is the station expected to be at a given hour, weekday, and month?

---

## 6.2 Directional Split Model

After generating the total flow `N_t`, the model decides how many of these passengers are `outs`.

We model:

```text
O_t ~ Binomial(N_t, p_t)
```

where:

```text
p_t = probability that a passenger movement is an outflow
```

The probability `p_t` is modeled using a logistic function:

```text
logit(p_t) = η
           + hour_direction_effect[hour_t]
           + weekday_direction_effect[weekday_t]
           + month_direction_effect[month_t]
```

where:

| Term | Meaning |
|---|---|
| `η` | Global baseline direction tendency |
| `hour_direction_effect` | How directionality changes by hour |
| `weekday_direction_effect` | How directionality changes by weekday |
| `month_direction_effect` | How directionality changes by month or season |

Interpretation of `p_t`:

| Value of `p_t` | Interpretation |
|---|---|
| `p_t > 0.5` | More passengers are exiting at the airport: departure-oriented flow |
| `p_t < 0.5` | More passengers are entering the metro from the airport: arrival-oriented flow |
| `p_t ≈ 0.5` | Balanced flow |

---

## 7. Full Generative Story

The full generative story is:

```text
For each hour h:
    hour_effect[h] ~ Normal(0, σ_hour)
    hour_direction_effect[h] ~ Normal(0, τ_hour)

For each weekday d:
    weekday_effect[d] ~ Normal(0, σ_weekday)
    weekday_direction_effect[d] ~ Normal(0, τ_weekday)

For each month m:
    month_effect[m] ~ Normal(0, σ_month)
    month_direction_effect[m] ~ Normal(0, τ_month)

For each time step t:
    log(λ_t) = μ + hour_effect[hour_t] + weekday_effect[weekday_t] + month_effect[month_t]
    N_t ~ NegativeBinomial(mean = λ_t, dispersion = φ)

    logit(p_t) = η + hour_direction_effect[hour_t] + weekday_direction_effect[weekday_t] + month_direction_effect[month_t]
    O_t ~ Binomial(N_t, p_t)

    I_t = N_t - O_t
```

---

## 8. Probabilistic Graphical Model Structure

A simplified PGM can be described as:

```text
hour_t ───────────────► λ_t ─────► total_t
weekday_t ────────────► λ_t
month_t ──────────────► λ_t

hour_t ───────────────► p_t ─────► outs_t
weekday_t ────────────► p_t
month_t ──────────────► p_t

total_t ────────────────────────► outs_t
```

Observed variables:

```text
total_t, outs_t, hour_t, weekday_t, month_t
```

Unknown parameters:

```text
μ, η, temporal effects, direction effects, dispersion parameters
```

---

## 9. Why This Is a Good MBML Project

This is a good Model-Based Machine Learning project because it includes:

- A clear real-world problem.
- A generative story.
- A probabilistic graphical model.
- Bayesian priors over parameters.
- Count-data likelihoods.
- Interpretable posterior distributions.
- Uncertainty quantification.
- A natural Pyro implementation.

It is also different from the other group members' topics because it is not mainly about:

- hidden state discovery,
- clustering,
- anomaly detection.

Instead, it focuses on an interpretable Bayesian decomposition of airport passenger flow.

---

## 10. Expected Outputs and Insights

The model can answer questions such as:

### Total demand insights

- Which hours of the day have the highest total passenger demand?
- Which weekdays are busiest?
- Which months or seasons have the strongest passenger flow?

### Directional insights

- Which hours are departure-oriented?
- Which hours are arrival-oriented?
- Are Fridays more departure-oriented than other weekdays?
- Does summer have a different directional pattern than winter?

### Uncertainty insights

Because the model is Bayesian, we do not only get point estimates. We also get uncertainty intervals.

For example, instead of saying:

```text
06:00 is departure-oriented.
```

we can say:

```text
At 06:00, the posterior mean of p_t is high, with a credible interval showing strong evidence that the flow is departure-oriented.
```

---

## 11. Suggested Baselines

To make the project stronger, compare the main model against simpler baselines.

### Baseline 1: Independent models for ins and outs

```text
ins_t  ~ NegativeBinomial(λ_in_t, φ_in)
outs_t ~ NegativeBinomial(λ_out_t, φ_out)
```

This baseline treats `ins` and `outs` as separate count variables.

---

### Baseline 2: Total-demand-only model

```text
total_t ~ NegativeBinomial(λ_t, φ)
```

This baseline models how busy the station is but ignores directionality.

---

### Main model: Total demand + directional split

```text
total_t ~ NegativeBinomial(λ_t, φ)
outs_t | total_t ~ Binomial(total_t, p_t)
```

This model explains both total traffic and directional behavior.

---

## 12. Evaluation Ideas

Possible evaluation methods:

1. **Train/test split**  
   Train on earlier months and test on later months.

2. **Predictive log likelihood**  
   Compare how well the model explains held-out data.

3. **MAE or RMSE for total flow**  
   Evaluate total demand prediction.

4. **Direction prediction accuracy**  
   Check whether the model correctly predicts whether `outs > ins`.

5. **Posterior predictive checks**  
   Simulate data from the fitted model and compare it with real observed data.

---

## 13. One-Sentence Summary

The project models airport metro passenger flow as a two-step generative process: first generating the total number of passengers passing through the station, and then probabilistically splitting that total into passengers going toward the airport versus passengers leaving the airport toward the city.
