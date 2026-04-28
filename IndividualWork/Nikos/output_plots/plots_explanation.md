# Airport Flow Analysis — Plots Explanation

**Station:** Københavns Lufthavn (Copenhagen Airport)  
**Station ID:** 8603328 · **System:** M1/M2  
**Data range:** 2024-04-01 → 2026-03-31  
**Observations after cleaning:** 913,540  
**Zero-activity rows dropped:** 137,660

---

## Summary of Key Findings

| Metric | Value |
|---|---|
| Total passengers in | 9,209,543 |
| Total passengers out | 10,611,841 |
| Mean ins per observation | 10.08 |
| Mean outs per observation | 11.62 |
| Peak hour for arrivals | 17:00 |
| Peak hour for departures | 16:00 |
| Busiest weekday | Friday |
| Busiest month | August |

> Outs consistently exceed ins — the airport metro is used more for departures (people leaving the airport toward the city) than arrivals. This is counterintuitive at first but makes sense: one arriving flight produces many passengers over a short burst, while departing passengers trickle in over hours.

---

## Plot-by-Plot Explanation

---

### 01 — Raw Time Series (`01_raw_timeseries.png`)

**What it shows:** Every single observation in the dataset plotted as a line over time. Three lines: `ins` (blue), `outs` (orange), `passengerNo` (grey dashed).

**What to look for:**
- The overall density and noisiness of the raw signal — there are ~900k rows so individual measurements overlap heavily.
- Visible seasonal waves (summer peaks, winter dips).
- Any flat stretches where the sensor stopped recording (gaps in the lines).
- The grey `passengerNo` line acts as a smoothed reference since it is a rolling average of ins and outs.

**Outcome:** Gives you a "first look" at the raw data before any aggregation. Confirms the data is continuous with no major gaps and shows that outs slightly dominate ins across the full period.

---

### 02 — Daily Aggregated Flow (`02_daily_flow.png`)

**What it shows:** All observations are summed per calendar day. The area between the two lines is shaded blue when ins > outs (net arrivals) and orange when outs > ins (net departures).

**What to look for:**
- Weekly periodicity — dips on weekdays vs peaks on weekends or vice versa.
- Summer 2024 and Summer 2025 peaks (airport travel is seasonal).
- The colored shading tells you at a glance which direction the station "flows" on each day.

**Outcome:** Much cleaner than the raw plot. Clearly shows that outs dominate most days (orange shading), with occasional blue spikes. Summer months show visibly higher totals than winter.

---

### 03 — Average Hourly Profile (`03_hourly_profile.png`)

**What it shows:** The average number of passengers entering and leaving at each hour of the day (0–23), split into two subplots: weekday (top) and weekend (bottom).

**What to look for:**
- On weekdays: commuter double-peak (morning rush + evening rush) or a flat airport-specific profile.
- On weekends: smoother, more spread-out traffic throughout the day.
- The crossover between ins and outs at specific hours tells you when the station switches from being "departure-heavy" to "arrival-heavy".

**Outcome:** Peak hours for ins are around 17:00 and for outs around 16:00, matching the afternoon travel surge. Weekends show a flatter, more evenly distributed profile compared to weekdays — consistent with leisure/tourist travel rather than commuter patterns.

---

### 04 — Day-of-Week Pattern (`04_weekday_pattern.png`)

**What it shows:** Side-by-side bar chart with the mean ins (blue) and outs (orange) for each day of the week, Monday to Sunday. Error bars show the standard deviation — how much the values vary day to day.

**What to look for:**
- Which day of the week is busiest overall.
- Whether weekends are higher or lower than weekdays (for an airport station, weekends and Fridays are often higher).
- Large error bars indicate high variability on that day (e.g. a day affected by both quiet weeks and peak holiday periods).

**Outcome:** Friday is the busiest day for ins. The error bars are large across all days, reflecting the high variability driven by seasonality and special events. Sunday shows elevated outs, consistent with people returning from weekend trips.

---

### 05 — Monthly Seasonality (`05_monthly_seasonality.png`)

**What it shows:** The mean ins and outs grouped by month (January = 1, December = 12), with a shaded band showing ± one standard deviation.

**What to look for:**
- A clear summer peak (June–August) driven by tourist and holiday travel.
- A winter dip, especially around December–January.
- Whether ins and outs track each other closely or diverge in certain months.

**Outcome:** August is the busiest month for ins. The shaded band is wide in summer months, meaning some days are extremely busy while others are quieter — reflecting the mix of peak travel days and off-peak days within the same month.

---

### 06 — Heatmap: Hour × Weekday (`06_heatmap_hour_weekday.png`)

**What it shows:** A grid where each cell is the mean `passengerNo` for a specific hour of the day (rows, 0–23) and day of the week (columns, Mon–Sun). Darker red = more passengers.

**What to look for:**
- The hottest cells reveal the exact hour+day combinations with the highest traffic.
- Any "cold" rows (low-traffic hours, typically 02:00–05:00 overnight).
- Whether the pattern differs significantly between weekdays and weekends.

**Outcome:** The afternoon block (15:00–19:00) across Friday–Sunday is the hottest region. Overnight hours (01:00–05:00) are consistently dark (low activity), as expected. The heatmap makes the combined hour+weekday interaction immediately visible — something no individual bar chart can show.

---

### 07 — Net Flow Over Time (`07_net_flow.png`)

**What it shows:** The daily sum of `net_flow = ins - outs`. Green fill = more arrivals than departures that day. Red fill = more departures than arrivals. A dashed line marks zero.

**What to look for:**
- Whether the station is structurally a "departure" station (mostly red) or "arrival" station (mostly green).
- Seasonal shifts — does the balance change in summer?
- Specific days with extreme net flows (could match known events or disruptions).

**Outcome:** The plot is predominantly red — the airport metro handles more departures than arrivals on most days. This aligns with the overall totals (10.6M outs vs 9.2M ins). Green spikes correspond to days with unusual inbound traffic surges.

---

### 08 — Distribution of Ins & Outs (`08_distributions.png`)

**What it shows:** Histograms with a KDE (smooth density curve) overlaid for `ins` (left) and `outs` (right), using only non-zero observations. Vertical lines mark the mean (black dashed) and median (grey dotted).

**What to look for:**
- Whether the distribution is symmetric or right-skewed (a long tail to the right means rare very-high-traffic moments).
- The gap between mean and median: if mean > median, the distribution is right-skewed, pulled up by occasional bursts.
- Whether ins and outs have similar shapes or one is more spread out.

**Outcome:** Both distributions are strongly right-skewed — most observations are low-count (sensor readings during quiet periods) but a small number of readings are very high (burst arrivals from flights). The mean exceeds the median in both cases, confirming the skew.

---

### 09 — 7-Day Rolling Average (`09_rolling_average.png`)

**What it shows:** Daily totals plotted faintly in the background, with the 7-day centered rolling mean drawn as a bold line on top. This smooths out the day-to-day noise to reveal the underlying trend.

**What to look for:**
- Long-term growth or decline in passenger numbers.
- Clear seasonal waves (summer rise, winter fall).
- Disruptions where the rolling average drops sharply and recovers.

**Outcome:** The rolling average clearly shows a summer 2024 peak, a winter 2024/2025 dip, and a recovery into summer 2025. The raw daily values (faint) fluctuate widely but the trend line is smooth and interpretable.

---

### 10 — Ins vs Outs Scatter (`10_ins_vs_outs_scatter.png`)

**What it shows:** Each point is one observation. X = ins, Y = outs, color = hour of day (purple/yellow from the plasma colormap). The dashed diagonal line is `y = x` (equal ins and outs).

**What to look for:**
- Points above the diagonal: outs > ins (net departure).
- Points below the diagonal: ins > outs (net arrival).
- Whether color clusters appear — do certain hours tend to be departure-heavy vs arrival-heavy?
- Outlier points far from the main cloud.

**Outcome:** The cloud sits slightly above the diagonal (outs > ins on average). Afternoon hours (yellow/bright colors, 14:00–18:00) cluster toward higher values on both axes. The scatter reveals a strong linear correlation between ins and outs — when the station is busy in one direction, it tends to be busy in the other too.

---

### 11 — Year-over-Year Comparison (`11_year_over_year.png`)

**What it shows:** Monthly totals of ins and outs, with one colored line per year. Allows direct comparison of the same month across different years.

**What to look for:**
- Is 2025 higher or lower than 2024 in the same months?
- Do the seasonal shapes (summer peak, winter dip) look similar across years?
- Any months where one year strongly diverges (could indicate an event, strike, or service change).

**Outcome:** Both 2024 and 2025 follow the same seasonal shape. 2025 summer months are slightly higher than 2024, suggesting modest passenger growth year-on-year. Note: 2024 data starts in April so early months (Jan–Mar) only have 2025 data.

---

### 12 — Ins/Outs Ratio Over Time (`12_ratio_over_time.png`)

**What it shows:** The daily average of `ratio = ins / (outs + ε)`. Values above 1 (green) mean more arrivals; below 1 (red) mean more departures. A dashed line marks ratio = 1.

**What to look for:**
- Structural bias — is the ratio almost always below 1 (departure-heavy station)?
- Seasonal patterns in the ratio itself (does summer shift toward more arrivals?).
- Specific spikes above 1 that don't show up as obvious in other plots.

**Outcome:** The ratio fluctuates around 0.85–0.95, confirming the station is consistently departure-heavy. Occasional green spikes above 1 correspond to days with heavy inbound passenger surges (e.g. major flight arrivals, events nearby).

---

### 13 — Outlier Detection (`13_outliers.png`)

**What it shows:** The daily ins and outs as lines, with red scatter points marking days whose z-score exceeds ±2.5 in either metric. Z-score measures how many standard deviations a day is from the mean.

**Top 10 anomalous dates found:**

| Date | ins | outs | Likely cause |
|---|---|---|---|
| 2025-06-20 | 27,531 | 29,630 | Extreme traffic spike — z > 5.7 |
| 2025-10-29 | 3,648 | 3,903 | Very low traffic — possible service disruption |
| 2024-12-25 | 5,030 | 5,939 | Christmas Day — reduced travel |
| 2025-08-29 | 21,004 | 17,671 | High-traffic day, possibly end of summer peak |
| 2024-12-24 | 4,786 | 6,675 | Christmas Eve — reduced travel |
| 2025-12-25 | 5,699 | 6,961 | Christmas Day 2025 |
| 2025-12-05 | 19,966 | 19,533 | Unexplained spike |
| 2024-09-13 | 19,926 | 18,152 | Unexplained spike |
| 2024-10-11 | 16,663 | 21,838 | Asymmetric — heavy outs |
| 2025-12-24 | 5,537 | 8,303 | Christmas Eve 2025 |

**Outcome:** Two clear patterns emerge — anomalously high days (events near the airport or peak travel bursts) and anomalously low days (public holidays, especially Christmas). The 2025-06-20 spike is the most extreme in the entire dataset, nearly 6 standard deviations above normal.

---

### 14 — Cumulative Flow (`14_cumulative_flow.png`)

**What it shows:** The running total of ins and outs accumulated day by day from the start of the dataset. Both lines grow monotonically (they can never decrease).

**What to look for:**
- The slope of each line — steeper = more passengers per day on average.
- The gap between the two lines widening over time (outs persistently outpacing ins).
- Any visible kinks (slope changes) corresponding to disruptions or seasonal slowdowns.

**Outcome:** By end of March 2026, cumulative outs (~10.6M) clearly exceed cumulative ins (~9.2M). The gap between the lines grows slowly but steadily, confirming the structural departure bias. Slope is slightly steeper in summer periods, visible as subtle accelerations in both lines.

---

## How These Plots Connect

```
Raw signal          →  01 (see everything), 02 (daily), 09 (smoothed trend)
Temporal patterns   →  03 (hour), 04 (weekday), 05 (month), 06 (hour×weekday heatmap)
Flow balance        →  07 (net), 12 (ratio)
Statistical profile →  08 (distributions), 10 (scatter)
Multi-year view     →  11 (year-over-year), 14 (cumulative)
Data quality        →  13 (outliers)
```

Together they answer the three core questions for model-based ML work:

1. **What does the signal look like?** → Plots 01, 02, 09
2. **When is the station busy?** → Plots 03, 04, 05, 06
3. **Is the data clean enough to model?** → Plots 08, 13
