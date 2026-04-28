from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "DatabaseScripts"))

from DataFilter import MetroDataQuery

PARQUET_PATH = ROOT / "metro_passengers.parquet"
STATION_ID   = 8603328
OUT_DIR      = Path(__file__).parent / "output_plots"
OUT_DIR.mkdir(exist_ok=True)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

BLUE   = "#2878B5"
ORANGE = "#FF7F0E"
GREY   = "#AAAAAA"
DPI    = 150

# ── load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
query = (
    MetroDataQuery(PARQUET_PATH)
    .filter_station_id(STATION_ID)
    .filter_system("m1m2")
)
df = query.to_pandas(sort_by_time=True)

# ── preprocess ─────────────────────────────────────────────────────────────────
for col in ["ins", "outs", "passengerNo"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["date"]       = df["timestamp"].dt.date
df["hour"]       = df["timestamp"].dt.hour
df["weekday"]    = df["timestamp"].dt.day_name()
df["month"]      = df["timestamp"].dt.month
df["year"]       = df["timestamp"].dt.year
df["week"]       = df["timestamp"].dt.isocalendar().week.astype(int)
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
df["net_flow"]   = df["ins"] - df["outs"]

zero_rows = int(((df["ins"] == 0) & (df["outs"] == 0)).sum())
df = df[(df["ins"] > 0) | (df["outs"] > 0)].copy()

print(f"Shape after cleaning : {df.shape}")
print(f"Date range           : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

# ── helpers ────────────────────────────────────────────────────────────────────
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved → {path.name}")

# ── plot 1: raw time series ────────────────────────────────────────────────────
print("\nPlot 01 – raw time series")
try:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"], df["ins"],         color=BLUE,   alpha=0.7, lw=0.8, label="ins")
    ax.plot(df["timestamp"], df["outs"],        color=ORANGE, alpha=0.7, lw=0.8, label="outs")
    ax.plot(df["timestamp"], df["passengerNo"], color=GREY,   alpha=0.6, lw=0.8, ls="--", label="passengerNo")
    ax.set_title("Raw Passenger Flow – Copenhagen Airport Metro Station")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Passengers")
    ax.legend()
    save(fig, "01_raw_timeseries.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 2: daily aggregated flow ──────────────────────────────────────────────
print("Plot 02 – daily flow")
try:
    daily = df.set_index("timestamp").resample("D")[["ins", "outs"]].sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily.index, daily["ins"],  color=BLUE,   lw=1.2, label="ins")
    ax.plot(daily.index, daily["outs"], color=ORANGE, lw=1.2, label="outs")
    ax.fill_between(daily.index, daily["ins"], daily["outs"],
                    where=daily["ins"] >= daily["outs"], alpha=0.2, color=BLUE,   label="net arrivals")
    ax.fill_between(daily.index, daily["ins"], daily["outs"],
                    where=daily["ins"] <  daily["outs"], alpha=0.2, color=ORANGE, label="net departures")
    ax.set_title("Daily Total Ins & Outs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Passengers")
    ax.legend()
    save(fig, "02_daily_flow.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 3: hourly profile weekday vs weekend ──────────────────────────────────
print("Plot 03 – hourly profile")
try:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=True)
    for ax, is_wknd, label in zip(axes, [False, True], ["Weekday", "Weekend"]):
        sub = df[df["is_weekend"] == is_wknd].groupby("hour")[["ins", "outs"]].mean()
        ax.plot(sub.index, sub["ins"],  marker="o", color=BLUE,   label="ins")
        ax.plot(sub.index, sub["outs"], marker="o", color=ORANGE, label="outs")
        ax.set_title(f"{label}")
        ax.set_ylabel("Mean Passengers")
        ax.legend()
        ax.set_xticks(range(24))
    axes[1].set_xlabel("Hour of Day")
    fig.suptitle("Average Hourly Passenger Profile – Weekday vs Weekend", fontsize=13)
    save(fig, "03_hourly_profile.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 4: day-of-week pattern ────────────────────────────────────────────────
print("Plot 04 – weekday pattern")
try:
    wkd = df.groupby("weekday")[["ins", "outs"]].agg(["mean", "std"]).reindex(WEEKDAY_ORDER)
    x   = np.arange(len(WEEKDAY_ORDER))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, wkd["ins"]["mean"],  w, yerr=wkd["ins"]["std"],  color=BLUE,   alpha=0.8, label="ins",  capsize=4)
    ax.bar(x + w/2, wkd["outs"]["mean"], w, yerr=wkd["outs"]["std"], color=ORANGE, alpha=0.8, label="outs", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(WEEKDAY_ORDER, rotation=15)
    ax.set_title("Mean Passenger Flow by Day of Week")
    ax.set_ylabel("Mean Passengers")
    ax.legend()
    save(fig, "04_weekday_pattern.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 5: monthly seasonality ────────────────────────────────────────────────
print("Plot 05 – monthly seasonality")
try:
    mon = df.groupby("month")[["ins", "outs"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, color in [("ins", BLUE), ("outs", ORANGE)]:
        m   = mon[col]["mean"]
        s   = mon[col]["std"]
        ax.plot(m.index, m.values, marker="o", color=color, label=col)
        ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.15)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Monthly Seasonality of Passenger Flow")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Passengers")
    ax.legend()
    save(fig, "05_monthly_seasonality.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 6: heatmap hour x weekday ────────────────────────────────────────────
print("Plot 06 – heatmap hour × weekday")
try:
    pivot = (
        df.groupby(["hour", "weekday"])["passengerNo"]
        .mean()
        .unstack("weekday")
        .reindex(columns=WEEKDAY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False, linewidths=0.3, cbar_kws={"label": "Mean passengerNo"})
    ax.set_title("Passenger Intensity Heatmap (Hour × Weekday)")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Hour of Day")
    save(fig, "06_heatmap_hour_weekday.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 7: net flow ───────────────────────────────────────────────────────────
print("Plot 07 – net flow")
try:
    net_daily = df.set_index("timestamp").resample("D")["net_flow"].sum()
    fig, ax   = plt.subplots(figsize=(14, 6))
    ax.fill_between(net_daily.index, net_daily, 0,
                    where=net_daily >= 0, color="green", alpha=0.4, label="net arrivals")
    ax.fill_between(net_daily.index, net_daily, 0,
                    where=net_daily <  0, color="red",   alpha=0.4, label="net departures")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Net Flow (Arrivals – Departures) – Daily")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Flow")
    ax.legend()
    save(fig, "07_net_flow.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 8: distributions ──────────────────────────────────────────────────────
print("Plot 08 – distributions")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, color in zip(axes, ["ins", "outs"], [BLUE, ORANGE]):
        data = df[df[col] > 0][col]
        sns.histplot(data, kde=True, ax=ax, color=color, alpha=0.6)
        ax.axvline(data.mean(),   color="black", ls="--", lw=1.2, label=f"mean={data.mean():.1f}")
        ax.axvline(data.median(), color="grey",  ls=":",  lw=1.2, label=f"median={data.median():.1f}")
        ax.set_title(f"Distribution of {col} (non-zero)")
        ax.set_xlabel(col)
        ax.legend()
    fig.suptitle("Distribution of Passenger Counts (non-zero observations)", fontsize=13)
    save(fig, "08_distributions.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 9: rolling 7-day average ─────────────────────────────────────────────
print("Plot 09 – rolling average")
try:
    daily_raw = df.set_index("timestamp").resample("D")[["ins", "outs"]].sum()
    roll      = daily_raw.rolling(7, center=True, min_periods=1).mean()
    fig, ax   = plt.subplots(figsize=(14, 6))
    for col, color in [("ins", BLUE), ("outs", ORANGE)]:
        ax.plot(daily_raw.index, daily_raw[col], color=color, alpha=0.25, lw=0.8)
        ax.plot(roll.index,      roll[col],      color=color, alpha=1.0,  lw=1.8, label=f"{col} (7d avg)")
    ax.set_title("7-Day Rolling Average of Daily Passenger Flow")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()
    save(fig, "09_rolling_average.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 10: ins vs outs scatter ───────────────────────────────────────────────
print("Plot 10 – ins vs outs scatter")
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["ins"], df["outs"], c=df["hour"], cmap="plasma", alpha=0.3, s=8)
    lim = max(df["ins"].max(), df["outs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.colorbar(sc, ax=ax, label="Hour of Day")
    ax.set_title("Ins vs Outs per Observation – Coloured by Hour of Day")
    ax.set_xlabel("ins")
    ax.set_ylabel("outs")
    ax.legend()
    save(fig, "10_ins_vs_outs_scatter.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 11: year-over-year ────────────────────────────────────────────────────
print("Plot 11 – year-over-year")
try:
    years = sorted(df["year"].unique())
    if len(years) < 2:
        print("  WARNING: fewer than 2 years in data, skipping YoY plot")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        palette   = plt.cm.tab10.colors
        for ax, col in zip(axes, ["ins", "outs"]):
            for i, yr in enumerate(years):
                sub = df[df["year"] == yr].groupby("month")[col].sum()
                ax.plot(sub.index, sub.values, marker="o",
                        color=palette[i % len(palette)], label=str(yr))
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(MONTH_NAMES, rotation=30)
            ax.set_title(f"Monthly {col}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Total")
            ax.legend(title="Year")
        fig.suptitle("Year-over-Year Monthly Passenger Flow", fontsize=13)
        save(fig, "11_year_over_year.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 12: ins/outs ratio ────────────────────────────────────────────────────
print("Plot 12 – ins/outs ratio")
try:
    df["ratio"] = df["ins"] / (df["outs"] + 1e-6)
    ratio_daily = df.set_index("timestamp").resample("D")["ratio"].mean()
    fig, ax     = plt.subplots(figsize=(14, 6))
    ax.plot(ratio_daily.index, ratio_daily, color=BLUE, lw=1.0)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ratio = 1")
    ax.fill_between(ratio_daily.index, ratio_daily, 1,
                    where=ratio_daily >= 1, color="green", alpha=0.2, label=">1 net arrivals")
    ax.fill_between(ratio_daily.index, ratio_daily, 1,
                    where=ratio_daily <  1, color="red",   alpha=0.2, label="<1 net departures")
    ax.set_title("Daily Ins/Outs Ratio (>1 = net arrivals, <1 = net departures)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ratio")
    ax.legend()
    save(fig, "12_ratio_over_time.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 13: outliers ──────────────────────────────────────────────────────────
print("Plot 13 – outliers")
try:
    daily_out = df.set_index("timestamp").resample("D")[["ins", "outs"]].sum()
    for col in ["ins", "outs"]:
        daily_out[f"z_{col}"] = stats.zscore(daily_out[col].fillna(0))
    daily_out["is_outlier"] = (daily_out["z_ins"].abs() > 2.5) | (daily_out["z_outs"].abs() > 2.5)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_out.index, daily_out["ins"],  color=BLUE,   lw=1.0, label="ins")
    ax.plot(daily_out.index, daily_out["outs"], color=ORANGE, lw=1.0, label="outs")
    outlier_days = daily_out[daily_out["is_outlier"]]
    ax.scatter(outlier_days.index, outlier_days["ins"],  color="red", zorder=5, s=40, label="outlier")
    ax.scatter(outlier_days.index, outlier_days["outs"], color="red", zorder=5, s=40)
    ax.set_title("Outlier Detection – Daily Flow (z-score threshold = 2.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Passengers")
    ax.legend()
    save(fig, "13_outliers.png")

    top10 = (
        daily_out[daily_out["is_outlier"]][["ins", "outs", "z_ins", "z_outs"]]
        .assign(max_z=lambda x: x[["z_ins", "z_outs"]].abs().max(axis=1))
        .sort_values("max_z", ascending=False)
        .head(10)
    )
    print("\n  Top 10 anomalous dates:")
    print(top10[["ins", "outs", "z_ins", "z_outs"]].to_string())
except Exception as e:
    print(f"  WARNING: {e}")

# ── plot 14: cumulative flow ───────────────────────────────────────────────────
print("\nPlot 14 – cumulative flow")
try:
    daily_cum = df.set_index("timestamp").resample("D")[["ins", "outs"]].sum().cumsum()
    fig, ax   = plt.subplots(figsize=(14, 6))
    ax.plot(daily_cum.index, daily_cum["ins"],  color=BLUE,   lw=1.5, label="cumulative ins")
    ax.plot(daily_cum.index, daily_cum["outs"], color=ORANGE, lw=1.5, label="cumulative outs")
    ax.set_title("Cumulative Total Passengers – Airport Station")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Passengers")
    ax.legend()
    save(fig, "14_cumulative_flow.png")
except Exception as e:
    print(f"  WARNING: {e}")

# ── summary statistics ─────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  AIRPORT STATION SUMMARY (Københavns Lufthavn)")
print("=" * 58)
print(f"  Total observations     : {len(df):,}")
print(f"  Date range             : {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"  Total ins              : {df['ins'].sum():,.0f}")
print(f"  Total outs             : {df['outs'].sum():,.0f}")
print(f"  Mean ins per obs       : {df['ins'].mean():.2f}")
print(f"  Mean outs per obs      : {df['outs'].mean():.2f}")
print(f"  Peak hour (ins)        : {int(df.groupby('hour')['ins'].mean().idxmax()):02d}:00")
print(f"  Peak hour (outs)       : {int(df.groupby('hour')['outs'].mean().idxmax()):02d}:00")
print(f"  Busiest weekday (ins)  : {df.groupby('weekday')['ins'].mean().idxmax()}")
print(f"  Busiest month (ins)    : {MONTH_NAMES[int(df.groupby('month')['ins'].mean().idxmax()) - 1]}")
print(f"  Zero-activity rows     : {zero_rows:,}")
print("=" * 58)
print(f"\nAll plots saved to: {OUT_DIR}")
