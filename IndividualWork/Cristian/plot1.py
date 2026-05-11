from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("airport_15min_model_data.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

FIG_PATH = FIG_DIR / "slot_total_profile_large.pdf"


# ------------------------------------------------------------
# Load cleaned 15-minute airport dataset
# ------------------------------------------------------------

model_df = pd.read_parquet(DATA_PATH)

model_df["timestamp"] = pd.to_datetime(model_df["timestamp"], utc=True)
model_df = model_df.sort_values("timestamp").reset_index(drop=True)

print(model_df.shape)
print(model_df.head())


# ------------------------------------------------------------
# Mean total passenger flow by 15-minute slot
# ------------------------------------------------------------

slot_total = (
    model_df
    .groupby("slot")["total"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

slot_total["se"] = slot_total["std"] / np.sqrt(slot_total["count"])
slot_total["ci_low"] = slot_total["mean"] - 1.96 * slot_total["se"]
slot_total["ci_high"] = slot_total["mean"] + 1.96 * slot_total["se"]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4.2))

ax.plot(
    slot_total["slot"],
    slot_total["mean"],
    linewidth=2.5,
    label="Mean total flow",
)

ax.fill_between(
    slot_total["slot"],
    slot_total["ci_low"],
    slot_total["ci_high"],
    alpha=0.25,
    label="95% confidence band",
)

ax.set_title(
    "Mean Total Passenger Flow by 15-Minute Slot",
    fontsize=18,
    pad=12,
)

ax.set_xlabel("Time of day", fontsize=15)
ax.set_ylabel("Mean total passengers", fontsize=15)

ax.set_xticks([0, 24, 48, 72, 95])
ax.set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00", "23:45"],
    fontsize=13,
)

ax.tick_params(axis="y", labelsize=13)

ax.grid(alpha=0.3)

ax.legend(
    frameon=False,
    fontsize=13,
    loc="best",
)

fig.tight_layout()
fig.savefig(FIG_PATH, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {FIG_PATH}")