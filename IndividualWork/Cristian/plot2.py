from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("airport_15min_model_data.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

FIG_PATH = FIG_DIR / "slot_direction_profile_large.pdf"


model_df = pd.read_parquet(DATA_PATH)

model_df["timestamp"] = pd.to_datetime(model_df["timestamp"], utc=True)
model_df = model_df.sort_values("timestamp").reset_index(drop=True)


slot_direction = (
    model_df
    .groupby("slot")["outs_share"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

slot_direction["se"] = slot_direction["std"] / np.sqrt(slot_direction["count"])
slot_direction["ci_low"] = slot_direction["mean"] - 1.96 * slot_direction["se"]
slot_direction["ci_high"] = slot_direction["mean"] + 1.96 * slot_direction["se"]

slot_direction["ci_low"] = slot_direction["ci_low"].clip(lower=0)
slot_direction["ci_high"] = slot_direction["ci_high"].clip(upper=1)


fig, ax = plt.subplots(figsize=(9, 4.2))

ax.plot(
    slot_direction["slot"],
    slot_direction["mean"],
    linewidth=2.5,
    label="Mean out-share",
)

ax.fill_between(
    slot_direction["slot"],
    slot_direction["ci_low"],
    slot_direction["ci_high"],
    alpha=0.25,
    label="95% confidence band",
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.4,
    color="black",
    label="Balanced flow",
)

ax.set_title(
    "Mean Airport-Bound Share by 15-Minute Slot",
    fontsize=18,
    pad=12,
)

ax.set_xlabel("Time of day", fontsize=15)
ax.set_ylabel(r"Mean $outs_t / N_t$", fontsize=15)

ax.set_xticks([0, 24, 48, 72, 95])
ax.set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00", "23:45"],
    fontsize=13,
)

ax.tick_params(axis="y", labelsize=13)

ax.set_ylim(0, 1)
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