"""
Discover latent metro operational modes and test whether stations enter each
mode at the same time or as a wave pattern.

Model
Hierarchical Poisson HMM with station-specific bounded latent timing offsets.

For station s, timestep t, and hidden mode k:

    z[s,t]                 : latent operational mode
    ins[s,t], outs[s,t]    : Poisson emissions
    lambda_in[k,s]         : station/mode-specific entry rate
    lambda_out[k,s]        : station/mode-specific exit rate
    delay[s,k]             : station/mode-specific timing shift in hours

The key model-based output is delay[s,k].  A negative morning delay means the
station reaches morning-rush earlier than the canonical morning-rush center; a
positive delay means later.  Diagnostic threshold-onset plots are still exported,
but they are not the main scientific answer because busy central stations often
have mixed posterior state probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d
from transformers import MODEL_FOR_OBJECT_DETECTION_MAPPING
import re
from operational_modes.metro_mode_plots import plot_all, plot_station_timing_map

torch.manual_seed(42)
pyro.set_rng_seed(42)

_N_THREADS = min(6, max(1, os.cpu_count() or 1))
torch.set_num_threads(_N_THREADS)
print(f"PyTorch using {_N_THREADS} CPU threads.")

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "metro_combined.csv"
OUT_DIR = HERE
CHECKPOINT_PATH = OUT_DIR / "metro_hmm_checkpoint.pt"

RESAMPLE_FREQ = "15min"
DATE_START = "2024-04-01"  # widen to e.g. "2024-01-01" for a full-year run
DATE_END = "2025-10-01"

STATE_NAMES = ["Night", "Off-peak", "Morning-rush", "Evening-rush"]
STATE_COLORS = ["#1a1a2e", "#4a90d9", "#e74c3c", "#f39c12"]
NUM_STATES = 4

# 15-min data: 96 timesteps/day
STEPS_PER_DAY = 96
BATCH_DAYS = 20

# Time centers  defined hours
MODE_CENTER_HOURS = torch.tensor([2.0, 12.0, 7.5, 17.0])
'''
controls how wide or strict each mode is around its center
Small σ values are more tight, while larger value are more flexible
'''
# SIGMA_HOURS = torch.tensor([2.7, 3.0, 1.7, 1.9])
# TIME_DAY_INFL = 0.15
TIME_DAY_INFL = 0.6
SIGMA_HOURS = torch.tensor([2.0, 2.5, 1.0, 1.2])

MAX_DELAY_HOURS = 3.0
DELAY_PRIOR = 1.0

MODE_DETECT_THRESHOLD = 0.25
MODE_MIN_STEPS = 4
MODE_ACTIVE_TIME_WINDOWS: Dict[str, Tuple[int, int, bool]] = {
    "Night": (21, 5, True),
    "Off-peak": (10, 16, False),
    "Morning-rush": (5, 11, False),
    "Evening-rush": (15, 21, False),
}


# def load_data(
#         path: Path = DATA_PATH,
#         resample_freq: str = RESAMPLE_FREQ,
#         date_start: str | None = DATE_START,
#         date_end: str | None = DATE_END,
#         device: torch.device = torch.device("cpu"),
# ):
#     """
#     Load metro passenger data and return scaled count tensors.
#     The Poisson likelihood is fit on scaled counts to keep gradients stable.
#     """
#
#     # Load parquet data with optional date filter
#
#     dataset = ds.dataset(str(path), format="parquet")
#     date_selection_filter = None
#
#     if date_start is not None:
#         start_utc = pd.Timestamp(date_start, tz="UTC")
#         date_selection_filter = ds.field("timestamp") >= start_utc
#
#     if date_end is not None:
#         end_utc = pd.Timestamp(date_end, tz="UTC")
#         end_filter = ds.field("timestamp") < end_utc
#         date_selection_filter = end_filter if date_selection_filter is None else date_selection_filter & end_filter
#
#     if date_selection_filter is not None:
#         print(f"Date filter: {date_start} -> {date_end}")
#
#     df = dataset.to_table(filter=date_selection_filter).to_pandas()
#     print(f"Loaded shape: {df.shape}")
#
#     # Clean timestamps and count columns
#
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#
#     for col in ("ins", "outs", "passengerNo"):
#         if col in df.columns:
#             df[col] = (
#                 pd.to_numeric(df[col], errors="coerce")
#                 .fillna(0)
#                 .clip(lower=0)
#             )
#
#     # Extract station_ids and station_names
#
#     meta = (
#         df[["stationId", "system", "station"]]
#         .drop_duplicates("stationId")
#         .set_index("stationId")
#         .sort_index()
#     )
#
#     station_ids = meta.index.tolist()
#     station_names = meta["station"].tolist()
#
#     df = (
#         df.groupby(["stationId", "timestamp"], as_index=False)[["ins", "outs"]]
#         .sum())
#
#     # Resample each station to regular time intervals
#
#     print(f"Resampling to {resample_freq} ...")
#
#     df = (
#         df.set_index("timestamp")
#         .groupby("stationId")[["ins", "outs"]]
#         .resample(resample_freq)
#         .sum()
#         .reset_index()
#     )
#
#     print(f"After resample: {df.shape[0]:,} rows")
#
#     # Build time index
#
#     time_index = pd.date_range(
#         start=df["timestamp"].min(),
#         end=df["timestamp"].max(),
#         freq=resample_freq,
#     )
#
#     S = len(station_ids)
#     T = len(time_index)
#
#     print(
#         f"Stations: {S}   Timesteps: {T}   "
#         f"({time_index[0]} -> {time_index[-1]})"
#     )
#
#     print(
#         f"Memory for gamma [S={S}, T={T}, K={NUM_STATES}]: "
#         f"{S * T * NUM_STATES * 4 / 1e6:.1f} MB"
#     )
#
#     df_indexed = df.set_index(["stationId", "timestamp"])
#
#     ins_wide = (
#         df_indexed["ins"]
#         .unstack("timestamp")
#         .reindex(index=station_ids, columns=time_index, fill_value=0)
#         .fillna(0)
#         .round()
#         .astype(np.float32)
#     )
#
#     outs_wide = (
#         df_indexed["outs"]
#         .unstack("timestamp")
#         .reindex(index=station_ids, columns=time_index, fill_value=0)
#         .fillna(0)
#         .round()
#         .astype(np.float32)
#     )
#
#     # Convert matrices to PyTorch tensors
#
#     ins_raw = torch.tensor(
#         ins_wide.values,
#         dtype=torch.float32,
#         device=device,
#     )
#
#     outs_raw = torch.tensor(
#         outs_wide.values,
#         dtype=torch.float32,
#         device=device,
#     )
#
#     # Scale observations for stable Poisson training
#
#     obs_scale = max(float(ins_raw.mean()), float(outs_raw.mean()), 1.0)
#
#     print(
#         f"Observation scale factor: {obs_scale:.3f} "
#         "(rates are rescaled for output)"
#     )
#
#     ins = ins_raw / obs_scale
#     outs = outs_raw / obs_scale
#
#     # 10. Trim to full days
#
#     if T % STEPS_PER_DAY != 0:
#         T_trim = (T // STEPS_PER_DAY) * STEPS_PER_DAY
#
#         ins = ins[:, :T_trim]
#         outs = outs[:, :T_trim]
#         time_index = time_index[:T_trim]
#         T = T_trim
#
#         print(
#             f"Trimmed to {T} timesteps "
#             f"({T // STEPS_PER_DAY} complete days)."
#         )
#
#     # 11. Reshape to station × day × timestep format
#
#     D = T // STEPS_PER_DAY
#
#     ins_3d = ins.reshape(S, D, STEPS_PER_DAY)
#     outs_3d = outs.reshape(S, D, STEPS_PER_DAY)
#     print(f"Daily mini-batch view: [S={S}, D={D}, T_day={STEPS_PER_DAY}]")
#
#     return ins, outs, ins_3d, outs_3d, station_names, time_index, meta, obs_scale, D
def load_data(
        path: Path = DATA_PATH,
        resample_freq: str = RESAMPLE_FREQ,
        date_start: str | None = DATE_START,
        date_end: str | None = DATE_END,
        device: torch.device = torch.device("cpu"),
):
    print(f"Date filter: {date_start} -> {date_end}")

    # --- Read CSV ---
    df = pd.read_csv(
        path,
        usecols=["timestamp", "stationId", "stationName", "ins", "outs", "passengerNo", "location"],
        low_memory=False,
    )
    print(f"Loaded shape: {df.shape}")

    # --- Parse timestamps ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # --- Date filter ---
    if date_start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(date_start, tz="UTC")]
    if date_end is not None:
        df = df[df["timestamp"] < pd.Timestamp(date_end, tz="UTC")]

    # --- Clean numeric columns ---
    for col in ("ins", "outs", "passengerNo"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # --- Parse location column → lat, lon ---
    # Handles WKT "POINT (lon lat)", tuple "(lat, lon)", or "lat,lon"
    def _parse_location(loc):
        if pd.isna(loc):
            return np.nan, np.nan
        loc = str(loc).strip()
        # WKT format: POINT (12.568 55.676)  or POINT(12.568 55.676)
        wkt_match = re.match(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", loc, re.IGNORECASE)
        if wkt_match:
            lon, lat = float(wkt_match.group(1)), float(wkt_match.group(2))
            return lat, lon
        # "lat,lon" or "(lat, lon)" or "lat lon"
        nums = re.findall(r"[\d.\-]+", loc)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        return np.nan, np.nan

    coords = df[["stationId", "stationName", "location"]].drop_duplicates("stationId")
    coords[["lat", "lon"]] = coords["location"].apply(
        lambda x: pd.Series(_parse_location(x))
    )

    # --- Build meta ---
    meta = (
        coords[["stationId", "stationName", "lat", "lon"]]
        .set_index("stationId")
        .sort_index()
        .rename(columns={"stationName": "station"})
    )

    station_ids   = meta.index.tolist()
    station_names = meta["station"].tolist()

    # --- Aggregate and resample ---
    df = (
        df.groupby(["stationId", "timestamp"], as_index=False)[["ins", "outs"]]
        .sum()
    )

    print(f"Resampling to {resample_freq} ...")
    df = (
        df.set_index("timestamp")
        .groupby("stationId")[["ins", "outs"]]
        .resample(resample_freq)
        .sum()
        .reset_index()
    )
    print(f"After resample: {df.shape[0]:,} rows")

    # --- Time index ---
    time_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=resample_freq,
    )

    S = len(station_ids)
    T = len(time_index)
    print(f"Stations: {S}   Timesteps: {T}   ({time_index[0]} -> {time_index[-1]})")
    print(f"Memory for gamma [S={S}, T={T}, K={NUM_STATES}]: {S * T * NUM_STATES * 4 / 1e6:.1f} MB")

    df_indexed = df.set_index(["stationId", "timestamp"])

    ins_wide = (
        df_indexed["ins"]
        .unstack("timestamp")
        .reindex(index=station_ids, columns=time_index, fill_value=0)
        .fillna(0).round().astype(np.float32)
    )
    outs_wide = (
        df_indexed["outs"]
        .unstack("timestamp")
        .reindex(index=station_ids, columns=time_index, fill_value=0)
        .fillna(0).round().astype(np.float32)
    )

    ins_raw  = torch.tensor(ins_wide.values,  dtype=torch.float32, device=device)
    outs_raw = torch.tensor(outs_wide.values, dtype=torch.float32, device=device)

    obs_scale = max(float(ins_raw.mean()), float(outs_raw.mean()), 1.0)
    print(f"Observation scale factor: {obs_scale:.3f} (rates are rescaled for output)")

    ins  = ins_raw  / obs_scale
    outs = outs_raw / obs_scale

    if T % STEPS_PER_DAY != 0:
        T_trim   = (T // STEPS_PER_DAY) * STEPS_PER_DAY
        ins      = ins[:, :T_trim]
        outs     = outs[:, :T_trim]
        time_index = time_index[:T_trim]
        T        = T_trim

    D       = T // STEPS_PER_DAY
    ins_3d  = ins.reshape(S, D, STEPS_PER_DAY)
    outs_3d = outs.reshape(S, D, STEPS_PER_DAY)

    print(f"Daily mini-batch view: [S={S}, D={D}, T_day={STEPS_PER_DAY}]")
    return ins, outs, ins_3d, outs_3d, station_names, time_index, meta, obs_scale, D

@torch.jit.script
def _hmm_forward(
        log_emit_t: torch.Tensor,  # [T, N, K]
        log_A: torch.Tensor,  # [K, K]
        log_init: torch.Tensor,  # [K]
) -> torch.Tensor:
    T = log_emit_t.shape[0]
    log_alphas = torch.empty_like(log_emit_t)
    log_alphas[0] = log_init.unsqueeze(0) + log_emit_t[0]
    for t in range(1, T):
        log_alphas[t] = (log_alphas[t - 1].unsqueeze(2) + log_A.unsqueeze(0)).logsumexp(1) + log_emit_t[t]
    return log_alphas


@torch.jit.script
# Compute α_t(k) = P(observations up to time t AND state = k at time t)
def _hmm_backward(
        log_emit_t: torch.Tensor,
        log_A: torch.Tensor,
) -> torch.Tensor:
    T = log_emit_t.shape[0]
    log_betas = torch.zeros_like(log_emit_t)
    for t in range(T - 2, -1, -1):
        log_betas[t] = (
                log_betas[t + 1].unsqueeze(1)
                + log_A.unsqueeze(0)
                + log_emit_t[t + 1].unsqueeze(1)
        ).logsumexp(2)
    return log_betas


# β_t(k) = P(future observations | state = k at time t)
@torch.jit.script
def _hmm_log_marginal(
        log_emit: torch.Tensor,
        log_A: torch.Tensor,
        log_init: torch.Tensor,
) -> torch.Tensor:
    log_emit_t = log_emit.permute(1, 0, 2)
    log_alphas = _hmm_forward(log_emit_t, log_A, log_init)
    return log_alphas[-1].logsumexp(-1)


'''
This computes the Poisson log-probability mass function (log-PMF):
logP(X=x|λ)=x lnλ -λ -lnΓ(x+1)
'''


def _poisson_log_prob(x: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Poisson log-probability using lgamma, stable for float tensors."""
    rate = rate.clamp_min(1e-6)
    return torch.xlogy(x, rate) - rate - (x + 1.0).lgamma()


def _circular_hour_diff(hour: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Shortest signed distance between hour and center on a 24h circle."""
    return ((hour - center + 12.0) % 24.0) - 12.0


def _bounded_delay(delay: torch.Tensor) -> torch.Tensor:
    """Map unconstrained raw delay to a realistic station timing offset in hours."""
    return MAX_DELAY_HOURS * torch.tanh(delay)


def _bounded_delay_scale(raw_loc: torch.Tensor, raw_scale: torch.Tensor) -> torch.Tensor:
    """Delta-method approximation for uncertainty after tanh transformation."""
    derivative = MAX_DELAY_HOURS * (1.0 - torch.tanh(raw_loc).pow(2))
    return derivative.abs() * raw_scale


'''
TODO
MEAN VALUE PER STATION OR GENERAL MEAN VALUE
PLOT -> MAP per time (cluster multiple stations) per mode
AUTO_DETECT modes 
'''


def _station_timeOfDay_log_score(
        hours: torch.Tensor,  # [T]
        station_delay: torch.Tensor,  # [S, K]
) -> torch.Tensor:
    device = station_delay.device
    K = station_delay.shape[1]

    # Base definitions — extend if K > len of defaults
    _base_centers = torch.tensor([2.0, 12.0, 7.5, 17.0])  # Night, Off-peak, Morning, Evening
    _base_sigmas = torch.tensor([2.0, 2.5, 1.0, 1.2])

    if K <= len(_base_centers):
        centers = _base_centers[:K].to(device)
        sigmas = _base_sigmas[:K].to(device)
    else:
        # Pad extra states with generic off-peak values
        n_extra = K - len(_base_centers)
        extra_centers = torch.linspace(9.0, 20.0, n_extra)  # spread extras across the day
        extra_sigmas = torch.full((n_extra,), 2.0)  # moderate width
        centers = torch.cat([_base_centers, extra_centers]).to(device)
        sigmas = torch.cat([_base_sigmas, extra_sigmas]).to(device)

    h = hours.to(device)
    effective_center = centers.unsqueeze(0) + station_delay  # [S, K]
    diff = _circular_hour_diff(h.unsqueeze(0).unsqueeze(-1),  # [1, T, 1]
                               effective_center.unsqueeze(1))  # [S, 1, K]
    return -0.5 * (diff / sigmas.unsqueeze(0).unsqueeze(0)) ** 2  # [S, T, K]


def model(ins_b: torch.Tensor, outs_b: torch.Tensor, n_states: int = NUM_STATES):
    # Daily mini-batched hierarchical Poisson HMM. Each station-day is treated as one repeated daily HMM sequence.
    S, B, T_day = ins_b.shape
    device = ins_b.device

    # log_A[i, j] = log probability of transitioning from state i to state j
    log_A_params = pyro.sample(
        "log_A",
        dist.Normal(torch.zeros(n_states, n_states, device=device),
                    torch.ones(n_states, n_states, device=device)).to_event(2),
    )
    log_A = F.log_softmax(log_A_params, dim=-1)

    # Learned initial distribution for station-day chunks.
    init_logits = pyro.sample(
        "init_logits",
        dist.Normal(torch.zeros(n_states, device=device), torch.ones(n_states, device=device)).to_event(1),
    )
    log_init = F.log_softmax(init_logits, dim=-1)

    # Station/mode Poisson rates on the scaled observation axis.
    prior_alpha = torch.full((n_states, S), 3.0, device=device)
    prior_beta = torch.full((n_states, S), 3.0, device=device)
    prior_alpha[0, :] = 1.5
    prior_beta[0, :] = 7.5

    with pyro.plate("state_plate", n_states, dim=-2):
        with pyro.plate("station_plate_emit", S, dim=-1):
            lambda_in = pyro.sample("lambda_in", dist.Gamma(prior_alpha, prior_beta))
            lambda_out = pyro.sample("lambda_out", dist.Gamma(prior_alpha, prior_beta))

    # Raw latent timing offsets are bounded through tanh.
    raw_station_delay = pyro.sample(
        "raw_station_delay",
        dist.Normal(
            torch.zeros(S, n_states, device=device),
            torch.full((S, n_states), DELAY_PRIOR, device=device),
        ).to_event(2),
    )
    station_delay = _bounded_delay(raw_station_delay)

    #  Build the emission log-probabilities
    ins = ins_b.reshape(S * B, T_day)
    outs = outs_b.reshape(S * B, T_day)

    # The rate tensors are replicated B times so every day for station s references the same
    lam_in_exp = lambda_in.T.repeat_interleave(B, dim=0)
    lam_out_exp = lambda_out.T.repeat_interleave(B, dim=0)

    log_emit_poisson = (
            _poisson_log_prob(ins.unsqueeze(-1), lam_in_exp.unsqueeze(1))
            + _poisson_log_prob(outs.unsqueeze(-1), lam_out_exp.unsqueeze(1))
    )

    hours_day = torch.arange(T_day, dtype=torch.float32, device=device) / 4.0
    station_log_score = _station_timeOfDay_log_score(hours_day, station_delay)
    station_log_score = station_log_score.repeat_interleave(B, dim=0)

    log_emit = log_emit_poisson + TIME_DAY_INFL * station_log_score
    log_prob = _hmm_log_marginal(log_emit, log_A, log_init)

    # Optimise the average station-day log likelihood
    pyro.factor("hmm_obs", log_prob.mean())


'''
Guide is a tractable approximating distribution q(latents) that SVI optimises to be as close as possible
to that true posterior by minimising KL divergence. We use mean-field variational inference: every latent variable gets 
its own independent approximate posterior, with no correlations between them.

'''


def guide(ins_b: torch.Tensor, outs_b: torch.Tensor, n_states: int = NUM_STATES):
    S = ins_b.shape[0]
    device = ins_b.device

    # Transition guide: diagonal-dominant initialization.
    log_A_init = torch.zeros(n_states, n_states, device=device) + 4.0 * torch.eye(n_states, device=device)
    log_A_mu = pyro.param("log_A_mu", log_A_init)
    log_A_scale_raw = pyro.param("log_A_scale_raw", torch.full((n_states, n_states), -2.3, device=device))
    log_A_scale = F.softplus(log_A_scale_raw) + 1e-4
    pyro.sample("log_A", dist.Normal(log_A_mu, log_A_scale).to_event(2))

    # Initial distribution guide. Weak prior; learned from station-day chunks.
    _default_logits = torch.tensor([1.0, 0.2, -0.2, -0.2], dtype=torch.float32, device=device)
    if n_states <= len(_default_logits):
        init_vals = _default_logits[:n_states]
    else:
        init_vals = torch.cat([_default_logits, torch.zeros(n_states - len(_default_logits), device=device)])

    init_logits_loc = pyro.param("init_logits_mu", init_vals)
    init_logits_scale_raw = pyro.param("init_logits_scale_raw", torch.full((n_states,), -2.0, device=device))
    init_logits_scale = F.softplus(init_logits_scale_raw) + 1e-4
    pyro.sample("init_logits", dist.Normal(init_logits_loc, init_logits_scale).to_event(1))

    # Empirical-Bayes initialization of emission guide means from scaled data.
    station_mean_in = ins_b.mean(dim=(1, 2)).clamp_min(0.05)
    station_mean_out = outs_b.mean(dim=(1, 2)).clamp_min(0.05)
    '''
    Rather than starting all rates at the same value, each station gets a personalised starting point scaled from 
    its own observed mean (empirical Bayes)
    '''
    _default_multipliers = torch.tensor([0.20, 0.80, 2.00, 1.60], device=device)
    if n_states <= len(_default_multipliers):
        mult_vals = _default_multipliers[:n_states]
    else:
        extra = torch.ones(n_states - len(_default_multipliers), device=device)
        mult_vals = torch.cat([_default_multipliers, extra])

    multipliers = mult_vals.unsqueeze(1)
    init_mean_in = (multipliers * station_mean_in.unsqueeze(0)).clamp_min(0.02)
    init_mean_out = (multipliers * station_mean_out.unsqueeze(0)).clamp_min(0.02)

    def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-4)
        return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

    lambda_in_raw = pyro.param("lambda_in_raw", _inv_softplus(init_mean_in.clone()))
    lambda_out_raw = pyro.param("lambda_out_raw", _inv_softplus(init_mean_out.clone()))
    lambda_in_q = F.softplus(lambda_in_raw).clamp(min=1e-4, max=50.0)
    lambda_out_q = F.softplus(lambda_out_raw).clamp(min=1e-4, max=50.0)

    with pyro.plate("state_plate", n_states, dim=-2):
        with pyro.plate("station_plate_emit", S, dim=-1):
            pyro.sample("lambda_in", dist.Delta(lambda_in_q))
            pyro.sample("lambda_out", dist.Delta(lambda_out_q))

    # Station-specific timing offsets: variational family on raw delay, then
    # transformed to bounded hours in posterior_params().
    raw_delay_init = torch.zeros(S, n_states, device=device)
    raw_delay = pyro.param("raw_station_delay_mu", raw_delay_init)
    raw_station_delay_scale = pyro.param("raw_station_delay_scale", torch.full((S, n_states), -1.2, device=device))
    raw_station_delay_scale = F.softplus(raw_station_delay_scale) + 1e-3
    pyro.sample("raw_station_delay", dist.Normal(raw_delay, raw_station_delay_scale).to_event(2))


def train(
        ins_3d: torch.Tensor,
        outs_3d: torch.Tensor,
        D: int,
        n_states: int = NUM_STATES,
        n_steps: int = 600,
        batch_days: int = BATCH_DAYS,
        lr: float = 0.002,
        clip_norm: float = 0.5,
        smoothing: int = 50,
) -> List[float]:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("Run: pip install tqdm") from exc

    pyro.clear_param_store()
    svi = SVI(
        model,
        guide,
        ClippedAdam({"lr": lr, "lrd": 0.9995, "clip_norm": clip_norm}),
        loss=Trace_ELBO(),
    )

    losses: List[float] = []
    prev_smooth = None
    bar = tqdm(range(n_steps), desc="SVI training", unit="step", dynamic_ncols=True, colour="blue")

    for _ in bar:
        B = min(batch_days, D)
        day_idx = torch.randperm(D, device=ins_3d.device)[:B]
        ins_b = ins_3d[:, day_idx, :]
        outs_b = outs_3d[:, day_idx, :]

        loss = float(svi.step(ins_b, outs_b, n_states))
        losses.append(loss)
        window = losses[-smoothing:]
        smooth = sum(window) / len(window)
        delta = 0.0 if prev_smooth is None else smooth - prev_smooth
        prev_smooth = smooth
        bar.set_postfix(ELBO=f"{smooth:,.0f}", delta=f"{delta:+,.0f}", refresh=False)

    bar.close()
    return losses


@torch.no_grad()
def posterior_params(obs_scale: float = 1.0):
    log_A = F.log_softmax(pyro.param("log_A_mu").detach(), dim=-1)
    A = log_A.exp()
    log_init = F.log_softmax(pyro.param("init_logits_mu").detach(), dim=-1)

    lambda_in_scaled = F.softplus(pyro.param("lambda_in_raw").detach()).clamp(min=1e-4, max=50.0)
    lambda_out_scaled = F.softplus(pyro.param("lambda_out_raw").detach()).clamp(min=1e-4, max=50.0)

    raw_delay = pyro.param("raw_station_delay_mu").detach()
    print(f"Raw delay -> {raw_delay}")
    raw_delay_scale = F.softplus(pyro.param("raw_station_delay_scale").detach()) + 1e-3
    station_delay = _bounded_delay(raw_delay)
    station_delay_scale = _bounded_delay_scale(raw_delay, raw_delay_scale)

    # Keep both scaled rates for posterior inference and real rates for output.
    lambda_in = lambda_in_scaled * obs_scale
    lambda_out = lambda_out_scaled * obs_scale
    return log_A, A, log_init, lambda_in_scaled, lambda_out_scaled, lambda_in, lambda_out, station_delay, station_delay_scale


@torch.no_grad()
def compute_posterior_probs(
        ins: torch.Tensor,
        outs: torch.Tensor,
        time_index: pd.DatetimeIndex,
        obs_scale: float,
):
    log_A, A, log_init, lambda_in_scaled, lambda_out_scaled, lambda_in, lambda_out, station_delay, station_delay_scale = posterior_params(
        obs_scale)

    # computes how likely each observation (ins, outs) is under each of the 4 hidden states, using the Poisson log-PMF
    log_emit_poisson = (
            _poisson_log_prob(ins.unsqueeze(-1), lambda_in_scaled.T.unsqueeze(1))
            + _poisson_log_prob(outs.unsqueeze(-1), lambda_out_scaled.T.unsqueeze(1))
    )

    hours = torch.tensor(
        time_index.hour.values + time_index.minute.values / 60.0,
        dtype=torch.float32,
        device=ins.device,
    )
    station_tod_log_score = _station_timeOfDay_log_score(hours, station_delay)
    log_emit = log_emit_poisson + TIME_DAY_INFL * station_tod_log_score

    log_emit_t = log_emit.permute(1, 0, 2)
    log_alphas = _hmm_forward(log_emit_t, log_A, log_init)
    log_betas = _hmm_backward(log_emit_t, log_A)

    '''
    combines forward and backward: log_gamma = log_α + log_β, then normalises across states. gamma[s,t,k] is the 
    posterior probability that station s is in mode k at timestep t, using all observations.
    '''
    log_gamma = log_alphas.permute(1, 0, 2) + log_betas.permute(1, 0, 2)
    log_gamma = log_gamma - log_gamma.logsumexp(2, keepdim=True)
    gamma = log_gamma.exp()

    return gamma, lambda_in, lambda_out, A, log_init.exp(), station_delay, station_delay_scale


# =============================================================================
# 5. MODEL-BASED WAVE ANALYSIS
# =============================================================================

def export_station_delays(
        station_names: List[str],
        station_delay: torch.Tensor,
        station_delay_scale: torch.Tensor,
        out_path: Path,
) -> pd.DataFrame:
    delay = station_delay.cpu().numpy()
    scale = station_delay_scale.cpu().numpy()

    rows = []
    for s, station in enumerate(station_names):
        row = {"station": station}
        for k, mode in enumerate(STATE_NAMES):
            lo = delay[s, k] - 1.96 * scale[s, k]
            hi = delay[s, k] + 1.96 * scale[s, k]
            row[f"{mode}_delay_h"] = delay[s, k]
            row[f"{mode}_delay_sd_h"] = scale[s, k]
            row[f"{mode}_delay_lo95_h"] = lo
            row[f"{mode}_delay_hi95_h"] = hi
            row[f"{mode}_relative_center"] = _hour_to_hhmm(float(MODE_CENTER_HOURS[k]) + delay[s, k])
            row[f"{mode}_relative_center_lo95"] = _hour_to_hhmm(float(MODE_CENTER_HOURS[k]) + lo)
            row[f"{mode}_relative_center_hi95"] = _hour_to_hhmm(float(MODE_CENTER_HOURS[k]) + hi)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Station-specific latent timing offsets -> {out_path}")
    return df


def _hour_to_hhmm(hour: float) -> str:
    mins = int(round((hour % 24.0) * 60)) % (24 * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _min_to_hhmm(minutes: float) -> str:
    if pd.isna(minutes):
        return "—"
    m = int(round(minutes)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _min_to_datenum(minutes: float, day0: pd.Timestamp) -> float:
    ts = day0 + pd.Timedelta(minutes=float(minutes))
    return mdates.date2num(ts.to_pydatetime())


def export_posterior_states(
        gamma: torch.Tensor,
        station_names: List[str],
        time_index: pd.DatetimeIndex,
        out_path: Path,
):
    S, T, K = gamma.shape
    g_np = gamma.cpu().numpy()
    data = {
        "station": np.repeat(station_names, T),
        "timestamp": np.tile(time_index, S),
    }
    for k, name in enumerate(STATE_NAMES):
        data[f"p_{name.replace('-', '_')}"] = g_np[:, :, k].ravel()
    hard_idx = g_np.argmax(axis=2).ravel()
    data["hard_state"] = np.array(STATE_NAMES)[hard_idx]
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Posterior state probabilities -> {out_path}")


def save_checkpoint(
        losses: List[float],
        obs_scale: float,
        station_names: List[str],
        time_index: pd.DatetimeIndex,
        gamma: torch.Tensor,
        lambda_in: torch.Tensor,
        lambda_out: torch.Tensor,
        A: torch.Tensor,
        init_probs: torch.Tensor,
        station_delay: torch.Tensor,
        station_delay_scale: torch.Tensor,
        meta: pd.DataFrame,
        path: Path,
) -> None:
    """
    Save the complete training result to a single .pt file.

    What is saved
    -------------
    param_store          : all Pyro variational parameters (restores the guide)
    losses               : ELBO loss history
    obs_scale            : data normalisation factor (needed for posterior_params)
    station_names        : list of station name strings
    time_index           : pd.DatetimeIndex of the observation window
    gamma                : [S, T, K] posterior state probabilities
    lambda_in/out        : [K, S] Poisson rates in pax/15 min
    A                    : [K, K] transition matrix
    init_probs           : [K] learned initial distribution
    station_delay        : [S, K] timing offsets in hours
    station_delay_scale  : [S, K] posterior uncertainty on delays
    """
    param_store_dict = {k: v.detach().cpu() for k, v in pyro.get_param_store().items()}

    torch.save(
        {
            "param_store": param_store_dict,
            "losses": losses,
            "obs_scale": obs_scale,
            "station_names": station_names,
            "time_index": time_index,
            "gamma": gamma.cpu(),
            "lambda_in": lambda_in.cpu(),
            "lambda_out": lambda_out.cpu(),
            "A": A.cpu(),
            "init_probs": init_probs.cpu(),
            "station_delay": station_delay.cpu(),
            "station_delay_scale": station_delay_scale.cpu(),
            "meta": meta,
        },
        path
    )
    print(f"Checkpoint saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(path: Path) -> Optional[dict]:
    """
    Load a previously saved checkpoint.
    Returns None if the file does not exist (triggers fresh training).
    Restores the Pyro param store so posterior_params() works immediately.
    """
    if not path.exists():
        return None

    print(f"Checkpoint found ← {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Restore Pyro param store so compute_posterior_probs() can be called again
    pyro.clear_param_store()
    for k, v in data["param_store"].items():
        pyro.get_param_store()[k] = v.clone()

    print(
        f"  Loaded: {len(data['station_names'])} stations, "
        f"gamma {tuple(data['gamma'].shape)}, "
        f"{len(data['losses'])} training steps"
    )
    return data


# def main():
#     device = torch.device("cpu")
#
#     ins, outs, ins_3d, outs_3d, station_names, time_index, meta, obs_scale, D = load_data(DATA_PATH, device=device)
#
#     print("\nTraining hierarchical Poisson HMM with bounded station-specific timing offsets ...")
#     losses = train(ins_3d, outs_3d, D)
#
#     print("\nRunning forward-backward posterior smoothing ...")
#     gamma, lambda_in, lambda_out, A, init_probs, station_delay, station_delay_scale = compute_posterior_probs(
#         ins, outs, time_index, obs_scale
#     )
#
#     S, T, K = gamma.shape
#     print(f"\nPosterior Poisson rates per mode (mean across {S} stations, pax/15-min):")
#     print(f"  {'Mode':<14} {'lambda_in':>12} {'lambda_out':>12}")
#     print("  " + "-" * 42)
#     for k, name in enumerate(STATE_NAMES):
#         print(f"  {name:<14} {lambda_in[k].mean().item():>12.2f} {lambda_out[k].mean().item():>12.2f}")
#
#     print("\nPosterior transition matrix A[from -> to]:")
#     print("              " + "".join(f"{n:>14}" for n in STATE_NAMES))
#     for i, name in enumerate(STATE_NAMES):
#         print(f"  {name:<12}" + "".join(f"{A[i, j].item():>14.3f}" for j in range(NUM_STATES)))
#
#     print("\nLearned initial distribution for station-day chunks:")
#     for name, p in zip(STATE_NAMES, init_probs):
#         print(f"  {name:<14} {p.item():.3f}")
#
#     delay_df = export_station_delays(
#         station_names,
#         station_delay,
#         station_delay_scale,
#         OUT_DIR / "metro_station_mode_delays.csv",
#     )
#     print("\nModel-based wave summary: morning-rush timing offsets")
#     print(
#         delay_df[[
#             "station",
#             "Morning-rush_delay_h",
#             "Morning-rush_delay_lo95_h",
#             "Morning-rush_delay_hi95_h",
#             "Morning-rush_relative_center",
#         ]]
#         .sort_values("Morning-rush_delay_h")
#         .to_string(index=False)
#     )
#
#     export_posterior_states(gamma, station_names, time_index, OUT_DIR / "metro_posterior_states.csv")
#
#     plot_results(
#         gamma,
#         lambda_in,
#         lambda_out,
#         A,
#         init_probs,
#         losses,
#         station_names,
#         time_index,
#         station_delay,
#         station_delay_scale,
#         OUT_DIR / "metro_hmm_results.png",
#     )
#
#     return gamma, lambda_in, lambda_out, station_delay

def sweep_n_states(
        ins_3d, outs_3d, D,
        candidates=(2, 3, 4, 5, 6),
        n_steps=300,
        out_path=None,
):
    rows = []
    S = ins_3d.shape[0]

    for k in candidates:
        pyro.clear_param_store()
        losses = train(ins_3d, outs_3d, D, n_states=k, n_steps=n_steps)
        final_elbo = losses[-1]

        # k*k          : transition matrix A (mu)
        # k            : initial distribution
        # 2*k*S        : lambda_in + lambda_out (point-estimate params)
        # 2*S*k        : station timing delays (mu + scale per entry)
        n_params = k * k + k + 2 * k * S + 2 * S * k

        rows.append({
            "K": k,
            "final_elbo": final_elbo,
            "n_params": n_params,
        })
        print(f"  K={k}  ELBO={final_elbo:.1f}  params={n_params}")


    df = pd.DataFrame(rows)

    # Marginal ELBO gain per extra parameter — the real criterion
    df["elbo_gain"] = df["final_elbo"].diff()  # gain vs previous K
    df["param_cost"] = df["n_params"].diff()  # extra params vs previous K
    df["gain_per_param"] = df["elbo_gain"] / df["param_cost"]

    # Normalised ELBO — comparable across K without BIC's broken N assumption
    df["elbo_per_param"] = df["final_elbo"] / df["n_params"]

    if out_path:
        df.to_csv(out_path, index=False)
        _plot_sweep(df, out_path)

    return df


def _plot_sweep(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Raw ELBO — always increases, shows diminishing returns
    axes[0].plot(df["K"], df["final_elbo"], marker="o", color="#e74c3c")
    axes[0].set_title("Raw ELBO (higher = better fit)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("ELBO")
    axes[0].set_xticks(df["K"])

    # 2. Marginal gain per extra parameter — look for the elbow
    axes[1].plot(df["K"][1:], df["gain_per_param"][1:], marker="o", color="#4a90d9")
    axes[1].axhline(1.0, color="gray", linestyle="--", label="gain = cost threshold")
    axes[1].set_title("ELBO gain per extra parameter\n(elbow = best K)")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("ELBO gain / param cost")
    axes[1].set_xticks(df["K"][1:])
    axes[1].legend()

    # 3. ELBO per parameter — penalises complexity softly
    axes[2].plot(df["K"], df["elbo_per_param"], marker="o", color="#2ecc71")
    best_k = df.loc[df["elbo_per_param"].idxmax(), "K"]
    axes[2].axvline(best_k, color="gray", linestyle="--", label=f"Best K={best_k}")
    axes[2].set_title("ELBO per parameter\n(higher = better)")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("ELBO / n_params")
    axes[2].set_xticks(df["K"])
    axes[2].legend()

    fig.tight_layout()
    plot_path = out_path.parent / "sweep_bic.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved → {plot_path}")


def main(force_retrain: bool = True, run_sweep: bool = False) -> dict:
    device = torch.device("cpu")

    # --- checkpoint or fresh training ---
    ckpt = None if force_retrain else load_checkpoint(CHECKPOINT_PATH)

    if ckpt is not None:
        print("\nSkipping training — using saved checkpoint.")
        gamma = ckpt["gamma"]
        lambda_in = ckpt["lambda_in"]
        lambda_out = ckpt["lambda_out"]
        A = ckpt["A"]
        init_probs = ckpt["init_probs"]
        station_delay = ckpt["station_delay"]
        station_delay_scale = ckpt["station_delay_scale"]
        station_names = ckpt["station_names"]
        time_index = ckpt["time_index"]
        losses = ckpt["losses"]
        obs_scale = ckpt["obs_scale"]
        # meta saved since checkpoint v2 — fall back to CSV if missing
        if "meta" in ckpt:
            meta = ckpt["meta"]
        else:
            print("  meta not found in checkpoint — re-reading coordinates from CSV ...")
            _, _, _, _, _, _, meta, _, _ = load_data(DATA_PATH, device=device)

    else:
        # Fresh training path — load data once
        ins, outs, ins_3d, outs_3d, station_names, time_index, meta, obs_scale, D = \
            load_data(DATA_PATH, device=device)

        # --- K-selection sweep (optional, before training) ---
        if run_sweep:
            print("\nRunning K-selection sweep ...")
            sweep_df = sweep_n_states(
                ins_3d, outs_3d, D,
                candidates=(2, 3, 4, 5, 6),
                out_path=OUT_DIR / "sweep_results.csv",
            )
            print("\nSweep summary:")
            print(sweep_df.to_string(index=False))

            best_k = int(sweep_df.loc[sweep_df["elbo_per_param"].idxmax(), "K"])
            if best_k != NUM_STATES:
                print(f"\nSweep favours K={best_k}, but NUM_STATES={NUM_STATES}.")
            else:
                print(f"\nSweep confirms K={NUM_STATES}. Proceeding.")

            pyro.clear_param_store()  # wipe sweep params before training

        print("\nTraining hierarchical Poisson HMM with bounded station-specific timing offsets ...")
        losses = train(ins_3d, outs_3d, D)

        print("\nRunning forward-backward posterior smoothing ...")
        gamma, lambda_in, lambda_out, A, init_probs, station_delay, station_delay_scale = \
            compute_posterior_probs(ins, outs, time_index, obs_scale)

        print("\nSaving checkpoint ...")
        save_checkpoint(
            losses, obs_scale, station_names, time_index,
            gamma, lambda_in, lambda_out, A, init_probs,
            station_delay, station_delay_scale,
            meta,  # ← persisted so map works on future checkpoint loads
            CHECKPOINT_PATH,
        )

    # --- print diagnostics ---
    S, T, K = gamma.shape
    print(f"\nPosterior Poisson rates per mode (mean across {S} stations, pax/15-min):")
    print(f"  {'Mode':<14} {'lambda_in':>12} {'lambda_out':>12}")
    print("  " + "-" * 42)
    for k, name in enumerate(STATE_NAMES):
        print(f"  {name:<14} {lambda_in[k].mean().item():>12.2f} {lambda_out[k].mean().item():>12.2f}")

    print("\nPosterior transition matrix A[from -> to]:")
    print("              " + "".join(f"{n:>14}" for n in STATE_NAMES))
    for i, name in enumerate(STATE_NAMES):
        print(f"  {name:<12}" + "".join(f"{A[i, j].item():>14.3f}" for j in range(NUM_STATES)))

    print("\nLearned initial distribution for station-day chunks:")
    for name, p in zip(STATE_NAMES, init_probs):
        print(f"  {name:<14} {p.item():.3f}")

    # --- export delays CSV (must come before map plot) ---
    delay_df = export_station_delays(
        station_names, station_delay, station_delay_scale,
        OUT_DIR / "metro_station_mode_delays.csv",
    )
    print("\nModel-based wave summary: morning-rush timing offsets")
    print(
        delay_df[[
            "station",
            "Morning-rush_delay_h",
            "Morning-rush_delay_lo95_h",
            "Morning-rush_delay_hi95_h",
            "Morning-rush_relative_center",
        ]]
        .sort_values("Morning-rush_delay_h")
        .to_string(index=False)
    )

    export_posterior_states(gamma, station_names, time_index,
                            OUT_DIR / "metro_posterior_states.csv")

    # --- timing maps (one HTML per mode) ---
    print("\nGenerating station timing maps ...")
    plot_station_timing_map(
        delay_csv_path=OUT_DIR / "metro_station_mode_delays.csv",
        meta=meta,
        out_dir=OUT_DIR / "plots",
    )

    return {
        "gamma": gamma,
        "lambda_in": lambda_in,
        "lambda_out": lambda_out,
        "A": A,
        "init_probs": init_probs,
        "station_delay": station_delay,
        "station_delay_scale": station_delay_scale,
        "station_names": station_names,
        "time_index": time_index,
        "losses": losses,
        "obs_scale": obs_scale,
    }

if __name__ == "__main__":
    main()
