"""
metro_mode_plots.py
===================
All plotting for the metro hierarchical Poisson HMM.

Changes vs previous version
----------------------------
* Off-peak colour changed to green (#27ae60) throughout all plots.
* plot_wave_delays   — bar chart replaced with a two-panel dot / forest plot
                       (Morning-rush left, Evening-rush right) that directly
                       answers the wave question from absolute onset clock times.
* plot_count_rates   — collapsed to a single self-explanatory panel with a
                       context annotation converting pax/min to pax/hour.
* plot_mode_occupancy — units changed from % to hours/day; labels show "X.Xh".
* plot_daily_cycle   — bottom panel replaced: individual station
                       P(Morning-rush|hour) curves + mean ± 1 std band replace
                       the uninformative heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d


# ── Constants ────────────────────────────────────────────────────────────────

STATE_NAMES   = ["Night", "Off-peak", "Morning-rush", "Evening-rush"]

# Overview raster ListedColormap — Night stays very dark there.
STATE_COLORS  = ["#1a1a2e", "#27ae60", "#e74c3c", "#f39c12"]

# All mode-analysis plots (Night slightly lighter for dark-bg readability).
STATE_COLORS_PLOT = ["#3d5a80", "#27ae60", "#e74c3c", "#f39c12"]
#                     Night      Off-peak   Morning    Evening

MODE_CENTER_HOURS = np.array([2.0, 12.0, 7.5, 17.0])
MAX_DELAY_HOURS   = 3.0
STEPS_PER_DAY     = 96   # 15-min intervals per day


# ── Dark-theme palette ───────────────────────────────────────────────────────

BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
MUTED = "#8b949e"
WHITE = "#e6edf3"

def _to_np(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _hour_label(h: float) -> str:
    h = float(h) % 24.0
    m = int(round(h * 60)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _dark(ax: plt.Axes, bg: str = PANEL) -> None:
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)
    ax.grid(color=GRID, linewidth=0.4, linestyle="--", zorder=0)


def _dark_plain(ax: plt.Axes, bg: str = "#161b22") -> None:
    """Minimal dark styling used by the overview figure (no grid)."""
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.tick_params(colors="#aaa", labelsize=7)


def _fig(figsize: Tuple[float, float]) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BG)
    return fig


def _colorbar(im, ax: plt.Axes, label: str = "") -> None:
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label(label, color=MUTED, fontsize=8)
    cbar.outline.set_edgecolor("#30363d")


def _xticks_hours(ax: plt.Axes, step: int = 2) -> None:
    hours = list(range(0, 25, step))
    ax.set_xticks(hours)
    ax.set_xticklabels(
        [_hour_label(h) for h in hours],
        fontsize=7, rotation=30, ha="right",
    )


def _raster_time_ticks(
    ax: plt.Axes,
    time_index: pd.DatetimeIndex,
    T: int,
    n_ticks: int = 12,
) -> None:
    tick_positions = np.linspace(0, T - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [time_index[i].strftime("%b %d\n%H:%M") for i in tick_positions],
        color="#aaa", fontsize=6,
    )


def _daily_gamma(gamma: torch.Tensor) -> np.ndarray:
    """Average posteriors over calendar days → [S, 96, K]."""
    g = _to_np(gamma)
    S, T, K = g.shape
    D = T // STEPS_PER_DAY
    g3d = g[:, : D * STEPS_PER_DAY, :].reshape(S, D, STEPS_PER_DAY, K)
    return g3d.mean(axis=1)


def plot_overview(
    gamma: torch.Tensor,
    lambda_in: torch.Tensor,
    lambda_out: torch.Tensor,
    A: torch.Tensor,
    init_probs: torch.Tensor,
    losses: List[float],
    station_names: List[str],
    time_index: pd.DatetimeIndex,
    station_delay: torch.Tensor,
    station_delay_scale: torch.Tensor,
    out_path: Path,
) -> None:
    """Six-panel HMM training overview."""
    S, T, K = gamma.shape
    bg = "#161b22"

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38,
                          top=0.92, bottom=0.06, left=0.07, right=0.97)

    ax0 = fig.add_subplot(gs[0, 0]); _dark_plain(ax0)
    ax0.plot(losses, color="#4a90d9", lw=1.1)
    ax0.set_title("SVI Training Loss", color="white", fontsize=10)
    ax0.set_xlabel("step", color="#aaa", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1]); _dark_plain(ax1)
    x = np.arange(K)
    ax1.bar(x - 0.2, lambda_in.mean(1).cpu().numpy(),  0.35, label="ins",  color=STATE_COLORS, alpha=0.9)
    ax1.bar(x + 0.2, lambda_out.mean(1).cpu().numpy(), 0.35, label="outs", color=STATE_COLORS, alpha=0.5, hatch="//")
    ax1.set_xticks(x); ax1.set_xticklabels(STATE_NAMES, color="white", fontsize=8)
    ax1.set_title("Posterior Poisson Rates — pax/15min", color="white", fontsize=10)
    ax1.legend(fontsize=7, labelcolor="white", facecolor=bg, edgecolor="#333")

    ax2 = fig.add_subplot(gs[0, 2]); _dark_plain(ax2)
    A_np = _to_np(A)
    im = ax2.imshow(A_np, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(K)); ax2.set_yticks(range(K))
    ax2.set_xticklabels(STATE_NAMES, rotation=30, ha="right", color="white", fontsize=7)
    ax2.set_yticklabels(STATE_NAMES, color="white", fontsize=7)
    ax2.set_title("Posterior Transition Matrix", color="white", fontsize=10)
    for i in range(K):
        for j in range(K):
            ax2.text(j, i, f"{A_np[i,j]:.2f}", ha="center", va="center",
                     color="black" if A_np[i,j] > 0.5 else "white", fontsize=7)
    plt.colorbar(im, ax=ax2)

    ax3 = fig.add_subplot(gs[1, :]); _dark_plain(ax3)
    hard_z = gamma.argmax(-1).cpu().numpy()
    ax3.imshow(hard_z, aspect="auto",
               cmap=ListedColormap(STATE_COLORS), vmin=0, vmax=3, interpolation="nearest")
    ax3.set_yticks(range(S)); ax3.set_yticklabels(station_names, color="white", fontsize=7)
    ax3.set_title("Hard-assigned Hidden State per Station × Time", color="white", fontsize=10)
    ax3.legend(handles=[Patch(facecolor=STATE_COLORS[k], label=STATE_NAMES[k]) for k in range(K)],
               loc="lower right", fontsize=7, facecolor="#0d1117", edgecolor="#333", labelcolor="white")
    _raster_time_ticks(ax3, time_index, T)

    ax4 = fig.add_subplot(gs[2, :2]); _dark_plain(ax4)
    delay_np = _to_np(station_delay)
    scale_np = _to_np(station_delay_scale)
    y = np.arange(S)
    bar_h = 0.18; offsets = [-1.5, -0.5, 0.5, 1.5]
    for k, mode in enumerate(STATE_NAMES):
        yk = y + offsets[k] * bar_h
        ax4.barh(yk, delay_np[:, k], bar_h * 0.9, color=STATE_COLORS[k], alpha=0.80, label=mode)
        ax4.errorbar(delay_np[:, k], yk, xerr=1.96 * scale_np[:, k],
                     fmt="none", ecolor=STATE_COLORS[k], elinewidth=0.7, alpha=0.25, capsize=0)
    ax4.axvline(0, color="#ddd", lw=0.8, alpha=0.5)
    ax4.set_yticks(y); ax4.set_yticklabels(station_names, color="white", fontsize=6)
    ax4.invert_yaxis()
    ax4.set_xlabel("δ[s,mode] in hours (negative = earlier); thin bars ≈ 95% CI",
                   color="#aaa", fontsize=8)
    ax4.set_title("Model-based Wave Pattern — Bounded Station-specific Timing Offsets",
                  color="white", fontsize=10)
    ax4.legend(fontsize=6, labelcolor="white", facecolor=bg, edgecolor="#333")

    ax5 = fig.add_subplot(gs[2, 2]); _dark_plain(ax5)
    one_day = min(STEPS_PER_DAY, T)
    t_slice = time_index[:one_day]
    for k in range(K):
        ax5.plot(t_slice,
                 uniform_filter1d(gamma[0, :one_day, k].cpu().numpy(), size=3),
                 color=STATE_COLORS[k], lw=1.5, label=STATE_NAMES[k])
    ax5.set_title(f"Soft Posteriors — {station_names[0]} day 1", color="white", fontsize=10)
    ax5.set_ylabel("P(state)", color="#aaa", fontsize=8)
    ax5.legend(fontsize=6, labelcolor="white", facecolor=bg, edgecolor="#333")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax5.tick_params(axis="x", labelsize=6, colors="#aaa", rotation=30)

    init_str = ", ".join(f"{n}:{p:.2f}" for n, p in zip(STATE_NAMES, init_probs.cpu().numpy()))
    fig.suptitle(
        "Metro Latent Mode Discovery — Hierarchical Poisson HMM with Bounded Station Timing Offsets\n"
        f"Learned initial distribution: {init_str}",
        color="white", fontsize=13, y=0.975, fontweight="bold",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[0/6] Overview figure        → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY PLOT 1 — Mode onset wave (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mode_onset_wave(
    station_delay: torch.Tensor,
    station_delay_scale: torch.Tensor,
    station_names: List[str],
    out_path: Path,
) -> None:
    """2 × 2 grid — one dot-plot panel per mode, stations ranked by onset time."""
    delay  = _to_np(station_delay)
    dscale = _to_np(station_delay_scale)
    S, K   = delay.shape
    onset_abs = MODE_CENTER_HOURS[None, :] + delay

    fig = _fig((14, max(8, 0.22 * S + 4)))
    fig.suptitle(
        "Mode Onset Wave — Estimated clock time each station enters each operational mode\n"
        "Stations ranked earliest → latest  |  bar width = 95 % CI  "
        "|  dashed line = canonical centre",
        color=WHITE, fontsize=11, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                           top=0.93, bottom=0.06, left=0.14, right=0.97)

    for k, (mode, col) in enumerate(zip(STATE_NAMES, STATE_COLORS_PLOT)):
        ax = fig.add_subplot(gs[k // 2, k % 2]); _dark(ax)
        order        = np.argsort(onset_abs[:, k])
        names_ranked = [station_names[i] for i in order]
        onset_ranked = onset_abs[order, k]
        ci_ranked    = 1.96 * dscale[order, k]
        y = np.arange(S)
        ax.barh(y, ci_ranked * 2, left=onset_ranked - ci_ranked,
                height=0.55, color=col, alpha=0.18, zorder=1)
        ax.scatter(onset_ranked, y, color=col, s=30, zorder=3, linewidths=0)
        center = MODE_CENTER_HOURS[k]
        ax.axvline(center, color=WHITE, lw=0.9, ls="--", alpha=0.55,
                   label=f"Canonical: {_hour_label(center)}")
        spread = onset_ranked.max() - onset_ranked.min()
        ax.text(0.97, 0.03, f"Spread: {spread:.1f} h",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(names_ranked, fontsize=max(5, 8 - S // 10), color=MUTED)
        ax.invert_yaxis()
        pad  = MAX_DELAY_HOURS + 0.5
        xmin = center - pad; xmax = center + pad
        ax.set_xlim(xmin, xmax)
        tick_h = np.arange(np.ceil(xmin), np.floor(xmax) + 1)
        ax.set_xticks(tick_h)
        ax.set_xticklabels([_hour_label(h) for h in tick_h], fontsize=7, rotation=40, ha="right")
        ax.set_xlabel("Estimated onset (clock time)", fontsize=8)
        ax.set_title(mode, color=col, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[1/6] Mode onset wave        → {out_path}")


def plot_count_rates(
    lambda_in: torch.Tensor,
    lambda_out: torch.Tensor,
    station_names: List[str],
    out_path: Path,
    ref_rate: float = 5.0,
) -> None:
    """
    Single self-explanatory panel.

    pax/min = passengers boarding or alighting per minute at a typical station.
    Each bar is annotated with the rate in pax/min AND pax/hour so the number
    is immediately interpretable without domain knowledge.
    Small jittered dots show individual station values behind each mean bar.
    """
    lin  = _to_np(lambda_in)  / 15.0   # [K, S]  pax/15min → pax/min
    lout = _to_np(lambda_out) / 15.0
    K, S = lin.shape
    mean_in  = lin.mean(axis=1)
    mean_out = lout.mean(axis=1)

    fig = _fig((10, 6))
    fig.suptitle(
        "Posterior Count Rates per Mode\n"
        "pax / min  =  passengers boarding or alighting per minute at a typical station",
        color=WHITE, fontsize=12, fontweight="bold",
    )

    ax = fig.add_subplot(1, 1, 1)
    _dark(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.4, linestyle="--", zorder=0)
    ax.grid(axis="x", visible=False)

    x   = np.arange(K)
    w   = 0.34
    rng = np.random.default_rng(0)

    bars_in  = ax.bar(x - w / 2, mean_in,  w, color=STATE_COLORS_PLOT,
                      alpha=0.92, label="Boarding  (λ_in)",   zorder=3)
    bars_out = ax.bar(x + w / 2, mean_out, w, color=STATE_COLORS_PLOT,
                      alpha=0.48, hatch="///", label="Alighting  (λ_out)", zorder=3)

    for k in range(K):
        jit = rng.uniform(-0.11, 0.11, S)
        ax.scatter(k - w / 2 + jit, lin[k],  s=11, color=STATE_COLORS_PLOT[k],
                   alpha=0.30, linewidths=0, zorder=2)
        ax.scatter(k + w / 2 + jit, lout[k], s=11, color=STATE_COLORS_PLOT[k],
                   alpha=0.18, linewidths=0, zorder=2)

    ax.axhline(ref_rate, color=WHITE, lw=1.8, ls="--", alpha=0.80,
               label=f"Benchmark: {ref_rate} pax/min", zorder=4)

    # Annotate bar tops: pax/min  AND  pax/hour
    for bar, val in zip(list(bars_in) + list(bars_out), list(mean_in) + list(mean_out)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.20,
            f"{val:.1f} /min\n({val * 60:.0f} /h)",
            ha="center", va="bottom", fontsize=8, color=WHITE, linespacing=1.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, color=WHITE, fontsize=11, fontweight="bold")
    ax.set_ylabel("pax / min", fontsize=10, color=MUTED)
    ax.legend(fontsize=9, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d",
              loc="upper left")

    # Reading guide box
    rush_k   = STATE_NAMES.index("Morning-rush")
    rush_val = mean_in[rush_k]
    ax.text(
        0.985, 0.97,
        f"Reading guide\n"
        f"────────────────────\n"
        f"Boarding in morning-rush:\n"
        f"  {rush_val:.1f} pax/min\n"
        f"  ≈ {rush_val * 60:.0f} passengers / hour\n"
        f"  per station on average\n\n"
        f"Bars  = system mean\n"
        f"Dots  = individual stations\n"
        f"Label = mean  (pax/min)\n"
        f"              (pax/hour)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=MUTED, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG, edgecolor="#30363d", alpha=0.90),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[2/6] Count rates (pax/min)  → {out_path}")


def plot_mode_occupancy(
    gamma: torch.Tensor,
    station_names: List[str],
    out_path: Path,
) -> None:
    """
    Stacked horizontal bars — hours per day each station spends in each mode.

    Concrete reading: "Nørreport spends 3.6 h/day in morning-rush mode."
    Segments are labelled once they exceed 1 h.
    Stations ranked by morning-rush hours (most at top).
    """
    g   = _to_np(gamma)
    occ = g.mean(axis=1) * 24.0    # [S, K] hours per day
    S, K = occ.shape

    order   = np.argsort(occ[:, STATE_NAMES.index("Morning-rush")])[::-1]
    occ_s   = occ[order]
    names_s = [station_names[i] for i in order]

    fig = _fig((11, max(6, 0.28 * S + 2.5)))
    ax  = fig.add_subplot(1, 1, 1)
    _dark(ax); ax.grid(False)

    y    = np.arange(S)
    left = np.zeros(S)

    for k in range(K):
        col = STATE_COLORS_PLOT[k]
        ax.barh(y, occ_s[:, k], left=left, height=0.72,
                color=col, alpha=0.88, label=STATE_NAMES[k])
        for i in range(S):
            hrs = occ_s[i, k]
            if hrs >= 1.0:
                ax.text(
                    left[i] + hrs / 2, y[i],
                    f"{hrs:.1f}h",
                    ha="center", va="center",
                    fontsize=max(5, 7 - S // 20),
                    color=WHITE, fontweight="bold",
                )
        left += occ_s[:, k]

    ax.axvline(24, color=WHITE, lw=0.6, ls=":", alpha=0.20)

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=max(5, 8 - S // 10), color=MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xticklabels(["0 h", "4 h", "8 h", "12 h", "16 h", "20 h", "24 h"],
                       fontsize=8, color=MUTED)
    ax.set_xlabel("Hours per day spent in each mode  (total = 24 h)", fontsize=9)
    ax.set_title(
        "Mode Occupancy — hours per day per station\n"
        "Ranked by morning-rush hours (highest at top)",
        color=WHITE, fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8, labelcolor=WHITE,
              facecolor=PANEL, edgecolor="#30363d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[3/6] Mode occupancy         → {out_path}")

def plot_daily_cycle(
    gamma: torch.Tensor,
    station_names: List[str],
    out_path: Path,
) -> None:
    """
    Two-panel figure.

    Top    : Mean P(mode | quarter-hour) across all stations & days.

    Bottom : P(Morning-rush | hour) per individual station (thin transparent
             lines) with mean curve and ± 1 std band overlaid.
             Narrow band → all stations peak simultaneously.
             Wide band   → stations peak at different times (wave pattern).
             The annotation states the std of peak hours across stations.
    """
    g_day   = _daily_gamma(gamma)     # [S, 96, K]
    S, _, K = g_day.shape
    hours   = np.arange(STEPS_PER_DAY) / 4.0
    rush_k  = STATE_NAMES.index("Morning-rush")
    rush_col = STATE_COLORS_PLOT[rush_k]

    fig = _fig((14, 9))
    fig.suptitle(
        "Average Daily Mode Cycle — system-wide rhythm and per-station morning-rush variation",
        color=WHITE, fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.1],
                           hspace=0.42, top=0.93, bottom=0.08,
                           left=0.09, right=0.97)

    # ── Top: system-wide mean ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0]); _dark(ax0)
    mean_p = g_day.mean(axis=0)   # [96, K]

    for k in range(K):
        col = STATE_COLORS_PLOT[k]
        sm  = uniform_filter1d(mean_p[:, k], size=4)
        ax0.plot(hours, sm, color=col, lw=2.2, label=STATE_NAMES[k])
        ax0.fill_between(hours, 0, sm, color=col, alpha=0.09)

    for k in range(K):
        ax0.axvline(MODE_CENTER_HOURS[k], color=STATE_COLORS_PLOT[k],
                    lw=0.7, ls=":", alpha=0.55)
        ax0.text(MODE_CENTER_HOURS[k], 1.01, _hour_label(MODE_CENTER_HOURS[k]),
                 ha="center", va="bottom", fontsize=6.5, color=STATE_COLORS_PLOT[k],
                 transform=ax0.get_xaxis_transform())

    ax0.set_xlim(0, 24); _xticks_hours(ax0, step=2)
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("P(mode)", fontsize=9)
    ax0.set_title("System-wide average  —  all stations, all days", color=WHITE, fontsize=10)
    ax0.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d", ncol=4)

    # ── Bottom: individual curves + mean ± std band ──────────────────────────
    ax1 = fig.add_subplot(gs[1]); _dark(ax1)
    rush_mat = g_day[:, :, rush_k]   # [S, 96]
    mean_r   = rush_mat.mean(axis=0)
    std_r    = rush_mat.std(axis=0)
    mean_sm  = uniform_filter1d(mean_r, size=4)
    std_sm   = uniform_filter1d(std_r,  size=4)

    # Individual station lines
    for s in range(S):
        ax1.plot(hours, uniform_filter1d(rush_mat[s], size=4),
                 color=rush_col, lw=0.55, alpha=0.12, zorder=1)

    # ± 1 std band
    ax1.fill_between(hours, mean_sm - std_sm, mean_sm + std_sm,
                     color=rush_col, alpha=0.22, zorder=2,
                     label="Mean ± 1 std  (station spread)")

    # Mean curve
    ax1.plot(hours, mean_sm, color=rush_col, lw=2.5, zorder=3, label="System mean")
    ax1.plot(hours, mean_sm + std_sm, color=rush_col, lw=0.9, ls="--", alpha=0.50, zorder=2)
    ax1.plot(hours, mean_sm - std_sm, color=rush_col, lw=0.9, ls="--", alpha=0.50, zorder=2)

    # Peak annotation
    peak_h = hours[np.argmax(mean_sm)]
    ax1.axvline(peak_h, color=WHITE, lw=0.9, ls="--", alpha=0.55)
    ax1.text(peak_h + 0.15, 0.98, f"Mean peak: {_hour_label(peak_h)}",
             transform=ax1.get_xaxis_transform(),
             va="top", ha="left", fontsize=8, color=WHITE)

    # Wave-spread annotation
    peak_hours_per_station = hours[rush_mat.argmax(axis=1)]
    peak_std = peak_hours_per_station.std()
    wave_txt = "→ wave detected" if peak_std > 0.4 else "→ stations synchronised"
    ax1.text(
        0.985, 0.97,
        f"Peak hour std across stations: {peak_std:.2f} h\n{wave_txt}",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=9, color=rush_col,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor="#30363d", alpha=0.88),
    )

    ax1.set_xlim(0, 24); _xticks_hours(ax1, step=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(Morning-rush)", fontsize=9)
    ax1.set_xlabel("Hour of day", fontsize=9)
    ax1.set_title(
        "P(Morning-rush | hour) per station — thin lines = individual stations\n"
        "Narrow band → all stations peak together  |  "
        "Wide band → stations peak at different times (wave)",
        color=WHITE, fontsize=9,
    )
    ax1.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[4/6] Daily cycle            → {out_path}")


def plot_weekday_weekend(
    gamma: torch.Tensor,
    station_names: List[str],
    time_index: pd.DatetimeIndex,
    out_path: Path,
) -> None:
    """2 × 2 grid — solid = weekday, dashed = weekend per mode."""
    g = _to_np(gamma)
    S, T, K = g.shape
    ti  = time_index[:T]
    dow = ti.dayofweek.values
    is_we = (dow >= 5)
    qh    = (ti.hour * 4 + ti.minute // 15).values

    g_mean_s = g.mean(axis=0)
    wd_sum = np.zeros((STEPS_PER_DAY, K)); we_sum = np.zeros((STEPS_PER_DAY, K))
    wd_cnt = np.zeros(STEPS_PER_DAY);     we_cnt = np.zeros(STEPS_PER_DAY)

    for t in range(T):
        q = qh[t]
        if is_we[t]:
            we_sum[q] += g_mean_s[t]; we_cnt[q] += 1
        else:
            wd_sum[q] += g_mean_s[t]; wd_cnt[q] += 1

    wd_p  = wd_sum / np.maximum(wd_cnt, 1)[:, None]
    we_p  = we_sum / np.maximum(we_cnt, 1)[:, None]
    hours = np.arange(STEPS_PER_DAY) / 4.0

    fig = _fig((14, 9))
    fig.suptitle(
        "Weekday vs Weekend Mode Cycle  —  all-station average P(mode | hour-of-day)\n"
        "Solid = Mon–Fri  |  Dashed = Sat–Sun  |  Shaded = difference region",
        color=WHITE, fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.91, bottom=0.07, left=0.08, right=0.97)

    for k in range(K):
        ax  = fig.add_subplot(gs[k // 2, k % 2]); _dark(ax)
        col = STATE_COLORS_PLOT[k]
        wd_sm = uniform_filter1d(wd_p[:, k], size=4)
        we_sm = uniform_filter1d(we_p[:, k], size=4)
        ax.plot(hours, wd_sm, color=col, lw=2.0, label="Weekday  (Mon–Fri)")
        ax.plot(hours, we_sm, color=col, lw=2.0, ls="--", alpha=0.65, label="Weekend  (Sat–Sun)")
        ax.fill_between(hours, wd_sm, we_sm, color=col, alpha=0.14)
        wd_peak = hours[np.argmax(wd_sm)]; we_peak = hours[np.argmax(we_sm)]
        ax.axvline(wd_peak, color=col, lw=0.7, ls=":", alpha=0.5)
        ax.axvline(we_peak, color=col, lw=0.7, ls=":", alpha=0.35)
        ax.text(0.97, 0.95,
                f"WD peak: {_hour_label(wd_peak)}\nWE peak: {_hour_label(we_peak)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        ax.axvline(MODE_CENTER_HOURS[k], color=WHITE, lw=0.6, ls=":", alpha=0.35)
        ax.set_xlim(0, 24); _xticks_hours(ax, step=4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(mode)" if k % 2 == 0 else "", fontsize=9)
        ax.set_title(STATE_NAMES[k], color=col, fontsize=11, fontweight="bold")
        if k == 0:
            ax.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[5/6] Weekday vs weekend     → {out_path}")


def plot_wave_delays(
    station_delay: torch.Tensor,
    station_delay_scale: torch.Tensor,
    station_names: List[str],
    out_path: Path,
) -> None:
    """
    Two-panel dot / forest plot — Morning-rush (left) and Evening-rush (right).

    X-axis : absolute estimated onset clock time
             = MODE_CENTER_HOURS[k] + δ[s, k]
    Y-axis : stations; BOTH panels sorted by morning-rush onset (earliest at top)
             so you can compare the two modes side-by-side with a fixed reference.
    Marker : filled dot = point estimate
    Line   : horizontal whisker = 95 % CI

    How to read
    -----------
    Dots stacked in a near-vertical column → all stations enter that mode at
    the same time (no wave).  Dots fanning diagonally top-left to bottom-right
    → temporal wave; early stations at the top, late stations at the bottom.
    The "Spread" annotation quantifies the gap in hours between earliest and latest.
    """
    delay  = _to_np(station_delay)
    dscale = _to_np(station_delay_scale)
    S, K   = delay.shape

    onset = MODE_CENTER_HOURS[None, :] + delay   # [S, K] absolute clock hours
    ci    = 1.96 * dscale

    rush_k = STATE_NAMES.index("Morning-rush")
    eve_k  = STATE_NAMES.index("Evening-rush")

    # Sort by morning-rush onset — keep same order for both panels
    order   = np.argsort(onset[:, rush_k])
    onset_s = onset[order]
    ci_s    = ci[order]
    names_s = [station_names[i] for i in order]

    fig = _fig((14, max(8, 0.22 * S + 4)))
    fig.suptitle(
        "Station Mode Onset Times — Is there a temporal wave?\n"
        "Both panels sorted by morning-rush onset (earliest at top)  "
        "|  whiskers = 95 % CI  |  dashed line = canonical centre",
        color=WHITE, fontsize=11, fontweight="bold", y=0.997,
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                           left=0.14, right=0.97, top=0.92, bottom=0.05)
    y = np.arange(S)

    for panel_i, (mode_k, mode_name) in enumerate(
        [(rush_k, "Morning-rush"), (eve_k, "Evening-rush")]
    ):
        ax  = fig.add_subplot(gs[0, panel_i]); _dark(ax)
        col = STATE_COLORS_PLOT[mode_k]

        lo = onset_s[:, mode_k] - ci_s[:, mode_k]
        hi = onset_s[:, mode_k] + ci_s[:, mode_k]

        # 95 % CI whiskers (horizontal lines, NOT filled bars)
        ax.hlines(y, lo, hi, color=col, linewidth=1.0, alpha=0.35, zorder=1)

        # Point estimate dots
        ax.scatter(onset_s[:, mode_k], y, color=col, s=40,
                   zorder=3, linewidths=0, label=mode_name)

        # Canonical centre
        center = MODE_CENTER_HOURS[mode_k]
        ax.axvline(center, color=WHITE, lw=1.1, ls="--", alpha=0.60,
                   label=f"Canonical: {_hour_label(center)}")

        # X-axis window around canonical centre
        spread = onset_s[:, mode_k].max() - onset_s[:, mode_k].min()
        pad    = max(MAX_DELAY_HOURS + 0.4, spread * 0.55 + 0.5)
        xmin   = center - pad; xmax = center + pad
        ax.set_xlim(xmin, xmax)
        tick_h = np.arange(np.ceil(xmin), np.floor(xmax) + 1)
        ax.set_xticks(tick_h)
        ax.set_xticklabels([_hour_label(h) for h in tick_h],
                           fontsize=7, rotation=40, ha="right")
        ax.set_xlabel("Estimated onset (clock time)", fontsize=9)

        # Y-axis: station names on left panel only
        ax.set_yticks(y)
        if panel_i == 0:
            ax.set_yticklabels(names_s, fontsize=max(5, 8 - S // 10), color=MUTED)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()

        ax.set_title(mode_name, color=col, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#30363d")

        # Spread / wave annotation
        wave = spread > 0.5
        ax.text(
            0.97, 0.015,
            f"Spread: {spread:.2f} h\n"
            f"{'▶ Wave detected' if wave else '◉ Stations synchronised'}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=col if wave else MUTED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                      edgecolor=col if wave else "#30363d", alpha=0.90),
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[6/6] Wave delays            → {out_path}")

def plot_all(
    gamma: torch.Tensor,
    lambda_in: torch.Tensor,
    lambda_out: torch.Tensor,
    station_names: List[str],
    time_index: pd.DatetimeIndex,
    station_delay: torch.Tensor,
    station_delay_scale: torch.Tensor,
    losses: Optional[List[float]] = None,
    A: Optional[torch.Tensor] = None,
    init_probs: Optional[torch.Tensor] = None,
    out_dir: Path = Path("."),
    ref_rate: float = 5.0,
) -> None:
    """
    Generate all plots and save to *out_dir*.

    metro_hmm_results.png      — [0/6] overview  (requires losses, A, init_probs)
    metro_mode_onset_wave.png  — [1/6] onset clock time per station/mode (2×2 grid)
    metro_count_rates.png      — [2/6] pax/min per mode, single annotated panel
    metro_mode_occupancy.png   — [3/6] hours/day per mode per station
    metro_daily_cycle.png      — [4/6] system mean + individual station curves
    metro_weekday_weekend.png  — [5/6] weekday vs weekend comparison
    metro_wave_delays.png      — [6/6] dot-plot onset times (morning + evening rush)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}/\n")

    if losses is not None and A is not None and init_probs is not None:
        plot_overview(
            gamma, lambda_in, lambda_out,
            A, init_probs, losses,
            station_names, time_index,
            station_delay, station_delay_scale,
            out_dir / "metro_hmm_results.png",
        )
    plot_mode_onset_wave(
        station_delay, station_delay_scale, station_names,
        out_dir / "metro_mode_onset_wave.png",
    )
    plot_count_rates(
        lambda_in, lambda_out, station_names,
        out_dir / "metro_count_rates.png",
        ref_rate=ref_rate,
    )
    plot_mode_occupancy(
        gamma, station_names,
        out_dir / "metro_mode_occupancy.png",
    )
    plot_daily_cycle(
        gamma, station_names,
        out_dir / "metro_daily_cycle.png",
    )
    plot_weekday_weekend(
        gamma, station_names, time_index,
        out_dir / "metro_weekday_weekend.png",
    )
    plot_wave_delays(
        station_delay, station_delay_scale, station_names,
        out_dir / "metro_wave_delays.png",
    )
    print(f"\nDone.  All plots saved to {out_dir}/")

def plot_station_timing_map(
        delay_csv_path: Path,
        meta: pd.DataFrame,
        out_dir: Path,
        modes: list[str] | None = None,
) -> None:
    try:
        import folium
    except ImportError:
        print("Install folium first:  pip install folium")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    coords = (
        meta[["station", "lat", "lon"]]
        .drop_duplicates("station")
        .dropna(subset=["lat", "lon"])
        .set_index("station")
    )

    df = pd.read_csv(delay_csv_path)
    modes = modes or STATE_NAMES
    map_center = [coords["lat"].mean(), coords["lon"].mean()]

    for mode in modes:
        delay_col  = f"{mode}_delay_h"
        center_col = f"{mode}_relative_center"

        if delay_col not in df.columns:
            print(f"  Skipping {mode} — column {delay_col} not found")
            continue

        df_mode = df[["station", delay_col, center_col]].dropna()
        df_mode = df_mode[df_mode["station"].isin(coords.index)].copy()

        if df_mode.empty:
            print(f"  Skipping {mode} — no coordinate matches")
            continue

        def _time_str_to_minutes(t: str) -> int:
            h, m = map(int, str(t).strip().split(":"))
            return h * 60 + m

        df_mode["entry_minutes"] = df_mode[center_col].apply(_time_str_to_minutes)

        # Snap to 15-minute window: 07:17 → bin 29 (floor(437/15)=29 → 07:15)
        df_mode["bin_15"]        = df_mode["entry_minutes"] // 15
        df_mode["bin_label"]     = df_mode["bin_15"].apply(
            lambda b: f"{(b * 15) // 60:02d}:{(b * 15) % 60:02d}"
            f"–{((b + 1) * 15) // 60:02d}:{((b + 1) * 15) % 60:02d}"
        )

        unique_bins  = sorted(df_mode["bin_15"].unique())
        n_bins       = len(unique_bins)

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        cmap   = plt.cm.RdYlGn_r                        # green=early, red=late
        bin_to_color = {
            b: mcolors.to_hex(cmap(i / max(n_bins - 1, 1)))
            for i, b in enumerate(unique_bins)
        }
        df_mode["color"] = df_mode["bin_15"].map(bin_to_color)

        # --- Build map ---
        fmap = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")

        # Legend — one entry per 15-min window that has at least one station
        legend_rows = "".join(
            f'<div style="display:flex;align-items:center;margin-bottom:3px">'
            f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;'
            f'background:{bin_to_color[b]};margin-right:7px;flex-shrink:0"></span>'
            f'<span>{df_mode[df_mode["bin_15"]==b]["bin_label"].iloc[0]}</span></div>'
            for b in unique_bins
        )
        legend_html = f"""
        <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                    background:white; padding:14px 18px; border-radius:10px;
                    box-shadow:2px 2px 8px rgba(0,0,0,0.25);
                    font-family:sans-serif; font-size:12px; max-height:400px; overflow-y:auto;">
          <b style="font-size:13px">{mode}<br>Entry time (15-min window)</b>
          <div style="margin-top:8px">{legend_rows}</div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

        # Plot markers
        for _, row in df_mode.iterrows():
            lat = coords.loc[row["station"], "lat"]
            lon = coords.loc[row["station"], "lon"]

            folium.CircleMarker(
                location=[lat, lon],
                radius=11,
                color="white",
                weight=2,
                fill=True,
                fill_color=row["color"],
                fill_opacity=0.9,
                tooltip=folium.Tooltip(
                    f"<b>{row['station']}</b><br>"
                    f"Enters {mode} at ~<b>{row[center_col]}</b><br>"
                    f"Window: {row['bin_label']}<br>"
                    f"Offset: {row[delay_col]:+.2f}h vs average"
                ),
            ).add_to(fmap)

        safe_mode = mode.lower().replace("-", "_").replace(" ", "_")
        out_path  = out_dir / f"map_{safe_mode}_timing.html"
        fmap.save(str(out_path))
        print(f"  [{mode}] map saved → {out_path}  ({n_bins} distinct 15-min windows)")

    print(f"\nAll timing maps saved to {out_dir}")