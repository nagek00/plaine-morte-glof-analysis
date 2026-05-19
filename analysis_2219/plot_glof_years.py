"""
plot_glof_years.py
------------------
For each GLOF year, plots the summer daily discharge (May-Oct) from BAFU 2219,
highlighting:
  - red shaded band : ±5-day masking window around the outburst
  - red dashed line : GLOF date
  - red star        : discharge ON the outburst date
  - black diamond   : annual maximum (date + value)
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

from utils import load_2219_discharge, GLOF_DATES, GLOF_WINDOW, extract_annual_maxima

# ── Load data ─────────────────────────────────────────────────────────────────
Q_raw = load_2219_discharge(Path('../data/raw/discharge/2219_Abfluss_Tagesmaxima_1974-01-01_2026-05-16.csv'))
Q = Q_raw[Q_raw.index.year <= 2025]

FIGS = Path('../figures_2219/01_eda')
FIGS.mkdir(parents=True, exist_ok=True)

# ── Build annual maxima lookup ─────────────────────────────────────────────────
ann_max_date = {}
ann_max_val  = {}
for yr in GLOF_DATES:
    yr_data = Q[Q.index.year == yr]
    if len(yr_data) == 0:
        continue
    ann_max_date[yr] = yr_data.idxmax()
    ann_max_val[yr]  = yr_data.max()

# ── Plot ───────────────────────────────────────────────────────────────────────
glof_years = sorted(GLOF_DATES.keys())
n = len(glof_years)
ncols = 2
nrows = (n + 1) // ncols   # 4 rows for 7 years (last row has 1 panel)

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
axes_flat = axes.flatten()

for ax, yr in zip(axes_flat, glof_years):
    glof_dt = GLOF_DATES[yr]
    max_dt  = ann_max_date[yr]
    max_val = ann_max_val[yr]
    q_glof  = Q.get(glof_dt, np.nan)

    # Summer window: May 1 – Oct 15
    start = pd.Timestamp(f'{yr}-05-01')
    end   = pd.Timestamp(f'{yr}-10-15')
    Q_yr  = Q[(Q.index >= start) & (Q.index <= end)]

    # ±5-day masking band
    band_lo = glof_dt - pd.Timedelta(days=GLOF_WINDOW)
    band_hi = glof_dt + pd.Timedelta(days=GLOF_WINDOW)

    ax.fill_betweenx(
        [0, max_val * 1.25],
        band_lo, band_hi,
        color='crimson', alpha=0.12, zorder=0,
        label=f'±{GLOF_WINDOW}-day window'
    )

    # Discharge time series
    ax.plot(Q_yr.index, Q_yr.values, color='steelblue', lw=1.2, zorder=2)

    # GLOF date — dashed vertical line
    ax.axvline(glof_dt, color='crimson', lw=1.5, ls='--', zorder=3, label=f'GLOF {glof_dt.date()}')

    # Discharge on GLOF date — red star
    ax.scatter(glof_dt, q_glof, color='crimson', marker='*', s=180, zorder=5,
               label=f'Q outburst = {q_glof:.1f} m³/s')

    # Annual maximum — black diamond
    ax.scatter(max_dt, max_val, color='black', marker='D', s=70, zorder=5,
               label=f'Annual max = {max_val:.1f} m³/s ({max_dt.strftime("%d %b")})')

    ax.set_title(f'{yr}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Q (m³/s)', fontsize=8)
    ax.set_ylim(bottom=0, top=max_val * 1.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', labelsize=7.5)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc='upper right')

# Hide unused axes (last subplot if n is odd)
for ax in axes_flat[n:]:
    ax.set_visible(False)

fig.suptitle(
    'Simme at Lenk (BAFU 2219) — GLOF years: summer discharge\n'
    'Red dashed = outburst date  |  ★ = Q on outburst day  |  ◆ = annual maximum  |  shading = ±5-day window',
    fontsize=10, y=1.01
)
plt.tight_layout()

out = FIGS / '01_glof_years_detail.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved -> {out}')
