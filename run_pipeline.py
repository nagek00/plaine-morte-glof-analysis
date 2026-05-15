"""
Standalone pipeline — discharge loading only.
Generates data/processed/annual_summary.csv and figures/01_discharge_overview.png.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_RAW  = Path("data/raw")
DATA_PROC = Path("data/processed")
FIGS      = Path("figures")
DATA_PROC.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

GLOF_YEARS = {2011, 2018}
DISCHARGE_FILE = DATA_RAW / "discharge/2026-04-26_11-35/6935331_Q_Day.Cmd.txt"

def parse_grdc_discharge(filepath: Path) -> pd.Series:
    df = pd.read_csv(
        filepath,
        sep=";",
        comment="#",
        header=0,
        names=["date", "time", "Q_m3s"],
        parse_dates=["date"],
        na_values=["-999.000", "-999"],
        encoding="latin-1",
    )
    df = df.set_index("date")["Q_m3s"]
    df = pd.to_numeric(df, errors="coerce")
    return df.where(df > 0).sort_index()

discharge_raw = parse_grdc_discharge(DISCHARGE_FILE)
print(f"Discharge: {discharge_raw.index[0].date()} — {discharge_raw.index[-1].date()}")
print(f"  {len(discharge_raw):,} days | NaN={discharge_raw.isna().sum()} | "
      f"min={discharge_raw.min():.3f}  max={discharge_raw.max():.1f}  mean={discharge_raw.mean():.3f} m³/s")

# Annual maxima — use "A" (year-end) for pandas < 2.2 compatibility
Q_max      = discharge_raw.resample("A").max().rename("Q_max")
Q_max_date = discharge_raw.resample("A").apply(lambda s: s.idxmax()).rename("Q_max_date")
Q_max.index      = Q_max.index.year
Q_max_date.index = Q_max_date.index.year

annual = pd.DataFrame({"Q_max": Q_max, "Q_max_date": Q_max_date})
annual.index.name = "year"
annual["is_glof"] = annual.index.isin(GLOF_YEARS)
annual = annual.dropna(subset=["Q_max"])

annual.to_csv(DATA_PROC / "annual_summary.csv")
print(f"\nAnnual table saved: {len(annual)} years, {annual['is_glof'].sum()} GLOF year(s)")
print("\nTop-10 annual maxima:")
print(annual["Q_max"].nlargest(10).to_frame()
      .join(annual[["Q_max_date", "is_glof"]]).to_string())

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
axes[0].plot(discharge_raw.index, discharge_raw.values, lw=0.4, color="steelblue")
axes[0].set_ylabel("Q (m³/s)")
axes[0].set_title("Simme at Lenk (GRDC 6935331) — daily discharge 1944–2020")

glof_mask = annual["is_glof"]
axes[1].bar(annual.index, annual["Q_max"], color="steelblue", alpha=0.65, label="Annual max Q")
axes[1].bar(annual.index[glof_mask], annual.loc[glof_mask, "Q_max"],
            color="crimson", alpha=0.9, label="Known GLOF year")
for yr in annual.index[glof_mask]:
    axes[1].text(yr, annual.loc[yr, "Q_max"] + 0.3, str(yr),
                 ha="center", va="bottom", fontsize=8, color="crimson")
axes[1].set_ylabel("Q_max (m³/s)")
axes[1].set_xlabel("Year")
axes[1].legend()
plt.tight_layout()
fig.savefig(FIGS / "01_discharge_overview.png", dpi=150)
print(f"\nSaved -> {FIGS}/01_discharge_overview.png")
