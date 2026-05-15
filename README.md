# Stochastic Hydrology — Semester Project FS 2026
## Impact of GLOFs on Flood Frequency Statistics of the Simme River

**Course:** Stochastic Hydrology, ETH Zürich (Prof. Passalacqua)  
**Due:** 9 June 2026 | PDF report (max 10 pages) + 10-min recorded presentation  
**Group 4:** Thierry Zehnder, Tim Renggli

---

## Research Question

Do recurring Glacial Lake Outburst Floods (GLOFs) from Plaine Morte glacier systematically
distort flood frequency statistics of the Simme river at Lenk (GRDC station 6935331, 1944–2020)?
Are flood magnitudes observed since 2011 still consistent with the catchment's natural regime?

## Data

| Dataset | Source | File |
|---------|--------|------|
| Daily discharge — Simme at Lenk | GRDC station 6935331 | `data/raw/discharge/` |
| GLOF event dates | gletschersee-lenk.ch, TC 2021 | hardcoded in `utils.py` |

## Folder Structure

```
utils.py                      ← shared functions (GEV fitting, tests, data loading)
notebooks/
  01_data_loading.ipynb       ← raw data QC, annual block maxima, export annual_summary.csv
  02_univariate_gev.ipynb     ← GEV MLE for 3 series, design quantiles, return period plots
  03_homogeneity_tests.ipynb  ← KS tests, Mann-Whitney U, P-P plots
  _archive_03_univariate_ffe.ipynb  ← original monolithic notebook (kept for reference)
data/
  raw/discharge/              ← GRDC raw file (825 KB daily record 1944-2020)
  processed/
    annual_summary.csv        ← annual maxima with GLOF flag (output of notebook 01)
figures/
  01_eda/                     ← discharge time series, annual maxima bar chart
  02_gev/                     ← return period plots, three-curve GEV, quantile comparison
  03_tests/                   ← P-P plots, reference GEV test figures
report/                       ← final PDF report (to be added)
run_pipeline.py               ← standalone data loading script (superseded by notebook 01)
run_gev.py                    ← standalone GEV script (superseded by notebook 02)
```

## Methods (notebook × lecture module mapping)

| Notebook | Methods | Lecture Modules |
|----------|---------|----------------|
| 01 | Block maxima, descriptive stats | Module 2, §4.1 · Module 1, §1.2 |
| 02 | GEV MLE, delta-method CIs, design quantiles, AIC | Module 1, §1.4.2 · Module 2, §4.2.1–4.2.2, Eq. 4.22/4.26/5.12 |
| 03 | PIT, KS GoF test, P–P plot, Mann-Whitney U | Module 1, §1.3.1/§3.1.2/§3.2 |

## Three-Series GEV Comparison

| Series | n | Purpose |
|--------|---|---------|
| Baseline 1944–2010 | 67 | Pre-GLOF reference (homogeneous) |
| Full 1944–2020 | 77 | Includes GLOF-inflated annual maxima |
| GLOF-masked 1944–2020 | 77 | ±5-day window around each GLOF removed before block maxima |

## Confirmed GLOF Events

2011-07-27, 2012-07-24, 2013-08-06, 2014-08-08, 2015-08-03, 2017-07-19, 2018-07-27
(2016 excluded — no documented major outburst)
