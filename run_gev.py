"""Run notebook 03 analysis as a plain script to verify correctness."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DATA_PROC = Path("data/processed")
FIGS      = Path("figures")
FIGS.mkdir(exist_ok=True)

annual = pd.read_csv(DATA_PROC / "annual_summary.csv", index_col="year")
Q = annual["Q_max"].dropna().values
n = len(Q)

# ── Gumbel MoM ────────────────────────────────────────────────────────────────
gamma_EM  = 0.5772156649
sigma_mom = np.std(Q, ddof=1) * np.sqrt(6) / np.pi
mu_mom    = np.mean(Q) - gamma_EM * sigma_mom

# ── Gumbel MLE ────────────────────────────────────────────────────────────────
loc_mle, scale_mle = stats.gumbel_r.fit(Q)

# ── GEV MLE ───────────────────────────────────────────────────────────────────
c_mle, loc_gev, scale_gev = stats.genextreme.fit(Q)
xi_mle = -c_mle

# ── AIC ───────────────────────────────────────────────────────────────────────
ll_gumbel_mom = np.sum(stats.gumbel_r.logpdf(Q, loc=mu_mom,   scale=sigma_mom))
ll_gumbel_mle = np.sum(stats.gumbel_r.logpdf(Q, loc=loc_mle,  scale=scale_mle))
ll_gev_mle    = np.sum(stats.genextreme.logpdf(Q, c=c_mle, loc=loc_gev, scale=scale_gev))

print("=== Parameter estimates ===")
print(f"Gumbel MoM : mu={mu_mom:.4f}, sigma={sigma_mom:.4f}")
print(f"Gumbel MLE : mu={loc_mle:.4f}, sigma={scale_mle:.4f}")
type_label = ("Frechet/Type II (heavy tail)" if xi_mle > 0.05
              else "Weibull/Type III (bounded)" if xi_mle < -0.05
              else "near-Gumbel/Type I")
print(f"GEV    MLE : xi={xi_mle:.4f} ({type_label}), mu={loc_gev:.4f}, sigma={scale_gev:.4f}")

print("\n=== AIC (lower = better) ===")
print(f"Gumbel MoM : {2*2 - 2*ll_gumbel_mom:.2f}")
print(f"Gumbel MLE : {2*2 - 2*ll_gumbel_mle:.2f}")
print(f"GEV    MLE : {2*3 - 2*ll_gev_mle:.2f}  (dAIC vs Gumbel MLE: {(2*3-2*ll_gev_mle)-(2*2-2*ll_gumbel_mle):+.2f})")

# ── Return period curves ──────────────────────────────────────────────────────
T_plot = np.logspace(np.log10(1.01), np.log10(500), 300)

def gev_q(T, xi, mu, sigma):
    p = 1 - 1/T
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))
    return mu + sigma * ((-np.log(p))**(-xi) - 1) / xi

def gumbel_q(T, mu, sigma):
    return mu - sigma * np.log(-np.log(1 - 1/T))

# Bootstrap CI
rng = np.random.default_rng(42)
boot_q = np.zeros((2000, len(T_plot)))
for i in range(2000):
    s = rng.choice(Q, size=n, replace=True)
    cb, lb, sb = stats.genextreme.fit(s)
    boot_q[i] = [gev_q(t, -cb, lb, sb) for t in T_plot]
ci_lo = np.percentile(boot_q, 2.5,  axis=0)
ci_hi = np.percentile(boot_q, 97.5, axis=0)

# Empirical (Gringorten)
Q_sorted = np.sort(Q)
ranks    = np.arange(1, n+1)
F_gring  = (ranks - 0.44) / (n + 0.12)
T_emp    = 1 / (1 - F_gring)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: return period
ax = axes[0]
ax.semilogx(T_emp, Q_sorted, "ko", ms=4, zorder=5, label="Observed (Gringorten)")
ax.semilogx(T_plot, [gumbel_q(t, mu_mom, sigma_mom) for t in T_plot],
            "b--", lw=1.5, label="Gumbel MoM")
ax.semilogx(T_plot, [gumbel_q(t, loc_mle, scale_mle) for t in T_plot],
            "b-",  lw=1.8, label="Gumbel MLE")
ax.semilogx(T_plot, [gev_q(t, xi_mle, loc_gev, scale_gev) for t in T_plot],
            "r-",  lw=2.0, label=f"GEV MLE (xi={xi_mle:.3f})")
ax.fill_between(T_plot, ci_lo, ci_hi, color="red", alpha=0.15, label="GEV 95% CI")
for yr in annual.index[annual["is_glof"]]:
    q_yr = annual.loc[yr, "Q_max"]
    ax.axhline(q_yr, color="gray", lw=0.6, ls=":")
    ax.text(1.05, q_yr + 0.15, str(yr), fontsize=7, color="darkred")
ax.set_xlabel("Return period T (years)")
ax.set_ylabel("Q_max (m3/s)")
ax.set_title("Return period curves")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# Panel B: Gumbel probability plot
y_gring = -np.log(-np.log(F_gring))
ax2 = axes[1]
ax2.scatter(y_gring, Q_sorted, s=18, color="steelblue", zorder=4, label="Data")
y_range = np.linspace(y_gring.min()-0.5, y_gring.max()+1.5, 100)
ax2.plot(y_range, loc_mle + scale_mle * y_range, "b-",  lw=1.5, label="Gumbel MLE")
ax2.plot(y_range, mu_mom  + sigma_mom  * y_range, "b--", lw=1.2, label="Gumbel MoM")
T_range = 1 / (1 - (1 - np.exp(-np.exp(-y_range))))
q_gev_range = np.array([gev_q(t, xi_mle, loc_gev, scale_gev) for t in T_range])
ax2.plot(y_range, q_gev_range, "r-", lw=1.8, label=f"GEV MLE")
for yr in annual.index[annual["is_glof"]]:
    q_yr  = annual.loc[yr, "Q_max"]
    idx_q = np.searchsorted(Q_sorted, q_yr)
    y_yr  = y_gring[min(idx_q, len(y_gring)-1)]
    ax2.scatter(y_yr, q_yr, s=60, color="crimson", zorder=5)
    ax2.annotate(str(yr), (y_yr, q_yr), xytext=(4, 3),
                 textcoords="offset points", fontsize=7, color="crimson")
ax2.set_xlabel("Reduced Gumbel variate y")
ax2.set_ylabel("Q_max (m3/s)")
ax2.set_title("Gumbel probability plot")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIGS / "03_gev_analysis.png", dpi=150)
print("\nFigure saved -> figures/03_gev_analysis.png")

# ── Design quantiles ──────────────────────────────────────────────────────────
print("\nDesign quantiles (m3/s):")
print(f"{'T':>6}  {'Gumbel_MoM':>11}  {'Gumbel_MLE':>11}  {'GEV_MLE':>9}  {'CI_lo':>8}  {'CI_hi':>8}")
for T in [2, 5, 10, 20, 50, 100, 200]:
    idx = np.argmin(np.abs(T_plot - T))
    print(f"{T:>6}  {gumbel_q(T, mu_mom, sigma_mom):>11.2f}  "
          f"{gumbel_q(T, loc_mle, scale_mle):>11.2f}  "
          f"{gev_q(T, xi_mle, loc_gev, scale_gev):>9.2f}  "
          f"{ci_lo[idx]:>8.2f}  {ci_hi[idx]:>8.2f}")
