"""
Shared utility functions for the Plaine Morte GLOF project.
All notebooks import from here to avoid code duplication.
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import inv as mat_inv
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

DISCHARGE_FILE = Path("../data/raw/discharge/2026-04-26_11-35/6935331_Q_Day.Cmd.txt")
DISCHARGE_FILE_2219 = Path("../data/raw/discharge/2219_Abfluss_Tagesmaxima_1974-01-01_2026-05-16.csv")
MONATSMAXIMA_FILE_2219 = Path("../data/raw/2219_Abfluss_Monatsmaxima_1944-01-01_2025-12-31.csv")

# Peak outburst dates derived from lake-level time series (Lac des Faverges data).
# 2011: no lake data — date from press/cantonal archive.
# 2012–2018: lake discharge peak from time_vol_YYYY.csv (data/lake_data/).
# 2016 was previously omitted; lake data confirm a 40 m³/s outburst on Aug 28.
GLOF_DATES = {
    2011: pd.Timestamp("2011-07-27"),  # external sources only
    2012: pd.Timestamp("2012-07-16"),  # corrected from Jul-24; lake peak Jul-16
    2013: pd.Timestamp("2013-08-02"),  # corrected from Aug-06; lake peak Aug-02
    2014: pd.Timestamp("2014-08-07"),  # corrected from Aug-08; lake peak Aug-07
    2015: pd.Timestamp("2015-08-01"),  # corrected from Aug-03; lake peak Aug-01
    2016: pd.Timestamp("2016-08-28"),  # newly added; 40 m³/s lake peak, gauge confirmed
    2017: pd.Timestamp("2017-07-19"),  # confirmed by lake data
    2018: pd.Timestamp("2018-07-27"),  # confirmed by lake data
}

GLOF_WINDOW = 5  # days masked on each side of a GLOF date

# ── Data loading ───────────────────────────────────────────────────────────────

def load_grdc_discharge(filepath: Path = DISCHARGE_FILE) -> pd.Series:
    """Parse GRDC daily discharge file; return Series indexed by date."""
    return (
        pd.read_csv(
            filepath, sep=";", comment="#", header=0,
            names=["date", "time", "Q_m3s"], parse_dates=["date"],
            na_values=["-999.000", "-999"], encoding="latin-1",
        )
        .set_index("date")["Q_m3s"]
        .pipe(pd.to_numeric, errors="coerce")
        .pipe(lambda s: s.where(s > 0))
        .sort_index()
    )


def load_2219_discharge(filepath: Path = DISCHARGE_FILE_2219) -> pd.Series:
    """Parse BAFU/FOEN station 2219 Tagesmaxima CSV; return daily-max Series indexed by date.

    File format: 8 metadata rows, then a semicolon-separated header row, then data.
    Relevant columns: Zeitstempel (date), Wert (daily max discharge m³/s).
    """
    return (
        pd.read_csv(
            filepath, sep=";", skiprows=8, header=0,
            usecols=["Zeitstempel", "Wert"],
            parse_dates=["Zeitstempel"],
            na_values=["-", ""],
            encoding="latin-1",
        )
        .rename(columns={"Zeitstempel": "date", "Wert": "Q_m3s"})
        .set_index("date")["Q_m3s"]
        .pipe(pd.to_numeric, errors="coerce")
        .pipe(lambda s: s.where(s > 0))
        .sort_index()
    )


def load_monatsmaxima(filepath: Path = MONATSMAXIMA_FILE_2219) -> pd.DataFrame:
    """Load BAFU 2219 monthly maxima (1944-2025).

    Returns a DataFrame with columns [date, Q_m3s, year] where `date` is the
    exact timestamp of occurrence of the monthly maximum (Zeitpunkt_des_Auftretens).
    """
    df = (
        pd.read_csv(
            filepath, sep=";", skiprows=8, header=0,
            usecols=["Zeitpunkt_des_Auftretens", "Wert"],
            parse_dates=["Zeitpunkt_des_Auftretens"],
            na_values=["-", ""],
            encoding="latin-1",
        )
        .rename(columns={"Zeitpunkt_des_Auftretens": "date", "Wert": "Q_m3s"})
    )
    df["Q_m3s"] = pd.to_numeric(df["Q_m3s"], errors="coerce").where(lambda s: s > 0)
    df["year"] = df["date"].dt.year
    return df.dropna(subset=["date", "Q_m3s"]).sort_values("date").reset_index(drop=True)


def build_extended_ams(
    tagesmaxima: pd.Series = None,
    monatsmaxima: pd.DataFrame = None,
) -> pd.Series:
    """Build extended BAFU 2219 AMS spanning 1944–2025.

    Uses monthly maxima for 1944–1973 (pre-Tagesmaxima era) and daily maxima for
    1974–2025.  The two sub-series are identical over their overlap period (verified:
    max difference = 0.000 m³/s), so splicing at 1974 introduces no discontinuity.
    """
    if monatsmaxima is None:
        monatsmaxima = load_monatsmaxima()
    if tagesmaxima is None:
        Q = load_2219_discharge()
        tagesmaxima = Q[Q.index.year <= 2025]

    pre = monatsmaxima[monatsmaxima["year"] <= 1973].groupby("year")["Q_m3s"].max()
    post = extract_annual_maxima(tagesmaxima)
    return pd.concat([pre, post]).sort_index()


def extract_annual_maxima(Q_daily: pd.Series) -> pd.Series:
    """Return annual block maxima Series indexed by integer year."""
    Q_ann = Q_daily.resample("YE").max()
    Q_ann.index = Q_ann.index.year
    return Q_ann.dropna()


def mask_glof_windows(
    Q_daily: pd.Series,
    glof_dates: dict = None,
    window: int = GLOF_WINDOW,
) -> pd.Series:
    """Set discharge values within ±window days of each GLOF date to NaN."""
    if glof_dates is None:
        glof_dates = GLOF_DATES
    Q_masked = Q_daily.copy()
    for dt in glof_dates.values():
        Q_masked.loc[
            (Q_masked.index >= dt - pd.Timedelta(days=window))
            & (Q_masked.index <= dt + pd.Timedelta(days=window))
        ] = np.nan
    return Q_masked


# ── Plotting positions ─────────────────────────────────────────────────────────

def gringorten(data: np.ndarray):
    """
    Gringorten plotting positions (Module 1, Table 3.1).
    Returns (sorted_data, return_periods).
    """
    qs = np.sort(data)
    F = (np.arange(1, len(data) + 1) - 0.44) / (len(data) + 0.12)
    return qs, 1.0 / (1.0 - F)


# ── GEV fitting ────────────────────────────────────────────────────────────────

def gev_quantile(T, xi, mu, sigma):
    """GEV quantile for return period T (Module 2, Eq. 4.21)."""
    yT = -np.log(1.0 - 1.0 / T)
    if abs(xi) < 1e-8:
        return mu - sigma * np.log(yT)
    return mu + sigma * (yT ** (-xi) - 1.0) / xi


def _gev_nll(params, data):
    xi, mu, sigma = params
    if sigma <= 0:
        return 1e20
    return -np.sum(stats.genextreme.logpdf(data, c=-xi, loc=mu, scale=sigma))


def _hessian(f, x0, eps=1e-4):
    """Numerical Hessian via central finite differences."""
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return H


def fit_gev(data: np.ndarray, T_arr: np.ndarray, alpha: float = 0.05):
    """
    GEV MLE + delta-method CIs on quantiles (Module 2, §4.2.1, Eq. 4.26).

    scipy convention: c = -xi; we convert back to standard xi.

    Returns
    -------
    xi, mu, sigma : float
        Fitted GEV parameters.
    quantiles : ndarray
        T-year quantile estimates.
    ci_lo, ci_hi : ndarray
        Lower/upper 95 % confidence bounds (delta method).
    """
    c_, mu_, sigma_ = stats.genextreme.fit(data)
    xi_ = -c_
    V = mat_inv(_hessian(lambda p: _gev_nll(p, data), np.array([xi_, mu_, sigma_])))
    z = stats.norm.ppf(1 - alpha / 2)
    qs, lo, hi = [], [], []
    for T in T_arr:
        yT = -np.log(1 - 1 / T)
        q = gev_quantile(T, xi_, mu_, sigma_)
        if abs(xi_) < 1e-8:
            grad = np.array([0.0, 1.0, -np.log(yT)])
        else:
            yTxi = yT ** (-xi_)
            grad = np.array([
                sigma_ * (-(np.log(yT) * yTxi * xi_ + yTxi - 1)) / xi_ ** 2,
                1.0,
                (yTxi - 1) / xi_,
            ])
        se = np.sqrt(max(float(grad @ V @ grad), 0.0))
        qs.append(q); lo.append(q - z * se); hi.append(q + z * se)
    return xi_, mu_, sigma_, np.array(qs), np.array(lo), np.array(hi)


def gev_aic(data: np.ndarray, xi: float, mu: float, sigma: float) -> float:
    """AIC for a fitted GEV (Module 2, §5.5.2, Eq. 5.12)."""
    ll = np.sum(stats.genextreme.logpdf(data, c=-xi, loc=mu, scale=sigma))
    return 2 * 3 - 2 * ll


# ── Statistical tests ──────────────────────────────────────────────────────────

def probability_integral_transform(
    data: np.ndarray, xi: float, mu: float, sigma: float
) -> np.ndarray:
    """Apply reference GEV CDF to observations (Module 1, §1.3.1)."""
    return stats.genextreme.cdf(data, c=-xi, loc=mu, scale=sigma)


def ks_critical_value(n: int, alpha: float = 0.05) -> float:
    """
    KS critical value C_α (Module 1, §3.1.2, Eq. 3.3).

    C_α = K_α / (√n + 0.12 + 0.11/√n)

    K_α values: 0.10 → 1.224, 0.05 → 1.358, 0.01 → 1.628
    Reject H₀ when D_n > C_α.
    """
    K_alpha = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}
    if alpha not in K_alpha:
        raise ValueError(f"alpha must be one of {list(K_alpha.keys())}")
    return K_alpha[alpha] / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))


def ks_test_vs_uniform(u: np.ndarray):
    """
    One-sample KS test of PIT values against Uniform(0,1) (Module 1, §3.1.2).
    Returns (D, p_value).
    """
    return stats.kstest(u, "uniform")


def rank_sum_test(group1: np.ndarray, group2: np.ndarray):
    """
    Rank-Sum Test for two independent samples (Module 1, §2.4.1).
    Also known as Wilcoxon rank-sum test and Mann-Whitney U test.
    H₀: P[X > Y] = 0.5 (both samples from the same population).
    Returns (U_statistic, p_value).
    """
    return stats.mannwhitneyu(group1, group2, alternative="two-sided")


# Keep old name for backward compatibility
def mann_whitney_test(group1: np.ndarray, group2: np.ndarray):
    return rank_sum_test(group1, group2)


def gumbel_mom_params(data: np.ndarray):
    """Gumbel Method of Moments parameters."""
    gamma_EM = 0.5772156649
    sigma = np.std(data, ddof=1) * np.sqrt(6) / np.pi
    mu = np.mean(data) - gamma_EM * sigma
    return mu, sigma


def gumbel_quantile(T, mu, sigma):
    """Gumbel quantile for return period T."""
    return mu - sigma * np.log(-np.log(1 - 1 / T))


def bootstrap_gev_ci(
    data: np.ndarray,
    T_arr: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Bootstrap CI for GEV quantiles (percentile method)."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_q = np.zeros((n_boot, len(T_arr)))
    for i in range(n_boot):
        s = rng.choice(data, size=n, replace=True)
        c_, l_, s_ = stats.genextreme.fit(s)
        boot_q[i] = [gev_quantile(T, -c_, l_, s_) for T in T_arr]
    return np.percentile(boot_q, 100 * alpha / 2, axis=0), np.percentile(
        boot_q, 100 * (1 - alpha / 2), axis=0
    )
