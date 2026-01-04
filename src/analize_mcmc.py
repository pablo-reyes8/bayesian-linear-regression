import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_beta_traces(
    beta_post,
    max_plots=None,
    ncols=4,
    figsize=None,
    sharex=True,
    burnin=0,
    thin=1,
    linewidth=0.5,
    suptitle=None,
    show=True,
    savepath=None,
    dpi=200,
):
    """
    Plot trace plots for posterior draws of beta coefficients.

    Parameters
    ----------
    beta_post : array-like, shape (n_draws, n_coefs)
        Posterior draws for beta.
    max_plots : int or None
        Max number of coefficients to plot (useful if many coefs). If None, plot all.
    ncols : int
        Number of columns in the subplot grid.
    figsize : tuple or None
        Figure size. If None, chosen automatically.
    sharex : bool
        Share x-axis across subplots.
    burnin : int
        Number of initial draws to drop.
    thin : int
        Keep every `thin`-th draw after burn-in.
    linewidth : float
        Line width for trace plot.
    suptitle : str or None
        Figure-level title.
    show : bool
        Whether to call plt.show().
    savepath : str or None
        If provided, save the figure to this path.
    dpi : int
        DPI for saving.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, n_coefs). Got shape={beta_post.shape}")

    n_draws, n_coefs = beta_post.shape

    if burnin < 0 or burnin >= n_draws:
        raise ValueError(f"burnin must be in [0, {n_draws-1}]. Got burnin={burnin}")
    if thin <= 0:
        raise ValueError("thin must be a positive integer.")

    draws = beta_post[burnin::thin, :]
    n_to_plot = n_coefs if max_plots is None else min(int(max_plots), n_coefs)

    nrows = math.ceil(n_to_plot / ncols)
    if figsize is None:
        # sensible default scaling
        figsize = (4.0 * ncols, 2.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex)
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_to_plot):
        ax = axes[i]
        ax.plot(draws[:, i], linewidth=linewidth)
        ax.set_title(f"β[{i}] Trace")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"β[{i}]")

    # turn off unused axes
    for j in range(n_to_plot, len(axes)):
        axes[j].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.02)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def summarize_1d(samples, ci=0.95, ddof=1):
    """
    Basic posterior summary for a 1D array of samples.
    """
    samples = np.asarray(samples)
    lo, hi = np.percentile(samples, [(1-ci)/2*100, (1+ci)/2*100])
    lo_lab = f"{int(((1-ci)/2)*100)}%"
    hi_lab = f"{int(((1+ci)/2)*100)}%"
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "sd": float(samples.std(ddof=ddof)),
        lo_lab: float(lo),
        hi_lab: float(hi)}


def posterior_probabilities(samples_1d, tests):
    """
    Compute posterior probabilities for a 1D samples array given a list of tests.

    Parameters
    ----------
    samples_1d : array-like, shape (n_draws,)
    tests : list of dicts OR list of callables
        Option A (recommended): list of dicts:
            {"name": "P(>0)", "fn": lambda s: s > 0}
        where fn returns a boolean array of same length as s.
        Option B: list of callables; name inferred from __name__.

    Returns
    -------
    dict: {test_name: probability}
    """
    s = np.asarray(samples_1d)

    out = {}
    for t in tests:
        if isinstance(t, dict):
            name = t["name"]
            fn = t["fn"]
        else:
            fn = t
            name = getattr(t, "__name__", "P(test)")

        mask = fn(s)
        mask = np.asarray(mask)

        if mask.shape != s.shape:
            raise ValueError(f"Test '{name}' must return boolean array with shape {s.shape}, got {mask.shape}")

        out[name] = float(mask.mean())
    return out

def summarize_beta_posterior(
    beta_post,
    ci=0.95,
    param_prefix="β",
    tests=None):

    """
    Summarize posterior draws for beta coefficients and optionally append
    arbitrary posterior-probability columns.

    Parameters
    ----------
    beta_post : array-like, shape (n_draws, p)
    ci : float
        Credible interval level.
    param_prefix : str
        Used to label rows like 'β[0]'.
    tests : list[dict] or None
        Each element: {"name": "...", "fn": lambda s: ...}
        where s is 1D samples for a coefficient and fn returns boolean array.

        Example:
          tests = [
            {"name": "P(>0)", "fn": lambda s: s > 0},
            {"name": "P(|.|<0.1)", "fn": lambda s: np.abs(s) < 0.1},
          ]

    Returns
    -------
    summary_df : pd.DataFrame indexed by parameter name
    """
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, p). Got shape={beta_post.shape}")

    n_draws, p = beta_post.shape
    rows = []

    for i in range(p):
        s = beta_post[:, i]
        stats = summarize_1d(s, ci=ci)
        stats["parameter"] = f"{param_prefix}[{i}]"

        if tests is not None:
            stats.update(posterior_probabilities(s, tests))

        rows.append(stats)

    summary_df = pd.DataFrame(rows).set_index("parameter")
    return summary_df


def _acf_1d_fft(x, max_lag):
    """
    Fast ACF via FFT (normalized so acf[0] = 1).
    Returns acf array of length max_lag+1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0 or not np.isfinite(var):
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        return acf

    # Next power-of-two for speed
    m = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(fx * np.conj(fx), n=m)[:max_lag+1]
    acov = acov / n
    acf = acov / var
    acf[0] = 1.0
    return acf

def integrated_autocorr_time(x, max_lag=None, method="geyer"):
    """
    Estimate integrated autocorrelation time tau_int and ESS for a 1D chain.

    tau_int = 1 + 2 * sum_{k=1..K} rho_k

    method="geyer": uses Geyer's initial positive sequence on pair sums:
        stop when rho_{2m-1} + rho_{2m} becomes negative.

    Returns
    -------
    tau_int : float
    ess     : float
    k_used  : int  (largest lag included)
    acf     : np.ndarray (0..max_lag)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 5:
        return np.nan, np.nan, 0, np.array([1.0])

    if max_lag is None:
        # A common safe default: don't go too deep.
        max_lag = min(1000, n - 1)
    else:
        max_lag = int(max_lag)
        max_lag = min(max_lag, n - 1)
        if max_lag < 1:
            max_lag = 1

    acf = _acf_1d_fft(x, max_lag=max_lag)

    # If constant chain, ESS is undefined-ish; return N (tau=1)
    if not np.all(np.isfinite(acf)):
        return np.nan, np.nan, 0, acf

    if method.lower() != "geyer":
        K = 0
        for k in range(1, max_lag + 1):
            if acf[k] <= 0:
                break
            K = k
        tau = 1.0 + 2.0 * np.sum(acf[1:K+1])
        tau = max(tau, 1.0)
        return tau, n / tau, K, acf

    # Geyer initial positive sequence on pair sums
    # consider pairs (1,2), (3,4), ...
    s = 0.0
    K = 0
    mmax = max_lag // 2
    for m in range(1, mmax + 1):
        k1 = 2*m - 1
        k2 = 2*m
        pair_sum = acf[k1] + acf[k2]
        if pair_sum < 0:
            break
        s += pair_sum
        K = k2

    tau = 1.0 + 2.0 * s
    tau = max(tau, 1.0)
    ess = n / tau
    return tau, ess, K, acf


def mcmc_efficiency_report(
    samples,
    param_prefix="β",
    max_lag=None,
    rank_by="ess",          # "ess" (ascending) or "tau" (descending)
    top_k=10,
    return_acf=False):

    """
    Compute tau_int and ESS for each parameter (per chain if 3D), rank worst.

    Parameters
    ----------
    samples : array-like
        2D: (n_draws, p) or 3D: (n_chains, n_draws, p)
    rank_by : {"ess","tau"}
        - "ess": worst = smallest ESS
        - "tau": worst = largest tau_int
    top_k : int
        number of worst parameters to return in a second table
    return_acf : bool
        if True, returns a dict of acf arrays (can be heavy)

    Returns
    -------
    report : dict
        {
          "table": DataFrame (full),
          "worst": DataFrame (top_k worst),
          "acf": dict (optional)
        }
    """
    x = np.asarray(samples, dtype=float)

    if x.ndim == 2:
        # (draw, p)
        n_draws, p = x.shape
        n_chains = 1
        x = x[None, :, :]  # (chain, draw, p)
    elif x.ndim == 3:
        # (chain, draw, p)
        n_chains, n_draws, p = x.shape
    else:
        raise ValueError(f"samples must be 2D or 3D. Got shape={x.shape}")

    rows = []
    acf_store = {} if return_acf else None

    for j in range(p):
        taus = []
        esses = []
        ks = []
        for c in range(n_chains):
            tau, ess, k_used, acf = integrated_autocorr_time(x[c, :, j], max_lag=max_lag, method="geyer")
            taus.append(tau)
            esses.append(ess)
            ks.append(k_used)
            if return_acf:
                acf_store[(c, j)] = acf

        taus = np.array(taus, dtype=float)
        esses = np.array(esses, dtype=float)
        ks = np.array(ks, dtype=int)

        # conservative summaries across chains
        row = {
            "parameter": f"{param_prefix}[{j}]",
            "n_chains": n_chains,
            "n_draws_per_chain": n_draws,
            "tau_mean": float(np.nanmean(taus)),
            "tau_max": float(np.nanmax(taus)),
            "ess_mean": float(np.nanmean(esses)),
            "ess_min": float(np.nanmin(esses)),
            "ess_frac_min": float(np.nanmin(esses) / n_draws) if n_draws > 0 else np.nan,
            "K_used_mean": float(np.nanmean(ks))}
        
        rows.append(row)

    df = pd.DataFrame(rows).set_index("parameter")

    if rank_by == "ess":
        worst = df.sort_values(["ess_min", "tau_max"], ascending=[True, False]).head(top_k)
    elif rank_by == "tau":
        worst = df.sort_values(["tau_max", "ess_min"], ascending=[False, True]).head(top_k)
    else:
        raise ValueError("rank_by must be 'ess' or 'tau'.")

    out = {"table": df, "worst": worst}
    if return_acf:
        out["acf"] = acf_store
    return out

