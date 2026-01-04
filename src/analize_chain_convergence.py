import numpy as np
import pandas as pd
import arviz as az


def build_idata_from_chains(
    beta_chains,
    sigma2_chains=None,
    coef_names=None,
    truncate_to_min_draws=True,
):
    """
    Build an ArviZ InferenceData from multiple MCMC chains.

    Parameters
    ----------
    beta_chains : list[array], each shape (n_draws, p)
        Beta draws per chain.
    sigma2_chains : list[array] or None, each shape (n_draws,)
        Sigma^2 draws per chain (optional).
    coef_names : list[str] or None
        Names for coefficients. If None, uses 0..p-1.
    truncate_to_min_draws : bool
        If chains have different lengths, truncate all to the minimum length.

    Returns
    -------
    idata : arviz.InferenceData
    """
    if not isinstance(beta_chains, (list, tuple)) or len(beta_chains) < 1:
        raise ValueError("beta_chains must be a non-empty list of arrays (n_draws, p).")

    beta_chains = [np.asarray(b) for b in beta_chains]
    if any(b.ndim != 2 for b in beta_chains):
        raise ValueError("Each beta chain must be 2D with shape (n_draws, p).")

    ps = [b.shape[1] for b in beta_chains]
    if len(set(ps)) != 1:
        raise ValueError(f"All beta chains must have the same number of coefs p. Got {ps}.")
    p = ps[0]

    draws = [b.shape[0] for b in beta_chains]
    if len(set(draws)) != 1:
        if truncate_to_min_draws:
            n_draws = min(draws)
            beta_chains = [b[:n_draws] for b in beta_chains]
            if sigma2_chains is not None:
                sigma2_chains = [np.asarray(s)[:n_draws] for s in sigma2_chains]
        else:
            raise ValueError(f"All chains must have same n_draws (or set truncate_to_min_draws=True). Got {draws}.")

    beta_arviz = np.stack(beta_chains, axis=0)  # (chain, draw, coef)
    n_chains, n_draws, _ = beta_arviz.shape

    if coef_names is None:
        coef_names = [str(i) for i in range(p)]
    if len(coef_names) != p:
        raise ValueError(f"coef_names must have length p={p}. Got {len(coef_names)}.")

    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws), "coef": coef_names}
    dims = {"beta": ["chain", "draw", "coef"]}

    posterior = {"beta": beta_arviz}

    if sigma2_chains is not None:
        if not isinstance(sigma2_chains, (list, tuple)) or len(sigma2_chains) != n_chains:
            raise ValueError("sigma2_chains must be a list with same number of chains as beta_chains.")
        sigma2_chains = [np.asarray(s) for s in sigma2_chains]
        if any(s.ndim != 1 for s in sigma2_chains):
            raise ValueError("Each sigma2 chain must be 1D with shape (n_draws,).")
        if any(s.shape[0] != n_draws for s in sigma2_chains):
            raise ValueError("Each sigma2 chain must have same n_draws as beta chains (after truncation).")

        sigma2_arviz = np.stack(sigma2_chains, axis=0)  # (chain, draw)
        posterior["sigma2"] = sigma2_arviz
        dims["sigma2"] = ["chain", "draw"]

    idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
    return idata


def arviz_mcmc_report(
    idata,
    var_names=("beta", "sigma2"),
    hdi_prob=0.95,
    round_to=4,
    rhat_warn=1.01,
    ess_bulk_warn=400,
    mcse_rel_warn=0.10,
    make_latex=True,
):
    """
    Produce a nicer diagnostics report from an ArviZ InferenceData.

    Returns a dict with:
      - 'summary': pandas DataFrame (paper-like table)
      - 'diagnostics': dict with worst-case diagnostics
      - 'text': formatted executive summary string
      - 'latex': LaTeX table string (optional)
    """
    # Main summary table
    summ = az.summary(
        idata,
        var_names=list(var_names),
        hdi_prob=hdi_prob,
        round_to=round_to,)

    # Add a relative MCSE column (helpful to interpret MCSE)
    if "mcse_mean" in summ.columns and "mean" in summ.columns:
        summ["mcse_rel"] = (summ["mcse_mean"] / summ["mean"].replace(0, np.nan)).abs()

    # Worst-case diagnostics (global)
    diags = {}
    if "r_hat" in summ.columns:
        diags["worst_rhat"] = float(np.nanmax(summ["r_hat"].values))
        diags["worst_rhat_param"] = summ["r_hat"].astype(float).idxmax()
    if "ess_bulk" in summ.columns:
        diags["min_ess_bulk"] = float(np.nanmin(summ["ess_bulk"].values))
        diags["min_ess_bulk_param"] = summ["ess_bulk"].astype(float).idxmin()
    if "ess_tail" in summ.columns:
        diags["min_ess_tail"] = float(np.nanmin(summ["ess_tail"].values))
        diags["min_ess_tail_param"] = summ["ess_tail"].astype(float).idxmin()
    if "mcse_rel" in summ.columns:
        diags["max_mcse_rel"] = float(np.nanmax(summ["mcse_rel"].values))
        diags["max_mcse_rel_param"] = summ["mcse_rel"].astype(float).idxmax()

    # Flagging rules
    flags = []
    if "r_hat" in summ.columns:
        bad = summ["r_hat"] > rhat_warn
        if bool(np.any(bad)):
            flags.append(f"R-hat > {rhat_warn}: {int(bad.sum())} parámetro(s)")
    if "ess_bulk" in summ.columns:
        bad = summ["ess_bulk"] < ess_bulk_warn
        if bool(np.any(bad)):
            flags.append(f"ESS bulk < {ess_bulk_warn}: {int(bad.sum())} parámetro(s)")
    if "mcse_rel" in summ.columns:
        bad = summ["mcse_rel"] > mcse_rel_warn
        if bool(np.any(bad)):
            flags.append(f"MCSE_rel > {mcse_rel_warn:.2f}: {int(bad.sum())} parámetro(s)")

    n_chains = idata.posterior.dims.get("chain", None)
    n_draws = idata.posterior.dims.get("draw", None)

    text_lines = []
    text_lines.append("=== MCMC Diagnostics Report (ArviZ) ===")
    if n_chains is not None and n_draws is not None:
        text_lines.append(f"Chains: {n_chains} | Draws per chain: {n_draws} | Total draws: {n_chains * n_draws}")
    text_lines.append(f"HDI prob: {hdi_prob}")
    if flags:
        text_lines.append("Flags:")
        for f in flags:
            text_lines.append(f"  - {f}")
    else:
        text_lines.append("No red flags under chosen thresholds.")

    # Add worst-case details
    if diags:
        text_lines.append("Worst-case diagnostics:")
        for k, v in diags.items():
            text_lines.append(f"  - {k}: {v}")

    report = {
        "summary": summ,
        "diagnostics": diags,
        "text": "\n".join(text_lines),}

    if make_latex:
        # A clean subset of columns tends to read better in a paper
        preferred_cols = [c for c in ["mean", "sd", f"hdi_{int((1-hdi_prob)/2*100)}%", f"hdi_{int((1-(1-hdi_prob)/2)*100)}%",
                                     "ess_bulk", "ess_tail", "r_hat", "mcse_mean"] if c in summ.columns]
        latex_df = summ[preferred_cols].copy() if preferred_cols else summ.copy()
        report["latex"] = latex_df.to_latex(escape=True, float_format=lambda x: f"{x:.{round_to}f}")
    return report



from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import math 


def plot_beta_acf(
    beta_post,
    lags=20,
    alpha=0.05,
    max_plots=None,
    which=None,
    ncols=4,
    figsize=None,
    sharey=False,
    use_fft=True,
    suptitle=None,
    show=True,
    savepath=None,
    dpi=200):

    """
    Plot ACF (autocorrelation function) for MCMC chains of beta coefficients.

    Parameters
    ----------
    beta_post : array-like, shape (n_draws, p)
        Posterior draws for beta.
    lags : int
        Number of lags for ACF.
    alpha : float or None
        Significance level for confidence intervals in statsmodels.plot_acf.
        Set None to disable CI bands.
    max_plots : int or None
        Plot first max_plots coefficients (ignored if which provided).
    which : list[int] or None
        Plot specific coefficient indices.
    ncols : int
        Columns in subplot grid.
    figsize : tuple or None
        Figure size; auto if None.
    sharey : bool
        Share y-axis across subplots (often helpful for comparisons).
    use_fft : bool
        Use FFT-based ACF computation (faster for long chains).
    suptitle : str or None
        Figure-level title.
    show : bool
        Whether to plt.show().
    savepath : str or None
        Save figure to path if provided.
    dpi : int
        DPI for saving.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, p). Got shape={beta_post.shape}")

    n_draws, p = beta_post.shape

    # choose coefficients
    if which is not None:
        idx = [int(i) for i in which]
        for i in idx:
            if i < 0 or i >= p:
                raise ValueError(f"Coefficient index out of range: {i} (p={p})")
    else:
        n_to_plot = p if max_plots is None else min(int(max_plots), p)
        idx = list(range(n_to_plot))

    k = len(idx)
    if k == 0:
        raise ValueError("No coefficients selected to plot.")

    nrows = math.ceil(k / ncols)
    if figsize is None:
        figsize = (4.0 * ncols, 2.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    for j, coef_i in enumerate(idx):
        ax = axes[j]
        plot_acf(
            beta_post[:, coef_i],
            ax=ax,
            lags=lags,
            alpha=alpha,
            title=f"ACF β[{coef_i}]",
            fft=use_fft,)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorr.")

    # turn off unused axes
    for j in range(k, len(axes)):
        axes[j].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.02)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes