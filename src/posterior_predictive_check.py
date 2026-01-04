import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import gaussian_kde


def attach_posterior_predictive_y(idata, X, seed=123, var_name="y"):
    """
    Create posterior predictive draws y_rep from (beta, sigma2) and attach them to a new idata.

    Requires:
      idata.posterior["beta"]   dims: (chain, draw, coef)
      idata.posterior["sigma2"] dims: (chain, draw)
    Produces:
      new_idata.posterior_predictive[var_name] dims: (chain, draw, obs_id)
    """
    X = np.asarray(X)
    n, p = X.shape

    beta = idata.posterior["beta"].values      # (chain, draw, coef)
    sigma2 = idata.posterior["sigma2"].values  # (chain, draw)

    if beta.ndim != 3:
        raise ValueError(f"Expected beta with 3 dims (chain,draw,coef). Got {beta.shape}")
    if beta.shape[-1] != p:
        raise ValueError(f"X has p={p} columns but beta has {beta.shape[-1]} coefs.")

    # mu: (chain, draw, obs)
    mu = np.einsum("nc,tdc->tdn", X, beta)

    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2)[..., None], size=mu.shape)
    y_rep = mu + eps

    # coords (only those that matter)
    coords = {"coef": idata.posterior.coords["coef"].values, "obs_id": np.arange(n)}
    dims = {
        "beta": ["chain", "draw", "coef"],
        "sigma2": ["chain", "draw"],
        var_name: ["chain", "draw", "obs_id"],}

    idata_ppc = az.from_dict(
        posterior={
            "beta": beta,
            "sigma2": sigma2,
        },
        posterior_predictive={var_name: y_rep},
        coords=coords,
        dims=dims,)
    
    return idata_ppc



def plot_ppc_density_y(
    idata=None,
    y=None,
    y_rep=None,
    var_name="y",
    sample_dims=("chain", "draw"),
    bins=40,
    hdi_prob=0.94,
    kde_points=300,
    show_kde=True,
    show_hist_ppc=True,
    show_obs_hist=True,
    show_rug=True,
    ppc_alpha=0.20,
    hdi_alpha=0.10,
    figsize=(8, 4.5),
    title=None,
    xlim_pad=(0.9, 1.1),
    fmt_thousands=True,
    legend_loc="upper right",
    show=True,
    savepath=None,
    dpi=200,):

    """
    Posterior Predictive Check (density) for y: compare posterior predictive draws vs observed y.

    Provide either:
      - idata (with idata.posterior_predictive[var_name]) AND y (observed), OR
      - y_rep (array of posterior predictive draws) AND y (observed).

    Notes:
      - y_rep is flattened across samples.
      - HDI is computed over y_rep draws.
    """
    if y is None:
        raise ValueError("You must provide observed y.")

    obs_vals = np.asarray(y).ravel()

    if y_rep is None:
        if idata is None:
            raise ValueError("Provide either y_rep or (idata with posterior_predictive).")
        if not hasattr(idata, "posterior_predictive"):
            raise ValueError("idata has no posterior_predictive group.")
        if var_name not in idata.posterior_predictive:
            raise ValueError(f"'{var_name}' not found in idata.posterior_predictive.")
        y_rep_vals = (
            idata.posterior_predictive[var_name]
            .stack(sample=sample_dims)
            .values
            .ravel())
        
    else:
        y_rep_vals = np.asarray(y_rep).ravel()

    if y_rep_vals.size == 0:
        raise ValueError("y_rep is empty after flattening.")
    if obs_vals.size == 0:
        raise ValueError("Observed y is empty after flattening.")

    # KDE over y_rep
    xs = np.linspace(obs_vals.min(), obs_vals.max(), kde_points)
    kde_vals = None
    if show_kde:
        kde = gaussian_kde(y_rep_vals)
        kde_vals = kde(xs)

    # HDI over y_rep
    hdi_low, hdi_high = az.hdi(y_rep_vals, hdi_prob=hdi_prob)

    fig, ax = plt.subplots(figsize=figsize)

    # PPC histogram
    if show_hist_ppc:
        ax.hist(
            y_rep_vals,
            bins=bins,
            density=True,
            alpha=ppc_alpha,
            label="PPC samples",)

    # PPC KDE
    if show_kde:
        ax.plot(xs, kde_vals, lw=2, label="KDE PPC mean")

    # Observed histogram (step)
    if show_obs_hist:
        ax.hist(
            obs_vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Observed",)

    if show_rug:
        ymin, ymax = ax.get_ylim()
        rug_h = 0.05 * (ymax - ymin)
        ax.vlines(obs_vals, ymin, ymin + rug_h, linewidth=0.7, alpha=0.9)

    top_y = (kde_vals.max() if (show_kde and kde_vals is not None) else ax.get_ylim()[1]) * 1.1
    ax.fill_betweenx(
        [0, top_y],
        hdi_low,
        hdi_high,
        alpha=hdi_alpha,
        label=f"HDI {int(hdi_prob*100)}%",)


    lo, hi = obs_vals.min(), obs_vals.max()
    ax.set_xlim(lo * xlim_pad[0], hi * xlim_pad[1])

    if title is None:
        title = rf"Posterior Predictive Check para ${var_name}$"
    ax.set_title(title, pad=12)
    ax.set_xlabel(rf"Valor de ${var_name}$")
    ax.set_ylabel("Densidad")

    ax.legend(loc=legend_loc, frameon=True)

    if fmt_thousands:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax



def plot_ppc_residuals(
    idata,
    X,
    y,
    var_name="y",
    sample_dims=("chain", "draw"),
    standardize=True,
    use_posterior_predictive=True,  # if False, simulate y_rep
    rng_seed=123,
    subsample_draws=5000,           # to keep memory sane
    bins=50,
    hdi_prob=0.94,
    kde_points=400,
    figsize=(8, 4.5),
    title=None,
    legend_loc="upper right",
    show=True,
    savepath=None,
    dpi=200):

    """
    PPC for residuals in a Bayesian linear regression.

    Requires idata.posterior["beta"] (chain,draw,coef) and idata.posterior["sigma2"] (chain,draw).
    Optionally uses idata.posterior_predictive[var_name] for y_rep; otherwise simulates y_rep.

    Residuals:
      - Observed: e_obs = y - X * E[beta]
      - Replicated (per draw): e_rep = y_rep^(s) - X * beta^(s)

    If standardize=True:
      - z_obs = e_obs / sqrt(E[sigma2])
      - z_rep = e_rep / sqrt(sigma2^(s))
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n = y.size
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but y has length {n}.")

    # stack posterior draws
    beta = idata.posterior["beta"].stack(sample=sample_dims).values 
    # Ensure shape (S, p)
    if beta.ndim == 2 and beta.shape[0] == X.shape[1]:
        beta = beta.T
    elif beta.ndim != 2:
        raise ValueError(f"Unexpected beta shape after stacking: {beta.shape}")
    S, p = beta.shape

    sigma2 = idata.posterior["sigma2"].stack(sample=sample_dims).values.ravel()  # (S,)
    if sigma2.shape[0] != S:
        raise ValueError("beta and sigma2 must have the same number of posterior draws after stacking.")

    #subsample draws if needed
    if subsample_draws is not None and S > subsample_draws:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(S, size=subsample_draws, replace=False)
        beta = beta[idx, :]
        sigma2 = sigma2[idx]
        S = beta.shape[0]

    # observed residuals: use posterior mean beta for a single e_obs vector
    beta_mean = beta.mean(axis=0)
    yhat_mean = X @ beta_mean
    e_obs = y - yhat_mean
    rng = np.random.default_rng(rng_seed)


    if use_posterior_predictive and hasattr(idata, "posterior_predictive") and (var_name in idata.posterior_predictive):
        y_rep = idata.posterior_predictive[var_name].stack(sample=sample_dims).values
        if y_rep.ndim != 2:
            y_rep = np.asarray(y_rep).reshape(-1, n)
        if y_rep.shape[0] == n and y_rep.shape[1] != n:
            y_rep = y_rep.T 
        if subsample_draws is not None and y_rep.shape[0] > subsample_draws:
            # NOTE: if we subsampled posterior draws, we want the same indices.
            # But we don't have the original idx if S <= subsample_draws. Handle both cases:
            try:
                y_rep = y_rep[idx, :]
            except NameError:
                y_rep = y_rep[:subsample_draws, :]
        if y_rep.shape[0] != S:
            # last resort: align to min
            S2 = min(S, y_rep.shape[0])
            beta, sigma2, y_rep = beta[:S2], sigma2[:S2], y_rep[:S2]
            S = S2
    else:
        # simulate y_rep from likelihood: y_rep = X beta + eps, eps ~ N(0, sigma2)
        mu = (X @ beta.T).T  # (S, n)
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2)[:, None], size=(S, n))
        y_rep = mu + eps  # (S, n)

    # residuals per draw: e_rep = y_rep - X beta
    mu = (X @ beta.T).T  # (S, n)
    e_rep = (y_rep - mu).ravel()  # flattened (S*n,)

    # standardize if requested
    if standardize:
        denom_obs = np.sqrt(np.mean(sigma2))
        e_obs_plot = e_obs / denom_obs
        e_rep_plot = ( (y_rep - mu) / np.sqrt(sigma2)[:, None] ).ravel()
        xlab = "Residual estandarizado"
    else:
        e_obs_plot = e_obs
        e_rep_plot = e_rep
        xlab = "Residual"

    # KDE + HDI on replicated residuals
    kde = gaussian_kde(e_rep_plot)
    lo_x = min(e_obs_plot.min(), np.quantile(e_rep_plot, 0.001))
    hi_x = max(e_obs_plot.max(), np.quantile(e_rep_plot, 0.999))
    xs = np.linspace(lo_x, hi_x, kde_points)
    kde_vals = kde(xs)

    hdi_low, hdi_high = az.hdi(e_rep_plot, hdi_prob=hdi_prob)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(e_rep_plot, bins=bins, density=True, alpha=0.20, label="PPC residuals")
    ax.plot(xs, kde_vals, lw=2, label="KDE PPC")

    ax.hist(e_obs_plot, bins=bins, density=True, histtype="step", linewidth=2, label="Observed residuals")
    ymin, ymax = ax.get_ylim()
    rug_h = 0.05 * (ymax - ymin)
    ax.vlines(e_obs_plot, ymin, ymin + rug_h, linewidth=0.7, alpha=0.9)

    ax.fill_betweenx([0, kde_vals.max() * 1.1], hdi_low, hdi_high, alpha=0.10, label=f"HDI {int(hdi_prob*100)}%")

    if title is None:
        title = "PPC para residuales (estandarizados)" if standardize else "PPC para residuales"
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Densidad")
    ax.legend(loc=legend_loc, frameon=True)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    out = {
        "fig": fig,
        "ax": ax,
        "e_obs": e_obs_plot,
        "e_rep": e_rep_plot,
        "hdi": (float(hdi_low), float(hdi_high)),
        "ppc_pval_mean_gt0": float((e_rep_plot.mean() > 0)), 
        "ppc_pval_tail_area_mean": float((e_rep_plot.mean() >= e_obs_plot.mean()).mean()) if hasattr(e_rep_plot, "mean") else np.nan,}
    return out