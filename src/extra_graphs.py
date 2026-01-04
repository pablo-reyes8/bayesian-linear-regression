import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_posterior_pairplot(
    beta_post,
    max_plots=None,
    which=None,
    coef_names=None,
    sample_draws=None,
    seed=123,
    corner=True,
    diag_kind="kde",          # "kde" or "hist"
    kind="scatter",           # "scatter" or "reg"
    plot_kws=None,
    diag_kws=None,
    height=1.9,
    aspect=1.0,
    suptitle="Posterior Joint Distributions",
    title_y=1.02):

    """
    Seaborn pairplot for posterior draws of beta.

    Parameters
    ----------
    beta_post : array-like (n_draws, p)
    max_plots : int or None
        Plot first max_plots coefficients.
    which : list[int] or None
        Specific coefficient indices to plot (overrides max_plots).
    coef_names : list[str] or None
        Names for coefficients; if None uses β[i].
    sample_draws : int or None
        Subsample draws for speed.
    seed : int
        RNG seed for subsampling.
    corner, diag_kind, kind : seaborn pairplot options
    plot_kws, diag_kws : dict or None
        Passed to pairplot.
    height, aspect : figure sizing for pairplot grid
    suptitle : str or None
        Adds a figure title if provided.
    """
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, p). Got shape={beta_post.shape}")
    n_draws, p = beta_post.shape

    # choose indices
    if which is not None:
        idx = [int(i) for i in which]
    else:
        n_to_plot = p if max_plots is None else min(int(max_plots), p)
        idx = list(range(n_to_plot))
    if len(idx) < 2:
        raise ValueError("pairplot needs at least 2 parameters (select >=2).")

    # names
    if coef_names is None:
        names = [f"β[{i}]" for i in idx]
    else:
        if len(coef_names) != p:
            raise ValueError(f"coef_names must have length p={p}. Got {len(coef_names)}.")
        names = [coef_names[i] for i in idx]

    draws = beta_post[:, idx]

    # subsample draws for speed
    if sample_draws is not None and n_draws > sample_draws:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_draws, size=int(sample_draws), replace=False)
        draws = draws[sel, :]

    df = pd.DataFrame(draws, columns=names)

    if plot_kws is None:
        plot_kws = {"linewidth": 0.3, "s": 8, "alpha": 0.35}
    if diag_kws is None:
        diag_kws = {"fill": True}

    g = sns.pairplot(
        df,
        corner=corner,
        diag_kind=diag_kind,
        kind=kind,
        plot_kws=plot_kws,
        diag_kws=diag_kws,
        height=height,
        aspect=aspect,)

    if suptitle is not None:
        g.fig.suptitle(suptitle, y=title_y)

    return g


def plot_posterior_corr(
    beta_post,
    max_plots=None,
    which=None,
    coef_names=None,
    sample_draws=None,
    seed=123,
    method="pearson",
    annot=True,
    fmt=".2f",
    annot_kws=None,
    figsize=None,
    title="Posterior Correlation Matrix",
    center=0.0,
    vmin=-1.0,
    vmax=1.0,
    cmap="coolwarm",
    cbar=True,
    square=True,
    mask_upper=True,
    rotate_x=0,
    rotate_y=0,
    tick_fontsize=10,
    show=True,
):
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, p). Got shape={beta_post.shape}")
    n_draws, p = beta_post.shape

    # choose indices
    if which is not None:
        idx = [int(i) for i in which]
    else:
        n_to_plot = p if max_plots is None else min(int(max_plots), p)
        idx = list(range(n_to_plot))
    if len(idx) < 2:
        raise ValueError("Need at least 2 parameters.")

    # names
    if coef_names is None:
        names = [f"β[{i}]" for i in idx]
    else:
        if len(coef_names) != p:
            raise ValueError(f"coef_names must have length p={p}. Got {len(coef_names)}.")
        names = [coef_names[i] for i in idx]

    draws = beta_post[:, idx]

    # subsample draws for speed
    if sample_draws is not None and n_draws > sample_draws:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_draws, size=int(sample_draws), replace=False)
        draws = draws[sel, :]

    df = pd.DataFrame(draws, columns=names)
    corr = df.corr(method=method)

    k = len(names)
    if figsize is None:
        figsize = (0.8 * k + 4.0, 0.7 * k + 3.0)

    if annot_kws is None:
        annot_kws = {"size": max(7, 12 - k)} 

    fig, ax = plt.subplots(figsize=figsize)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        annot=annot,
        fmt=fmt,
        annot_kws=annot_kws,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        square=square,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=rotate_x, labelsize=tick_fontsize)
    ax.tick_params(axis="y", rotation=rotate_y, labelsize=tick_fontsize)

    plt.tight_layout()
    if show:
        plt.show()

    return corr, fig, ax