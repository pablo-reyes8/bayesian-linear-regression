import numpy as np
from scipy.stats import invgamma
from scipy.special import logsumexp
import time


# ---------- Priors for beta (log-density) ----------
def log_prior_beta(beta, prior, prior_kwargs=None, intercept_idx=0):
    """
    log p(beta) up to an additive constant (ok for MH).
    By default, leaves intercept unpenalized (flat) or weakly-normal if you specify.

    Computa el log de la 
    """
    if prior_kwargs is None:
        prior_kwargs = {}

    beta = np.asarray(beta)
    idx = np.arange(beta.size)

    # Option: exclude intercept from shrinkage priors
    shrink_idx = idx[idx != intercept_idx]

    name = prior.lower()

    if name == "normal":
        # Multivariate Normal: N(m0, V0)
        m0 = prior_kwargs["m0"]
        V0 = prior_kwargs["V0"]
        V0_inv = prior_kwargs.get("V0_inv", None)
        if V0_inv is None:
            V0_inv = np.linalg.inv(V0)
        d = beta - m0
        return -0.5 * (d @ V0_inv @ d)

    elif name == "laplace":
        # Independent Laplace(0, b) on coefficients (except intercept)
        # log p = -sum |beta_j|/b  (constants dropped)
        b = float(prior_kwargs.get("b", 1.0))
        return -np.sum(np.abs(beta[shrink_idx])) / b

    elif name == "student_t":
        # Independent Student-t with df nu, scale s, location 0 (except intercept)
        # log p ∝ - (nu+1)/2 * sum log(1 + (beta^2)/(nu*s^2))
        nu = float(prior_kwargs.get("nu", 3.0))
        s  = float(prior_kwargs.get("s", 1.0))
        z2 = (beta[shrink_idx] / s) ** 2
        return -0.5 * (nu + 1.0) * np.sum(np.log1p(z2 / nu))

    elif name == "cauchy":
        # Independent Cauchy(0, s): log p ∝ -sum log(1 + (beta/s)^2)
        s = float(prior_kwargs.get("s", 1.0))
        z2 = (beta[shrink_idx] / s) ** 2
        return -np.sum(np.log1p(z2))

    elif name == "spike_slab":
        # Independent mixture: pi*N(0, slab^2) + (1-pi)*N(0, spike^2)
        # log p = sum log( pi*phi_slab + (1-pi)*phi_spike )
        pi    = float(prior_kwargs.get("pi", 0.5))
        spike = float(prior_kwargs.get("spike", 0.1))
        slab  = float(prior_kwargs.get("slab", 2.0))

        b = beta[shrink_idx]

        # log N(0, s^2) up to constant: -0.5*(b^2/s^2) - log s
        log_phi_slab  = -0.5 * (b**2) / (slab**2)  - np.log(slab)
        log_phi_spike = -0.5 * (b**2) / (spike**2) - np.log(spike)

        # log( pi*exp(a) + (1-pi)*exp(b) ) = logsumexp([log pi + a, log(1-pi)+b])
        comp1 = np.log(pi)     + log_phi_slab
        comp2 = np.log(1.0-pi) + log_phi_spike
        return np.sum(logsumexp(np.vstack([comp1, comp2]), axis=0))

    else:
        raise ValueError(f"Unknown prior='{prior}'. Try: normal, laplace, student_t, cauchy, spike_slab.")
    

# ---------- MCMC: Metropolis-within-Gibbs ----------
def MCMC_LM_beta_nonconj_sigma_conj(
    X, y,
    a0, b0,
    n_draws,
    burn_in=0,
    thinning=1,
    seed=None,
    prior="laplace",
    prior_kwargs=None,
    beta_init=None,
    sigma2_init=1.0,
    proposal_scale=0.15,
    ridge=1e-8,
    intercept_idx=0,
    # ---- new user-friendly knobs ----
    progress=False,
    progress_every=1000,
    show_time=True,
    return_info=False,
):
    """
    Bayesian linear regression with MH for beta (non-conjugate prior allowed)
    and Gibbs for sigma2 (conjugate Inv-Gamma).

    Model:
      y | beta, sigma2 ~ N(X beta, sigma2 I)
      beta ~ prior (non-conjugate allowed; handled by log_prior_beta)
      sigma2 ~ Inv-Gamma(a0,b0)

    Parameters
    ----------
    burn_in : int or float
        If int: number of initial iterations discarded.
        If float in (0,1): fraction of n_draws discarded.
    thinning : int
        Keep every `thinning`-th draw after burn-in.
    progress : bool
        Print progress logs.
    progress_every : int or None
        Log every this many iterations. If 0/None -> no logs.
    show_time : bool
        If logging, also show time per block and ETA.
    return_info : bool
        If True, returns (beta_post, sigma_post, acc_rate, info_dict).
        Otherwise returns (beta_post, sigma_post, acc_rate).

    Returns
    -------
    beta_post : (n_kept, p)
    sigma_post: (n_kept,)
    acc_rate  : float
    info_dict : dict (optional)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]} but X has {n} rows.")

    if prior_kwargs is None:
        prior_kwargs = {}

    # Interpret burn_in if given as fraction
    if isinstance(burn_in, float):
        if not (0.0 <= burn_in < 1.0):
            raise ValueError("If burn_in is float, it must be in [0,1).")
        burn_in_int = int(np.floor(burn_in * n_draws))
    else:
        burn_in_int = int(burn_in)

    if burn_in_int < 0 or burn_in_int >= n_draws:
        raise ValueError(f"burn_in must be in [0, {n_draws-1}]. Got {burn_in_int}.")

    thinning = int(thinning)
    if thinning <= 0:
        raise ValueError("thinning must be a positive integer.")

    # Precompute proposal covariance base: Sigma_base = inv(X'X + ridge I)
    XtX = X.T @ X
    Sigma_base = np.linalg.inv(XtX + ridge * np.eye(p))
    L_base = np.linalg.cholesky(Sigma_base)

    # Init
    beta = np.zeros(p) if beta_init is None else np.asarray(beta_init).copy()
    if beta.shape[0] != p:
        raise ValueError(f"beta_init must have length p={p}. Got {beta.shape[0]}.")
    sigma2 = float(sigma2_init)
    if sigma2 <= 0:
        raise ValueError("sigma2_init must be > 0.")

    # Residual bookkeeping
    r = y - X @ beta
    rss = float(r @ r)

    # Likelihood cache
    def log_lik_from_rss(rss_val, sigma2_val):
        return -0.5 * rss_val / sigma2_val - 0.5 * n * np.log(sigma2_val)

    lp_beta = log_prior_beta(beta, prior, prior_kwargs, intercept_idx=intercept_idx)
    ll = log_lik_from_rss(rss, sigma2)

    # Storage
    kept_beta = []
    kept_sig = []

    # Acceptance counting
    acc = 0

    # Progress timing
    t0 = time.perf_counter()
    last_t = t0
    last_i = 0

    # To report acceptance over blocks
    acc_block = 0

    def _should_log(i):
        return bool(progress) and (progress_every not in (None, 0)) and ((i + 1) % int(progress_every) == 0)

    for t in range(n_draws):
        #  MH step for beta | sigma2, y 
        z = rng.standard_normal(p)
        beta_prop = beta + proposal_scale * (np.sqrt(sigma2) * (L_base @ z))

        delta = beta_prop - beta
        r_prop = r - X @ delta
        rss_prop = float(r_prop @ r_prop)

        lp_beta_prop = log_prior_beta(beta_prop, prior, prior_kwargs, intercept_idx=intercept_idx)
        ll_prop = log_lik_from_rss(rss_prop, sigma2)

        log_alpha = (lp_beta_prop + ll_prop) - (lp_beta + ll)

        if np.log(rng.random()) < log_alpha:
            beta = beta_prop
            r = r_prop
            rss = rss_prop
            lp_beta = lp_beta_prop
            ll = ll_prop
            acc += 1
            acc_block += 1

        #  Gibbs step for sigma2 | beta, y ---
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * rss
        sigma2 = invgamma.rvs(a=an, scale=bn, random_state=rng)

        # Update ll cache after sigma2 changes
        ll = log_lik_from_rss(rss, sigma2)

        #  store ---
        if t >= burn_in_int and ((t - burn_in_int) % thinning == 0):
            kept_beta.append(beta.copy())
            kept_sig.append(sigma2)

        #  logs ---
        if _should_log(t):
            now = time.perf_counter()
            block_time = now - last_t
            block_iters = (t + 1) - last_i
            it_s = block_iters / block_time if block_time > 0 else np.nan

            acc_rate_total = acc / (t + 1)
            acc_rate_block = acc_block / block_iters if block_iters > 0 else np.nan

            msg = (
                f"[{t+1:>7}/{n_draws}] "
                f"acc_total={acc_rate_total:.3f} acc_block={acc_rate_block:.3f} "
                f"sigma2={sigma2:.4g}")

            if show_time:
                remaining = n_draws - (t + 1)
                eta = remaining / it_s if (it_s and np.isfinite(it_s) and it_s > 0) else np.nan
                msg += f" | {it_s:,.1f} it/s | block={block_time:.2f}s | ETA≈{eta/60:.1f} min"

            print(msg)

            last_t = now
            last_i = t + 1
            acc_block = 0  # reset

    beta_post = np.vstack(kept_beta) if kept_beta else np.empty((0, p))
    sigma_post = np.array(kept_sig) if kept_sig else np.empty((0,))
    acc_rate = acc / n_draws

    info = {
        "n_draws": n_draws,
        "burn_in": burn_in_int,
        "thinning": thinning,
        "n_kept": beta_post.shape[0],
        "acc_rate": acc_rate,
        "proposal_scale": proposal_scale,
        "ridge": ridge,
        "prior": prior,
        "seed": seed,
        "runtime_sec": time.perf_counter() - t0}

    if return_info:
        return beta_post, sigma_post, acc_rate, info
    return beta_post, sigma_post, acc_rate