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
    


def MCMC_LM_beta_nonconj_sigma_conj_adaptcov_slopes(
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

    progress=False,
    progress_every=1000,
    show_time=True,
    return_info=False,

    adapt_scale=True,
    target_accept=0.25,
    adapt_every=200,
    adapt_start=200,
    adapt_max_scale=10.0,
    adapt_min_scale=1e-6,
    adapt_gain=1.0,

    # adaptive covariance for slopes (burn-in only) ---
    adapt_cov=True,
    cov_start="xtx",     # "xtx" or "identity"
    cov_ridge=1e-6,      # jitter added to empirical cov
    cov_update_every=200, # update Cholesky every this many iters during burn-in
    cov_start_at=500,     # start using empirical cov after this many iters
    max_jitter_tries=8,   # robust chol
):
    """
    Bayesian linear regression:
      y | beta, sigma2 ~ N(X beta, sigma2 I)
      beta ~ (non-conjugate allowed via log_prior_beta)
      sigma2 ~ Inv-Gamma(a0, b0)

    Kernel:
      - Intercept beta0 updated by Gibbs (exact) IF intercept not penalized
      - Slopes updated by MH with:
          (i) adaptive scaling (Robbins–Monro on log scale)
          (ii) adaptive covariance (Adaptive Metropolis) for slopes during burn-in
      - sigma2 updated by Gibbs (Inv-Gamma)

    Returns:
      beta_post: (n_kept, p)
      sigma_post: (n_kept,)
      acc_rate_slopes: float
      info (optional)
    """

    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]} but X has {n} rows.")
    if prior_kwargs is None:
        prior_kwargs = {}

    # burn_in handling 
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

    # indices: intercept vs slopes 
    if not (0 <= intercept_idx < p):
        raise ValueError(f"intercept_idx must be in [0, {p-1}]. Got {intercept_idx}.")
    slopes_idx = np.array([j for j in range(p) if j != intercept_idx], dtype=int)
    d = slopes_idx.size  # number of slopes

    x0 = X[:, intercept_idx]               
    Xs = X[:, slopes_idx]                
    x0Tx0 = float(x0 @ x0)              

    # init beta, sigma2 
    beta = np.zeros(p) if beta_init is None else np.asarray(beta_init).copy()
    if beta.shape[0] != p:
        raise ValueError(f"beta_init must have length p={p}. Got {beta.shape[0]}.")
    sigma2 = float(sigma2_init)
    if sigma2 <= 0:
        raise ValueError("sigma2_init must be > 0.")

    #  residual bookkeeping 
    # r = y - X beta
    r = y - X @ beta
    rss = float(r @ r)

    #  likelihood cache (up to constants) 
    def log_lik_from_rss(rss_val, sigma2_val):
        return -0.5 * rss_val / sigma2_val - 0.5 * n * np.log(sigma2_val)



    # current log prior & log lik
    lp_beta = log_prior_beta(beta, prior, prior_kwargs, intercept_idx=intercept_idx)
    ll = log_lik_from_rss(rss, sigma2)

    kept_beta = []
    kept_sig = []

    acc = 0
    acc_block = 0


    log_scale = np.log(float(proposal_scale))
    n_adapt = 0
    scale_path = []

    # --- proposal covariance for slopes ---
    # start with something sensible: inv(Xs'Xs + ridge I) or identity
    if cov_start == "xtx":
        XtX_s = Xs.T @ Xs
        Sigma_s = np.linalg.inv(XtX_s + ridge * np.eye(d))
    elif cov_start == "identity":
        Sigma_s = np.eye(d)
    else:
        raise ValueError("cov_start must be 'xtx' or 'identity'.")

    def _chol_pd(A):
        """Robust Cholesky with escalating jitter."""
        A = 0.5 * (A + A.T)
        jitter = cov_ridge
        for _ in range(max_jitter_tries):
            try:
                return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        # last try: big jitter
        return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))

    L_s = _chol_pd(Sigma_s)

    # --- online covariance estimation for slopes (Welford) ---
    # We track covariance of the chain states of slopes during burn-in.
    mean_s = beta[slopes_idx].copy()
    C_s = np.zeros((d, d), dtype=float)
    n_cov = 1  # we've seen 1 sample (initial)

    cov_path = []

    t0 = time.perf_counter()
    last_t = t0
    last_i = 0

    def _should_log(i):
        return bool(progress) and (progress_every not in (None, 0)) and ((i + 1) % int(progress_every) == 0)

    # ===========================
    #           LOOP
    # ===========================
    for t in range(n_draws):

        # ---------- Gibbs update for intercept beta0 (exact) ----------
        # This is valid if intercept is not in the non-conjugate prior (i.e., prior doesn't depend on beta0).
        # Conditional under flat prior for beta0:
        #   beta0 | slopes, sigma2, y ~ N( (x0'(y - Xs bs))/x0Tx0, sigma2/x0Tx0 )
        bs = beta[slopes_idx]
        # compute partial residual excluding intercept: y - Xs bs
        r_no0 = y - Xs @ bs
        mean_b0 = float((x0 @ r_no0) / x0Tx0)
        var_b0 = sigma2 / x0Tx0
        b0_new = mean_b0 + np.sqrt(var_b0) * rng.standard_normal()

        # update residuals cheaply after changing intercept
        delta0 = b0_new - beta[intercept_idx]
        if delta0 != 0.0:
            beta[intercept_idx] = b0_new
            # r = y - (old)Xbeta  -> new r = r - x0*delta0
            r = r - x0 * delta0
            rss = float(r @ r)
            lp_beta = log_prior_beta(beta, prior, prior_kwargs, intercept_idx=intercept_idx)
            ll = log_lik_from_rss(rss, sigma2)



        # ----------  MH update for slopes ----------
        proposal_scale_t = float(np.exp(log_scale))

        z = rng.standard_normal(d)
        bs_prop = bs + proposal_scale_t * (np.sqrt(sigma2) * (L_s @ z))

        beta_prop = beta.copy()
        beta_prop[slopes_idx] = bs_prop

        # residual update: r_prop = y - X beta_prop
        # but we have r = y - X beta, so:
        # delta_slopes = bs_prop - bs
        delta_s = bs_prop - bs
        r_prop = r - Xs @ delta_s
        rss_prop = float(r_prop @ r_prop)

        lp_beta_prop = log_prior_beta(beta_prop, prior, prior_kwargs, intercept_idx=intercept_idx)
        ll_prop = log_lik_from_rss(rss_prop, sigma2)

        log_alpha = (lp_beta_prop + ll_prop) - (lp_beta + ll)

        if np.isfinite(log_alpha) and (np.log(rng.random()) < log_alpha):
            beta = beta_prop
            r = r_prop
            rss = rss_prop
            lp_beta = lp_beta_prop
            ll = ll_prop
            acc += 1
            acc_block += 1

        # ----------  Gibbs for sigma2 ----------
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * rss
        sigma2 = invgamma.rvs(a=an, scale=bn, random_state=rng)
        ll = log_lik_from_rss(rss, sigma2)

        # ---------- update online cov stats for slopes (during burn-in only) ----------
        # Use current state (accepted or not) for adaptation.
        if adapt_cov and (t < burn_in_int):
            bs_curr = beta[slopes_idx]
            n_cov += 1
            delta = bs_curr - mean_s
            mean_s = mean_s + delta / n_cov
            C_s = C_s + np.outer(delta, (bs_curr - mean_s))

            # refresh proposal covariance occasionally after some warmup
            if (t + 1 >= cov_start_at) and ((t + 1) % cov_update_every == 0) and (n_cov > 2):
                emp_cov = C_s / (n_cov - 1)
                emp_cov = 0.5 * (emp_cov + emp_cov.T) + cov_ridge * np.eye(d)
                L_s = _chol_pd(emp_cov)
                if len(cov_path) < 50:
                    cov_path.append(emp_cov.copy())

        # ---------- adapt proposal SCALE (burn-in only) ----------
        if adapt_scale and (t < burn_in_int) and (t + 1 >= adapt_start) and ((t + 1) % adapt_every == 0):
            n_adapt += 1
            step = adapt_gain / np.sqrt(n_adapt)

            acc_rate_block = acc_block / adapt_every
            log_scale = log_scale + step * (acc_rate_block - target_accept)

            new_scale = float(np.exp(log_scale))
            new_scale = min(adapt_max_scale, max(adapt_min_scale, new_scale))
            log_scale = np.log(new_scale)

            scale_path.append(new_scale)
            acc_block = 0  # reset after adaptation

        # ---------- store ----------
        if t >= burn_in_int and ((t - burn_in_int) % thinning == 0):
            kept_beta.append(beta.copy())
            kept_sig.append(sigma2)

        # ----------  logs ----------
        if _should_log(t):
            now = time.perf_counter()
            block_time = now - last_t
            block_iters = (t + 1) - last_i
            it_s = block_iters / block_time if block_time > 0 else np.nan

            acc_rate_total = acc / (t + 1)

            msg = (
                f"[{t+1:>7}/{n_draws}] "
                f"acc_total={acc_rate_total:.3f} "
                f"sigma2={sigma2:.4g} "
                f"prop_scale={float(np.exp(log_scale)):.3g}")
            if show_time:
                remaining = n_draws - (t + 1)
                eta = remaining / it_s if (it_s and np.isfinite(it_s) and it_s > 0) else np.nan
                msg += f" | {it_s:,.1f} it/s | block={block_time:.2f}s | ETA≈{eta/60:.1f} min"
            print(msg)

            last_t = now
            last_i = t + 1

    beta_post = np.vstack(kept_beta) if kept_beta else np.empty((0, p))
    sigma_post = np.array(kept_sig) if kept_sig else np.empty((0,))
    acc_rate = acc / n_draws

    info = {
        "n_draws": n_draws,
        "burn_in": burn_in_int,
        "thinning": thinning,
        "n_kept": beta_post.shape[0],
        "acc_rate_slopes": acc_rate,
        "final_proposal_scale": float(np.exp(log_scale)),
        "proposal_scale_path": np.array(scale_path) if scale_path else None,
        "adapt_scale": adapt_scale,
        "target_accept": target_accept,
        "adapt_every": adapt_every,
        "adapt_start": adapt_start,
        "adapt_cov": adapt_cov,
        "cov_start": cov_start,
        "cov_ridge": cov_ridge,
        "cov_update_every": cov_update_every,
        "cov_start_at": cov_start_at,
        "cov_snapshots": cov_path if cov_path else None,
        "ridge": ridge,
        "prior": prior,
        "seed": seed,
        "runtime_sec": time.perf_counter() - t0}

    if return_info:
        return beta_post, sigma_post, acc_rate, info
    return beta_post, sigma_post, acc_rate
