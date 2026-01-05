import numpy as np
from scipy.special import logsumexp
from scipy.stats import invgamma
import time


def log_prior_sigma2(sigma2, prior="invgamma", prior_kwargs=None):
    """
    log p(sigma2) up to additive constant.
    sigma2 must be > 0.
    """
    if prior_kwargs is None:
        prior_kwargs = {}
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        return -np.inf

    name = prior.lower()

    if name == "invgamma":
        # p(s2) ∝ s2^{-(a+1)} exp(-b/s2)
        a = float(prior_kwargs.get("a", 2.0))
        b = float(prior_kwargs.get("b", 1.0))
        return -(a + 1.0) * np.log(sigma2) - b / sigma2

    elif name == "lognormal":
        # log(s2) ~ N(mu, tau^2)  =>  log p(s2) = -0.5*((log s2 - mu)/tau)^2 - log s2
        mu  = float(prior_kwargs.get("mu", 0.0))
        tau = float(prior_kwargs.get("tau", 1.0))
        ls2 = np.log(sigma2)
        return -0.5 * ((ls2 - mu) / tau) ** 2 - ls2

    elif name == "halfcauchy_sigma":
        # sigma ~ HalfCauchy(scale=s0). Induce p(sigma2):
        # p(sigma) ∝ 1/(1+(sigma/s0)^2), sigma>0
        # sigma = sqrt(s2), and p(s2)=p(sigma)*|d sigma / d s2| = p(sigma) * 1/(2 sqrt(s2))
        s0 = float(prior_kwargs.get("s0", 1.0))
        sigma = np.sqrt(sigma2)
        return -np.log1p((sigma / s0) ** 2) - np.log(sigma)  # constants dropped; includes -log(2) dropped

    else:
        raise ValueError("Unknown sigma2 prior. Try: invgamma, lognormal, halfcauchy_sigma.")
    

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


def MCMC_LM_beta_nonconj_sigma2_nonconj_adaptcov_slopes(
    X, y,
    n_draws,
    burn_in=0,
    thinning=1,
    seed=None,

    # beta prior
    prior="laplace",
    prior_kwargs=None,
    intercept_idx=0,
    gibbs_intercept=True,  # solo válido si el intercepto NO entra al prior (flat)

    # sigma2 prior (NON-conjugate)
    sigma2_prior="halfcauchy_sigma",
    sigma2_prior_kwargs=None,

    # init
    beta_init=None,
    sigma2_init=1.0,

    # slopes proposal
    proposal_scale=0.15,
    ridge=1e-8,

    # adapt slopes scale
    adapt_scale=True,
    target_accept=0.25,
    adapt_every=200,
    adapt_start=200,
    adapt_max_scale=10.0,
    adapt_min_scale=1e-6,
    adapt_gain=1.0,

    # adaptive covariance for slopes (burn-in only)
    adapt_cov=True,
    cov_start="xtx",
    cov_ridge=1e-6,
    cov_update_every=200,
    cov_start_at=500,
    max_jitter_tries=8,

    # sigma2 MH proposal (in log sigma2)
    sigma2_prop_scale=0.25,
    adapt_sigma2_scale=True,
    target_accept_sigma2=0.44,
    adapt_sigma2_every=200,
    adapt_sigma2_start=200,
    adapt_sigma2_gain=1.0,
    sigma2_prop_min=1e-6,
    sigma2_prop_max=10.0,

    progress=False,
    progress_every=1000,
    show_time=True,
    return_info=False,
):
    """
    y | beta, sigma2 ~ N(X beta, sigma2 I)
    beta ~ arbitrary prior via log_prior_beta (non-conjugate)
    sigma2 ~ arbitrary prior via log_prior_sigma2 (non-conjugate)

    Updates per iteration:
      1) Intercept: Gibbs (optional; only if prior doesn't depend on it)
      2) Slopes: MH block with adaptive cov + adaptive scale (burn-in)
      3) sigma2: MH on theta=log(sigma2) with adaptive scale (burn-in)

    Requires your function log_prior_beta(beta, prior, prior_kwargs, intercept_idx).
    Requires log_prior_sigma2(sigma2, sigma2_prior, sigma2_prior_kwargs).
    """

    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]} but X has {n} rows.")
    if prior_kwargs is None:
        prior_kwargs = {}
    if sigma2_prior_kwargs is None:
        sigma2_prior_kwargs = {}

    # burn-in handling
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

    if not (0 <= intercept_idx < p):
        raise ValueError(f"intercept_idx must be in [0, {p-1}]. Got {intercept_idx}.")
    slopes_idx = np.array([j for j in range(p) if j != intercept_idx], dtype=int)
    d = slopes_idx.size

    x0 = X[:, intercept_idx]
    Xs = X[:, slopes_idx]
    x0Tx0 = float(x0 @ x0)

    beta = np.zeros(p) if beta_init is None else np.asarray(beta_init).copy()
    if beta.shape[0] != p:
        raise ValueError(f"beta_init must have length p={p}. Got {beta.shape[0]}.")
    sigma2 = float(sigma2_init)
    if sigma2 <= 0:
        raise ValueError("sigma2_init must be > 0.")

    # residual bookkeeping
    r = y - X @ beta
    rss = float(r @ r)

    def log_lik_from_rss(rss_val, sigma2_val):
        # Gaussian iid
        return -0.5 * rss_val / sigma2_val - 0.5 * n * np.log(sigma2_val)

    # current logs
    lp_beta = log_prior_beta(beta, prior, prior_kwargs, intercept_idx=intercept_idx)
    ll = log_lik_from_rss(rss, sigma2)
    lp_s2 = log_prior_sigma2(sigma2, sigma2_prior, sigma2_prior_kwargs)

    kept_beta, kept_sig = [], []

    #  slopes proposal covariance init 
    if cov_start == "xtx":
        XtX_s = Xs.T @ Xs
        Sigma_s = np.linalg.inv(XtX_s + ridge * np.eye(d))
    elif cov_start == "identity":
        Sigma_s = np.eye(d)
    else:
        raise ValueError("cov_start must be 'xtx' or 'identity'.")

    def _chol_pd(A):
        A = 0.5 * (A + A.T)
        jitter = cov_ridge
        for _ in range(max_jitter_tries):
            try:
                return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))

    L_s = _chol_pd(Sigma_s)

    # online cov stats for slopes (burn-in)
    mean_s = beta[slopes_idx].copy()
    C_s = np.zeros((d, d), dtype=float)
    n_cov = 1
    cov_path = []

    # slopes scale adaptation
    log_scale = np.log(float(proposal_scale))
    n_adapt = 0
    acc_slopes = 0
    acc_block = 0
    scale_path = []

    # sigma2 MH adaptation (in log sigma2)
    log_s2 = np.log(sigma2)
    log_sig_step = np.log(float(sigma2_prop_scale))
    n_adapt_s2 = 0
    acc_s2 = 0
    acc_s2_block = 0
    s2_scale_path = []

    t0 = time.perf_counter()
    last_t = t0
    last_i = 0

    def _should_log(i):
        return bool(progress) and (progress_every not in (None, 0)) and ((i + 1) % int(progress_every) == 0)

    for t in range(n_draws):

        # -- Intercept update ---
        if gibbs_intercept:
            # ONLY valid if prior doesn't depend on intercept (flat / excluded)
            bs = beta[slopes_idx]
            r_no0 = y - Xs @ bs
            mean_b0 = float((x0 @ r_no0) / x0Tx0)
            var_b0 = sigma2 / x0Tx0
            b0_new = mean_b0 + np.sqrt(var_b0) * rng.standard_normal()

            delta0 = b0_new - beta[intercept_idx]
            if delta0 != 0.0:
                beta[intercept_idx] = b0_new
                r = r - x0 * delta0
                rss = float(r @ r)

                lp_beta = log_prior_beta(beta, prior, prior_kwargs, intercept_idx=intercept_idx)
                ll = log_lik_from_rss(rss, sigma2)
                # lp_s2 unchanged

        # -- MH for slopes ---
        bs = beta[slopes_idx]
        proposal_scale_t = float(np.exp(log_scale))
        sd_opt = 2.38 / np.sqrt(d) if d > 0 else 1.0

        z = rng.standard_normal(d)
        bs_prop = bs + proposal_scale_t * sd_opt * (L_s @ z)

        beta_prop = beta.copy()
        beta_prop[slopes_idx] = bs_prop

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
            acc_slopes += 1
            acc_block += 1


        # -- Adapt covariance for slopes (burn-in) ---
        if adapt_cov and (t < burn_in_int):
            bs_curr = beta[slopes_idx]
            n_cov += 1
            delta = bs_curr - mean_s
            mean_s = mean_s + delta / n_cov
            C_s = C_s + np.outer(delta, (bs_curr - mean_s))

            if (t + 1 >= cov_start_at) and ((t + 1) % cov_update_every == 0) and (n_cov > 2):
                emp_cov = C_s / (n_cov - 1)
                emp_cov = 0.5 * (emp_cov + emp_cov.T) + cov_ridge * np.eye(d)
                L_s = _chol_pd(emp_cov)
                if len(cov_path) < 50:
                    cov_path.append(emp_cov.copy())


        # -- Adapt slopes scale (burn-in) ---
        if adapt_scale and (t < burn_in_int) and (t + 1 >= adapt_start) and ((t + 1) % adapt_every == 0):
            n_adapt += 1
            step = adapt_gain / np.sqrt(n_adapt)

            acc_rate_block = acc_block / adapt_every
            log_scale = log_scale + step * (acc_rate_block - target_accept)

            new_scale = float(np.exp(log_scale))
            new_scale = min(adapt_max_scale, max(adapt_min_scale, new_scale))
            log_scale = np.log(new_scale)

            scale_path.append(new_scale)
            acc_block = 0


        # -- MH for sigma2 via theta=log(sigma2) ---
        # target in theta includes Jacobian: +theta
        step_s2 = float(np.exp(log_sig_step))
        theta_prop = log_s2 + step_s2 * rng.standard_normal()
        sigma2_prop = float(np.exp(theta_prop))

        ll_s2_prop = log_lik_from_rss(rss, sigma2_prop)
        lp_s2_prop = log_prior_sigma2(sigma2_prop, sigma2_prior, sigma2_prior_kwargs)

        log_post_theta_prop = ll_s2_prop + lp_s2_prop + theta_prop  # +theta = Jacobian
        log_post_theta_curr = ll + lp_s2 + log_s2

        log_alpha_s2 = log_post_theta_prop - log_post_theta_curr

        if np.isfinite(log_alpha_s2) and (np.log(rng.random()) < log_alpha_s2):
            sigma2 = sigma2_prop
            log_s2 = theta_prop
            ll = ll_s2_prop
            lp_s2 = lp_s2_prop
            acc_s2 += 1
            acc_s2_block += 1

        # -- Adapt sigma2 scale (burn-in) ---
        if adapt_sigma2_scale and (t < burn_in_int) and (t + 1 >= adapt_sigma2_start) and ((t + 1) % adapt_sigma2_every == 0):
            n_adapt_s2 += 1
            step = adapt_sigma2_gain / np.sqrt(n_adapt_s2)

            acc_rate_block = acc_s2_block / adapt_sigma2_every
            log_sig_step = log_sig_step + step * (acc_rate_block - target_accept_sigma2)

            new_step = float(np.exp(log_sig_step))
            new_step = min(sigma2_prop_max, max(sigma2_prop_min, new_step))
            log_sig_step = np.log(new_step)

            s2_scale_path.append(new_step)
            acc_s2_block = 0

        # -- store ---
        if t >= burn_in_int and ((t - burn_in_int) % thinning == 0):
            kept_beta.append(beta.copy())
            kept_sig.append(sigma2)

        # -- logs ---
        if _should_log(t):
            now = time.perf_counter()
            block_time = now - last_t
            block_iters = (t + 1) - last_i
            it_s = block_iters / block_time if block_time > 0 else np.nan

            msg = (
                f"[{t+1:>7}/{n_draws}] "
                f"acc_slopes={acc_slopes/(t+1):.3f} "
                f"acc_s2={acc_s2/(t+1):.3f} "
                f"sigma2={sigma2:.4g} "
                f"scale_slopes={float(np.exp(log_scale)):.3g} "
                f"step_logsig2={float(np.exp(log_sig_step)):.3g}")
            
            if show_time:
                remaining = n_draws - (t + 1)
                eta = remaining / it_s if (it_s and np.isfinite(it_s) and it_s > 0) else np.nan
                msg += f" | {it_s:,.1f} it/s | block={block_time:.2f}s | ETA≈{eta/60:.1f} min"
            print(msg)
            last_t = now
            last_i = t + 1

    beta_post = np.vstack(kept_beta) if kept_beta else np.empty((0, p))
    sigma_post = np.array(kept_sig) if kept_sig else np.empty((0,))

    info = {
        "n_draws": n_draws,
        "burn_in": burn_in_int,
        "thinning": thinning,
        "n_kept": beta_post.shape[0],
        "acc_rate_slopes": acc_slopes / n_draws,
        "acc_rate_sigma2": acc_s2 / n_draws,
        "final_slopes_scale": float(np.exp(log_scale)),
        "final_logsig2_step": float(np.exp(log_sig_step)),
        "slopes_scale_path": np.array(scale_path) if scale_path else None,
        "logsig2_step_path": np.array(s2_scale_path) if s2_scale_path else None,
        "adapt_cov": adapt_cov,
        "cov_snapshots": cov_path if cov_path else None,
        "prior_beta": prior,
        "prior_sigma2": sigma2_prior,
        "seed": seed,
        "runtime_sec": time.perf_counter() - t0}

    if return_info:
        return beta_post, sigma_post, info
    return beta_post, sigma_post

