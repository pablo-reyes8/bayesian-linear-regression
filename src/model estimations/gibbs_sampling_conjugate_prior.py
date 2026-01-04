import numpy as np
from scipy.stats import invgamma
from numpy.linalg import inv as inv 

def Gibbs_Sampling(m0, V0, a0, b0, n, p, X, y, n_draws, burn_in=0, thinning=1, seed=None):
    """
    Gibbs sampler for Bayesian linear regression with conjugate priors,
    micro-optimized:
      - pre-invert constant matrices outside the loop
      - sample beta via Cholesky rather than np.random.multivariate_normal
    Returns:
      beta_post  : array of shape (n_samples, p)
      sigma2_post: array of shape (n_samples,)
    """

    # RNG
    rng = np.random.default_rng(seed)
    
    # Precompute constant inverses & cross-products
    V0_inv = np.linalg.inv(V0)
    XtX    = X.T @ X
    Xty    = X.T @ y

    posterior_beta  = []
    posterior_sigma = []

    # Initialize
    beta   = m0.copy()
    sigma2 = 1.0

    for i in range(n_draws):
        #Posterior precision & mean for beta | sigma2, y
        Vn = np.linalg.inv(XtX + V0_inv)           # p×p
        mn = Vn @ (Xty + V0_inv @ m0)               # p-vector

        #Sample beta using Cholesky for stability
        L = np.linalg.cholesky(sigma2 * Vn)         # lower-triangular
        z = rng.standard_normal(p)
        beta = mn + L @ z

        #Update Inv-Gamma parameters for sigma2 | beta, y
        an = a0 + n/2
        resid = y - X @ beta
        bn = b0 + 0.5 * (resid @ resid + (beta - m0) @ V0_inv @ (beta - m0))

        #Sample sigma2
        sigma2 = invgamma.rvs(a=an, scale=bn, random_state=rng)

        #Store after burn-in
        if i >= burn_in and ((i - burn_in) % thinning == 0):
            posterior_beta.append(beta.copy())
            posterior_sigma.append(sigma2)

    beta_post  = np.vstack(posterior_beta)   
    sigma_post = np.array(posterior_sigma)   

    return beta_post, sigma_post