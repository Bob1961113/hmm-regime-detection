"""
Online forward-only filtering for live regime inference.

The PoC used `model.predict` (Viterbi) and `predict_proba` — both run the whole
sequence forward AND backward, so the estimate for any bar uses *future* bars.
That is fine for offline analysis but dishonest for a live detector that must not
retroactively rewrite history.

Here we implement the scaled forward recursion in log-space:

    alpha_t(j) ∝ b_j(o_t) * sum_i alpha_{t-1}(i) * A[i, j]

which gives the filtered posterior P(state_t = j | o_1..o_t) — only past and
present. It is O(states^2) per bar, i.e. instant. Call `reset()` at the start of
each session so the chain restarts from the model's initial distribution.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class OnlineFilter:
    def __init__(self, model):
        self.n = model.n_components
        with np.errstate(divide="ignore"):
            self.log_startprob = np.log(model.startprob_)
            self.log_transmat = np.log(model.transmat_)
        covars = model.covars_           # (n, d, d) for full covariance
        self.emissions = [
            multivariate_normal(mean=model.means_[i], cov=covars[i], allow_singular=True)
            for i in range(self.n)
        ]
        self.log_alpha: np.ndarray | None = None

    def reset(self) -> None:
        """Restart filtering from the model's initial state distribution."""
        self.log_alpha = None

    def update(self, x: np.ndarray) -> np.ndarray:
        """Advance one bar. `x` is the scaled feature vector. Returns the filtered
        posterior distribution over states (sums to 1)."""
        log_b = np.array([self.emissions[i].logpdf(x) for i in range(self.n)])
        if self.log_alpha is None:
            log_alpha = self.log_startprob + log_b
        else:
            pred = logsumexp(self.log_alpha[:, None] + self.log_transmat, axis=0)
            log_alpha = pred + log_b
        # normalize (scaled forward algorithm) to keep a proper distribution
        self.log_alpha = log_alpha - logsumexp(log_alpha)
        return np.exp(self.log_alpha)
