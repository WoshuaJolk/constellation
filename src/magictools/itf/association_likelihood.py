"""Conditional astrometric likelihoods, not calibrated association probabilities.

Independent 2D errors plus a shared 2D offset per night/site. The offset is
defined in a common tangent frame for each group. Supplied uncertainty policies
must be calibrated on separate controls before interpreting population coverage.
"""
import hashlib
import numpy as np
from scipy.special import logsumexp

POLICY = 'association-likelihood-v1'


class NightCovariance:
    """C = blockdiag(D_i) + Z blockdiag(B_g) Z.T; O(N) block solves."""
    def __init__(self, random_covariance, groups, shared_covariance):
        d = np.asarray(random_covariance, float)
        groups = np.asarray(groups)
        if d.ndim != 3 or d.shape[1:] != (2, 2) or len(d) == 0 or groups.shape != (len(d),):
            raise ValueError('Expected N nonempty 2D covariances and N group labels')
        if not np.isfinite(d).all() or not np.allclose(d, d.swapaxes(1, 2)):
            raise ValueError('Random covariances must be finite and symmetric')
        chol = np.linalg.cholesky(d)  # Reject non-positive-definite policies.
        self.random_whitener = np.linalg.inv(chol)
        self.inverse = np.linalg.inv(d)
        self.logdet = float(np.linalg.slogdet(d)[1].sum())
        labels, inverse = np.unique(groups, return_inverse=True)
        b = np.asarray(shared_covariance, float)
        b = np.broadcast_to(b, (len(labels), 2, 2))
        if not np.isfinite(b).all() or not np.allclose(b, b.swapaxes(1, 2)):
            raise ValueError('Shared covariances must be finite and symmetric')
        eigen, vectors = np.linalg.eigh(b)
        if np.any(eigen < -1e-14):
            raise ValueError('Shared covariance must be positive semidefinite')
        factors = vectors * np.sqrt(np.maximum(eigen, 0))[:, None, :]
        self.identity = dict(policy=POLICY, calibrated=False,
            covariance_sha256=hashlib.sha256(d.tobytes()+inverse.astype('<i8').tobytes()+b.tobytes()).hexdigest())
        self.blocks = []
        self.whitening_blocks = []
        for g, factor in enumerate(factors):
            ids = np.flatnonzero(inverse == g)
            precision = self.inverse[ids].sum(axis=0)
            small = np.eye(2) + factor.T @ precision @ factor
            self.logdet += float(np.linalg.slogdet(small)[1])
            self.blocks.append((ids, factor, np.linalg.inv(small)))
            projected = (self.random_whitener[ids] @ factor).reshape(-1, 2)
            u, singular, _ = np.linalg.svd(projected, full_matrices=False)
            self.whitening_blocks.append((ids, u, 1/np.sqrt(1+singular**2)-1))
        self.n = len(d)

    def whiten(self, rhs):
        """Apply W with W.T W = C^-1, to residuals or orbit Jacobians."""
        rhs = np.asarray(rhs, float)
        if rhs.shape[0] != 2*self.n or not np.isfinite(rhs).all():
            raise ValueError('RHS must contain finite stacked two-coordinate observations')
        vector = rhs.ndim == 1
        out = self.random_whitener @ rhs.reshape(self.n, 2, -1)
        for ids, u, shrink in self.whitening_blocks:
            block = out[ids].reshape(2*len(ids), -1)
            block += u @ (shrink[:,None] * (u.T @ block))
            out[ids] = block.reshape(len(ids), 2, -1)
        result = out.reshape(2*self.n, -1)
        return result[:, 0] if vector else result

    def solve(self, rhs):
        rhs = np.asarray(rhs, float)
        if rhs.shape[0] != 2*self.n or not np.isfinite(rhs).all():
            raise ValueError('RHS must contain finite stacked two-coordinate observations')
        vector = rhs.ndim == 1
        r = rhs.reshape(self.n, 2, -1)
        out = self.inverse @ r
        for ids, factor, small_inverse in self.blocks:
            correction = factor @ small_inverse @ factor.T @ out[ids].sum(axis=0)
            out[ids] -= self.inverse[ids] @ correction
        result = out.reshape(2*self.n, -1)
        return result[:, 0] if vector else result

    def predictive(self, residual, state_projection=None):
        """Normalized density with C + A A.T, including cross-observation terms.

        A must be the projection of a training-only state covariance. Training
        and validation groups must not share a latent nightly offset; otherwise
        the joint conditioning needs additional cross-covariance terms.
        """
        r = np.asarray(residual, float).reshape(-1)
        if len(r) != 2*self.n:
            raise ValueError('Residual dimensions differ from observation policy')
        a = np.zeros((len(r), 0)) if state_projection is None else np.asarray(state_projection, float)
        if a.ndim != 2 or a.shape[0] != len(r) or not np.isfinite(a).all():
            raise ValueError('Invalid state projection')
        # Whitening + thin SVD avoids subtracting nearly equal large quadratic
        # forms when short-arc state uncertainty is extremely broad.
        wr = self.whiten(r)
        statistic = float(wr @ wr)
        logdet = self.logdet
        if a.shape[1]:
            u, singular, _ = np.linalg.svd(self.whiten(a), full_matrices=False)
            projection = u.T @ wr
            orthogonal = wr-u@projection
            statistic = float(orthogonal@orthogonal + np.sum((projection/np.hypot(1.,singular))**2))
            logdet += float(2*np.log(np.hypot(1.,singular)).sum())
        return dict(chi2=statistic, log_determinant=logdet, dimensions=len(r),
                    log_density=-.5*(statistic+logdet+len(r)*np.log(2*np.pi)),
                    calibrated=False, policy=POLICY)


def predictive_background_score(mode_log_densities, mode_weights, background_log_density):
    """Compare densities in the SAME measured coordinates and selection domain.

    Weights must be justified externally; normalized best-fit likelihoods are
    not posterior masses. Unknown/zero background remains unresolved.
    """
    values = np.asarray(mode_log_densities, float)
    weights = np.asarray(mode_weights, float)
    if (values.ndim != 1 or not len(values) or values.shape != weights.shape or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
        raise ValueError('Supply finite mode densities and normalized justified weights')
    if background_log_density is None or not np.isfinite(background_log_density):
        return dict(status='unresolved', score=None, calibrated=False,
                    reason='Background density unavailable; no zero-density shortcut')
    use = weights > 0
    density = float(logsumexp(values[use] + np.log(weights[use])))
    return dict(status='ranking_feature', score=density-float(background_log_density),
                predictive_log_density=density, background_log_density=float(background_log_density),
                calibrated=False, association_probability=None)
