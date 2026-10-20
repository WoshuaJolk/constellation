"""Attempt C: exact finite-bandwidth SO(3) factorization, with adjoint.

This combines established spherical harmonic rotation and NUFFT identities.
It does not itself solve spherical reprojection, radial covering, realistic
image bandwidth, or orientation-dependent reception/emission time.
"""

from math import factorial, sqrt
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss

try:
    from scipy.special import sph_harm_y
except ImportError:  # SciPy <1.15
    from scipy.special import sph_harm

    def sph_harm_y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)


def wigner_d(l, m, n, beta):
    """Low-degree factorial formula; stable-recursion production code needed."""
    beta = np.asarray(beta)
    answer = np.zeros_like(beta, dtype=float)
    factor = sqrt(factorial(l+n)*factorial(l-n)*factorial(l+m)*factorial(l-m))
    for k in range(max(0, n-m), min(l+n, l-m)+1):
        coefficient = (-1)**(m-n+k)*factor/(factorial(l+n-k)*factorial(k)*factorial(m-n+k)*factorial(l-m-k))
        answer += coefficient*np.cos(beta/2)**(2*l+n-m-2*k)*np.sin(beta/2)**(m-n+2*k)
    return answer


def rotations(alpha, beta, gamma):
    """Active right-handed Rz(alpha) Ry(beta) Rz(gamma), broadcasting angles."""
    a, b, g = np.broadcast_arrays(alpha, beta, gamma)
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    return np.stack([ca*cb*cg-sa*sg, -ca*cb*sg-sa*cg, ca*sb,
                     sa*cb*cg+ca*sg, -sa*cb*sg+ca*cg, sa*sb,
                     -sb*cg, sb*sg, cb], axis=-1).reshape(a.shape+(3, 3))


def real_coefficients(rng, frames, degree):
    coeff = np.zeros((frames, (degree+1)**2), dtype=complex)
    for l in range(degree+1):
        coeff[:, l*l+l] = rng.normal(size=frames)/(l+1)
        for m in range(1, l+1):
            value = (rng.normal(size=frames)+1j*rng.normal(size=frames))/(l+1)
            coeff[:, l*l+l+m] = value
            coeff[:, l*l+l-m] = (-1)**m*value.conj()
    return coeff


def sample_sky(coefficients, directions):
    """Reference evaluation, one sky field per time; directions (R,T,3)."""
    directions = np.asarray(directions)
    degree = int(round(np.sqrt(coefficients.shape[1])))-1
    theta = np.arctan2(np.hypot(directions[..., 0], directions[..., 1]), directions[..., 2])
    phi = np.arctan2(directions[..., 1], directions[..., 0])
    result = np.zeros(directions.shape[:2], dtype=complex)
    for l in range(degree+1):
        for m in range(-l, l+1):
            result += coefficients[None, :, l*l+l+m]*sph_harm_y(l, m, theta, phi)
    return result


def evaluate_sky(coefficients, directions):
    return sample_sky(coefficients, directions).sum(axis=1)


def evaluate_directional(coefficients, orientations):
    """Slow independent Euler-angle reference for local Wigner coefficients."""
    degree = int(round(np.sqrt(coefficients.shape[1])))-1
    spin = (coefficients.shape[2]-1)//2
    beta = np.arccos(np.clip(orientations[..., 2, 2], -1., 1.))
    alpha = np.arctan2(orientations[..., 1, 2], orientations[..., 0, 2])
    gamma = np.arctan2(orientations[..., 2, 1], -orientations[..., 2, 0])
    if np.any(np.abs(np.sin(beta)) < 1e-12):
        raise ValueError("Reference Euler decomposition is singular; use nonsingular test orientations")
    result = np.zeros(orientations.shape[:2], dtype=complex)
    for l in range(degree+1):
        for m in range(-l, l+1):
            for s in range(-min(l, spin), min(l, spin)+1):
                result += coefficients[None, :, l*l+l+m, s+spin]*np.exp(1j*(m*alpha+s*gamma))*wigner_d(l, m, s, beta)
    return result.sum(axis=1)


def reproject(coefficients, radii, observers, degree):
    """Attempt moving-observer spherical reprojection at shared emission times.

    This quadrature is an experiment, with no asserted tail/error certificate.
    In particular it does NOT fix orientation-dependent light-time coupling.
    """
    input_degree = int(round(np.sqrt(coefficients.shape[1])))-1
    order = 2*(input_degree+degree)+8
    z, weights = leggauss(order)
    theta, phi = np.meshgrid(np.arccos(z), 2*np.pi*np.arange(2*order)/(2*order), indexing="ij")
    theta, phi = theta.ravel(), phi.ravel()
    weights = np.repeat(weights, 2*order)*(2*np.pi/(2*order))
    unit = np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
    directions = unit[:, None, :]*np.asarray(radii)[None, :, None]-observers[None]
    if np.any(np.linalg.norm(directions, axis=2) < 1e-8):
        raise ValueError("Reprojection crosses observer")
    samples = sample_sky(coefficients, directions)
    basis = np.array([sph_harm_y(l, m, theta, phi) for l in range(degree+1) for m in range(-l, l+1)])
    return samples.T @ (basis.conj()*weights).T


class RotationTransform:
    def __init__(self, phases, degree, backend="direct"):
        start = perf_counter()
        self.phases = np.asarray(phases, float)
        if self.phases.ndim != 1 or not len(self.phases) or not np.all(np.isfinite(self.phases)):
            raise ValueError("phases must be a finite vector")
        if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= 12:
            raise ValueError("Research factorial Wigner implementation supports degree 0..12")
        if backend not in {"direct", "nufft"}:
            raise ValueError("backend must be direct or nufft")
        self.degree, self.backend = degree, backend
        self.width = 2*degree+1
        self.modes = np.arange(-degree, degree+1)
        self.phase_matrix = np.exp(1j*self.phases[:, None]*self.modes) if backend == "direct" else None
        z, self.weights = leggauss(degree+1)
        self.beta = np.arccos(z)
        self.alpha = self.gamma = 2*np.pi*np.arange(self.width)/self.width
        self.d, self.equator = [], []
        for l in range(degree+1):
            d = np.empty((degree+1, 2*l+1, 2*l+1))
            for i, m in enumerate(range(-l, l+1)):
                for j, n in enumerate(range(-l, l+1)):
                    d[:, i, j] = wigner_d(l, m, n, self.beta)
            self.d.append(d)
            self.equator.append(sph_harm_y(l, np.arange(-l, l+1), np.pi/2, 0))
        self.construction_seconds = perf_counter()-start

    def _moments(self, coefficients):
        if self.backend == "direct":
            return coefficients.T @ self.phase_matrix
        import finufft
        points = (self.phases+np.pi) % (2*np.pi)-np.pi
        return finufft.nufft1d1(points, np.ascontiguousarray(coefficients.T),
                               n_modes=self.width, isign=1, eps=1e-12, nthreads=1)

    def coefficients(self, data):
        data = np.asarray(data, complex)
        if data.shape != (len(self.phases), (self.degree+1)**2) or not np.all(np.isfinite(data)):
            raise ValueError("Data must be finite (times,(degree+1)**2) harmonic coefficients")
        moments = self._moments(data)
        return [moments[l*l:(l+1)**2, self.degree-l:self.degree+l+1]*self.equator[l][None]
                for l in range(self.degree+1)]

    def forward(self, data):
        coefficients = self.coefficients(data)
        return self.synthesize(coefficients)

    def synthesize(self, coefficients):
        fourier = np.zeros((self.degree+1, self.width, self.width), dtype=complex)
        for l, c in enumerate(coefficients):
            indices = np.arange(-l, l+1) % self.width
            fourier[:, indices[:, None], indices[None, :]] += self.d[l]*c[None]
        return np.fft.ifft2(fourier, axes=(-2, -1))*self.width**2

    def adjoint(self, data):
        """Euclidean complex adjoint of forward; deliberately not called inverse."""
        data = np.asarray(data, complex)
        if data.shape != (self.degree+1, self.width, self.width):
            raise ValueError("Rotation sample shape mismatch")
        fourier = np.fft.fft2(data, axes=(-2, -1))
        moments = np.zeros(((self.degree+1)**2, self.width), dtype=complex)
        for l in range(self.degree+1):
            indices = np.arange(-l, l+1) % self.width
            c = np.sum(self.d[l]*fourier[:, indices[:, None], indices[None, :]], axis=0)
            moments[l*l:(l+1)**2, self.degree-l:self.degree+l+1] = c*self.equator[l].conj()[None]
        if self.backend == "direct":
            return (moments @ self.phase_matrix.conj().T).T
        import finufft
        points = (self.phases+np.pi) % (2*np.pi)-np.pi
        return finufft.nufft1d2(points, moments, isign=-1, eps=1e-12, nthreads=1).T

    def inverse_wigner(self, data):
        """Recover rotation coefficients, not the original time-separated images."""
        fourier = np.fft.fft2(data, axes=(-2, -1))
        output = []
        for l in range(self.degree+1):
            indices = np.arange(-l, l+1) % self.width
            output.append((2*l+1)/(2*self.width**2)*np.sum(
                self.weights[:, None, None]*self.d[l]*fourier[:, indices[:, None], indices[None, :]], axis=0))
        return output

    def grid_rotations(self):
        b, a, g = np.meshgrid(self.beta, self.alpha, self.gamma, indexing="ij")
        return rotations(a, b, g)


class DirectionalRotationTransform(RotationTransform):
    """Attempt C extension: retain local roll/velocity harmonics, not only sky.

    Each exposure supplies coefficients b[j,l,m,s] of its *local orientation*
    response. For Q_j=R Rz(phi_j) Ry(pi/2), composition of Wigner matrices gives
    C[l,m,n]=sum_s d[l,n,s](pi/2) sum_j b[j,l,m,s] exp(i*n*phi_j).
    The inner sums are NUFFTs. Constructing b at astronomical bandwidth is an
    unresolved part of the full problem; this class does not hide that cost.
    """

    def __init__(self, phases, degree, spin_limit, backend="direct"):
        super().__init__(phases, degree, backend)
        if isinstance(spin_limit, bool) or not isinstance(spin_limit, int) or not 0 <= spin_limit <= degree:
            raise ValueError("spin_limit must be in 0..degree")
        self.spin_limit = spin_limit

    def coefficients(self, data):
        data = np.asarray(data, complex)
        if data.shape != (len(self.phases), (self.degree+1)**2, 2*self.spin_limit+1):
            raise ValueError("Directional coefficients need (times,(degree+1)**2,2*spin_limit+1)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Nonfinite directional coefficients")
        output = [np.zeros((2*l+1, 2*l+1), complex) for l in range(self.degree+1)]
        for s in range(-self.spin_limit, self.spin_limit+1):
            if abs(s) and np.any(data[:, :s*s, s+self.spin_limit] != 0):
                raise ValueError("Coefficients with |spin|>degree must vanish")
            moments = self._moments(data[:, :, s+self.spin_limit])
            for l in range(abs(s), self.degree+1):
                rotation = np.array([wigner_d(l, n, s, np.pi/2) for n in range(-l, l+1)])
                output[l] += moments[l*l:(l+1)**2, self.degree-l:self.degree+l+1]*rotation[None]
        return output

    def adjoint(self, data):
        if np.shape(data) != (self.degree+1, self.width, self.width):
            raise ValueError("Rotation sample shape mismatch")
        fourier = np.fft.fft2(data, axes=(-2, -1))
        c = []
        for l in range(self.degree+1):
            indices = np.arange(-l, l+1) % self.width
            c.append(np.sum(self.d[l]*fourier[:, indices[:, None], indices[None, :]], axis=0))
        output = np.zeros((len(self.phases), (self.degree+1)**2, 2*self.spin_limit+1), complex)
        for s in range(-self.spin_limit, self.spin_limit+1):
            moments = np.zeros(((self.degree+1)**2, self.width), complex)
            for l in range(abs(s), self.degree+1):
                rotation = np.array([wigner_d(l, n, s, np.pi/2) for n in range(-l, l+1)])
                moments[l*l:(l+1)**2, self.degree-l:self.degree+l+1] = c[l]*rotation[None]
            if self.backend == "direct":
                output[:, :, s+self.spin_limit] = (moments @ self.phase_matrix.conj().T).T
            else:
                import finufft
                points = (self.phases+np.pi) % (2*np.pi)-np.pi
                output[:, :, s+self.spin_limit] = finufft.nufft1d2(points, moments, isign=-1, eps=1e-12, nthreads=1).T
        return output
