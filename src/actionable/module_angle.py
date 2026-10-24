"""Optimal relative angle between grid modules: our implementation of the paper's Appendix K (Fig. 17).

Recipe of Appendix K (our reading of eqs. 99-102):
  1. A module with many neurons has a frequency spectrum  N(k | 0, 1/mu) x DC(k | 1/nu): a hexagonal lattice of
     spacing 1/nu in frequency space under a Gaussian envelope of width 1/mu  (mu = grid-peak width, nu = lattice
     length-scale of the grid in space; the paper drops factors of 2 pi in these units).
  2. Every lattice point 'repels' other frequencies over a region of size ~ 1/(2L) ~ 1 (units L = 1): the lattice is
     convolved with a unit-width Gaussian  ->  smoothed 2-D density rho_2(k).
  3. Standing at the origin and looking out at angle psi:  rho(psi) = int_0^inf rho_2(k_r cos psi, k_r sin psi) k_r dk_r.
  4. Overlap of two modules rotated by dpsi:  O(dpsi) = int rho(psi) rho(psi + dpsi) dpsi;  the predicted offset is
     the dpsi that minimises O.  For three modules: the pair of offsets (d1, d2) that minimises
     int rho(psi) rho(psi + d1) rho(psi + d1 + d2) dpsi.
The radial integral of a Gaussian blob has a closed form (see `_blob_density`), so no numerical integration is needed.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr


def hex_lattice(spacing: float, k_max: float) -> np.ndarray:
    """All points a*(i e1 + j e2) (e1 = (1, 0), e2 = (1/2, sqrt(3)/2)) with 0 < |p| <= k_max (origin excluded)."""
    n = int(np.ceil(k_max / spacing * 1.2)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    P = spacing * np.stack([i + 0.5 * j, (np.sqrt(3) / 2) * j], -1).reshape(-1, 2)
    r = np.hypot(P[:, 0], P[:, 1])
    return P[(r > 1e-9) & (r <= k_max)]


def _blob_density(psi: np.ndarray, p: np.ndarray, s: float) -> np.ndarray:
    """int_0^inf N2(r u(psi) - p; s^2 I) r dr for one lattice point p (closed form)."""
    rp = np.hypot(p[0], p[1])
    phi = np.arctan2(p[1], p[0])
    a = rp * np.cos(psi - phi)
    return (np.exp(-(rp ** 2 - a ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 2)
            * (s ** 2 * np.exp(-a ** 2 / (2 * s ** 2)) + a * s * np.sqrt(2 * np.pi) * ndtr(a / s)))


def angular_density(mu: float, nu: float, n_psi: int = 720, smooth: float = 1.0, k_max_sd: float = 4.0,
                    orientation: float = 0.0, chunk: int = 3000) -> np.ndarray:
    """rho(psi) on n_psi angles in [-pi, pi): smoothed hexagonal lattice (spacing 1/nu) x Gaussian envelope (sd 1/mu)."""
    psi = np.linspace(-np.pi, np.pi, n_psi, endpoint=False)
    P = hex_lattice(1.0 / nu, k_max_sd / mu)
    if orientation:
        c, s_ = np.cos(orientation), np.sin(orientation)
        P = P @ np.array([[c, s_], [-s_, c]])
    rp_all = np.hypot(P[:, 0], P[:, 1])
    w_all = np.exp(-0.5 * (rp_all * mu) ** 2)
    phi_all = np.arctan2(P[:, 1], P[:, 0])
    rho = np.zeros(n_psi)
    s = smooth
    for i in range(0, len(P), chunk):                       # vectorised closed form of _blob_density over lattice points
        rp, phi, w = rp_all[i:i + chunk, None], phi_all[i:i + chunk, None], w_all[i:i + chunk, None]
        a = rp * np.cos(psi[None, :] - phi)
        dens = (np.exp(-(rp ** 2 - a ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 2)
                * (s ** 2 * np.exp(-a ** 2 / (2 * s ** 2)) + a * s * np.sqrt(2 * np.pi) * ndtr(a / s)))
        rho += (w * dens).sum(0)
    return rho


def overlap_curve(rho: np.ndarray):
    """O(dpsi) = int rho(psi) rho(psi + dpsi) dpsi for dpsi = 2 pi m / n  (circular correlation); returns (dpsi_deg, O)."""
    n = len(rho)
    O = np.real(np.fft.ifft(np.conj(np.fft.fft(rho)) * np.fft.fft(rho))) * (2 * np.pi / n)
    d = np.degrees(np.arange(n) * 2 * np.pi / n)
    d = np.where(d > 180, d - 360, d)
    order = np.argsort(d)
    return d[order], O[order]


def optimal_offset(mu: float, nu: float, n_psi: int = 1440, smooth: float = 1.0):
    """Predicted optimal offset between two identical modules, in degrees, searched over (0, 30] deg
    (the lattice has 60 deg and mirror symmetry).  Returns (offset_deg, overlap_at_offset, overlap_aligned)."""
    d, O = overlap_curve(angular_density(mu, nu, n_psi, smooth))
    m = (d > 0.0) & (d <= 30.0)
    i = np.argmin(np.where(m, O, np.inf))
    return float(d[i]), float(O[i]), float(O[np.argmin(np.abs(d))])


def three_module_offsets(mu: float, nu: float, n_psi: int = 360, smooth: float = 1.0, max_deg: float = 30.0):
    """Offsets (d1, d2) in degrees (0 < d1, d2 <= max_deg) minimising  int rho(psi) rho(psi+d1) rho(psi+d1+d2) dpsi."""
    rho = angular_density(mu, nu, n_psi, smooth)
    step = 360.0 / n_psi
    ks = np.arange(1, int(max_deg / step) + 1)
    best, arg = np.inf, (np.nan, np.nan)
    for a in ks:
        r1 = np.roll(rho, -a)
        for b in ks:
            v = float(np.sum(rho * r1 * np.roll(rho, -(a + b))))
            if v < best:
                best, arg = v, (a * step, b * step)
    return arg, best * 2 * np.pi / n_psi
