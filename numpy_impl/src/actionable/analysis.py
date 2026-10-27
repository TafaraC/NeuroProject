"""Analysis tools: rate maps, gridness scores, module clustering, lattice statistics.

Two ways of looking at a trained code g(x) = a0 + sum_d a_d sin(k_d.x) + b_d cos(k_d.x):

* *empirical*: rate maps on a finite window and the usual spatial autocorrelogram / gridness score;
* *analytic*: because distinct plane waves are orthogonal, the infinite-plane autocorrelogram of a neuron is
  AC(dx) = sum_d P_d cos(k_d . dx) / sum_d P_d with P_d = a_d^2 + b_d^2 (independent of the phases), so a whole
  module (cells that are translated copies of one another) shares ONE autocorrelogram and no window is involved.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage, optimize
from scipy.cluster.hierarchy import linkage, fcluster
from .core import encode, Params


def norms_from_samples(p: Params, n: int = 20000, L: float = 1.0, seed: int = 123) -> np.ndarray:
    """Per-neuron RMS rate over samples from p(x) (the paper's norm constraint target = 1)."""
    rng = np.random.default_rng(seed)
    dim = p["K"].shape[1]
    X = rng.normal(0, L, (n, dim)).astype(p["K"].dtype)
    return np.sqrt((encode(p, X) ** 2).mean(0))


def rate_maps(p: Params, half_width: float = 2.5, res: int = 64, center=(0.0, 0.0)):
    """Normalised rate maps g~_n(x) on a res x res grid -> (N, res, res)."""
    xs = np.linspace(-half_width, half_width, res) + center[0]
    ys = np.linspace(-half_width, half_width, res) + center[1]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel()], 1).astype(p["K"].dtype)
    G = encode(p, X) / norms_from_samples(p)
    return G.T.reshape(-1, res, res).astype(np.float64)


# --------------------------------------------------------------------------- gridness
def autocorr2d(rm: np.ndarray) -> np.ndarray:
    """Pearson spatial autocorrelogram with proper handling of partial overlap."""
    a = rm - rm.mean()
    n = rm.shape[0]
    ones = np.ones_like(a)
    F = lambda z: np.fft.rfft2(z, (2 * n, 2 * n))
    Ft = lambda z: np.fft.irfft2(z, (2 * n, 2 * n))
    cnt = np.fft.fftshift(Ft(F(ones) * np.conj(F(ones))))
    sxy = np.fft.fftshift(Ft(F(a) * np.conj(F(a))))
    sx = np.fft.fftshift(Ft(F(a) * np.conj(F(ones))))
    sy = np.fft.fftshift(Ft(F(ones) * np.conj(F(a))))
    sxx = np.fft.fftshift(Ft(F(a * a) * np.conj(F(ones))))
    syy = np.fft.fftshift(Ft(F(ones) * np.conj(F(a * a))))
    cnt = np.maximum(cnt, 1)
    num = sxy - sx * sy / cnt
    den = np.sqrt(np.maximum(sxx - sx ** 2 / cnt, 1e-12) * np.maximum(syy - sy ** 2 / cnt, 1e-12))
    ac = num / den
    ac[np.round(cnt) < 20] = np.nan                       # discard tiny-overlap lags
    c = n  # centre index after fftshift of the 2n array is n
    return ac[c - (n - 1):c + n, c - (n - 1):c + n]


def gridness_from_ac(ac: np.ndarray, return_details: bool = False):
    """Sargolini-style gridness of an autocorrelogram: min(r60, r120) - max(r30, r90, r150),
    maximised over the outer radius of the annulus that excludes the central peak."""
    n = ac.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(yy - c, xx - c)
    # radial profile -> central-peak radius = first zero crossing / minimum
    prof = np.array([np.nanmean(ac[(rr >= r - 0.5) & (rr < r + 0.5)]) for r in range(1, c)])
    below = np.where(prof < 0)[0]
    r_in = (below[0] + 1) if len(below) else (int(np.nanargmin(prof)) + 1)
    r_in = max(r_in, 2)
    acf = np.where(np.isnan(ac), 0.0, ac)
    rots = {a: ndimage.rotate(acf, a, reshape=False, order=1, mode="constant") for a in (30, 60, 90, 120, 150)}
    best, best_r = -2.0, None
    for r_out in range(r_in + 3, c - 1):
        mask = (rr > r_in) & (rr <= r_out) & ~np.isnan(ac)
        if mask.sum() < 30:
            continue
        ref = acf[mask]
        if ref.std() < 1e-9:
            continue
        cs = {a: np.corrcoef(ref, rots[a][mask])[0, 1] for a in rots}
        g = min(cs[60], cs[120]) - max(cs[30], cs[90], cs[150])
        if g > best:
            best, best_r = g, r_out
    if return_details:
        return best, dict(r_in=r_in, r_out=best_r, ac=ac)
    return best


def gridness(rm: np.ndarray, return_details: bool = False):
    """Gridness of a (windowed) rate map via its empirical autocorrelogram."""
    return gridness_from_ac(autocorr2d(rm), return_details)


def analytic_autocorr(K: np.ndarray, power: np.ndarray, half_width: float, res: int = 121) -> np.ndarray:
    """Infinite-plane Pearson autocorrelogram of  g(x) = a0 + sum_d a_d sin(k_d.x) + b_d cos(k_d.x).
    AC(dx) = sum_d P_d cos(k_d . dx) / sum_d P_d  with P_d = a_d^2 + b_d^2 (see module docstring)."""
    xs = np.linspace(-half_width, half_width, res)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    D = np.stack([XX.ravel(), YY.ravel()], 1)
    ac = (np.cos(D @ K.T) @ power) / power.sum()
    return ac.reshape(res, res)


def squareness(rm: np.ndarray) -> float:
    """Same as gridness but for 90-degree symmetry (r90 - max(r45, r135)); > 0 => square-ish."""
    ac = autocorr2d(rm)
    n = ac.shape[0]; c = n // 2
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(yy - c, xx - c)
    prof = np.array([np.nanmean(ac[(rr >= r - 0.5) & (rr < r + 0.5)]) for r in range(1, c)])
    below = np.where(prof < 0)[0]
    r_in = max((below[0] + 1) if len(below) else 2, 2)
    acf = np.where(np.isnan(ac), 0.0, ac)
    rots = {a: ndimage.rotate(acf, a, reshape=False, order=1, mode="constant") for a in (45, 90, 135)}
    best = -2.0
    for r_out in range(r_in + 3, c - 1):
        mask = (rr > r_in) & (rr <= r_out) & ~np.isnan(ac)
        if mask.sum() < 30 or acf[mask].std() < 1e-9:
            continue
        cs = {a: np.corrcoef(acf[mask], rots[a][mask])[0, 1] for a in rots}
        best = max(best, cs[90] - max(cs[45], cs[135]))
    return best


# --------------------------------------------------------------------------- frequency space / modules
def freq_power(p: Params) -> np.ndarray:
    """P[d, n] = a_dn^2 + b_dn^2 : power of frequency d in neuron n (relative to its norm)."""
    s2 = norms_from_samples(p) ** 2
    return (p["A"] ** 2 + p["B"] ** 2) / s2


def cluster_modules(p: Params, thresh: float = 0.5, min_size: int = 2):
    """Cluster neurons into modules.  Cells in one module are translated copies of one
    another, so they share an identical power spectrum |a_d|^2+|b_d|^2 over frequencies.
    We hierarchically cluster the (L2-normalised) spectra with cosine distance; clusters
    smaller than `min_size` are labelled -1 (stray cells)."""
    P = freq_power(p).T                                      # (N, D)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    Z = linkage(Pn, method="average", metric="cosine")
    lab = fcluster(Z, t=thresh, criterion="distance") - 1
    ids, counts = np.unique(lab, return_counts=True)
    order = ids[np.argsort(-counts)]
    out = -np.ones_like(lab)
    k = 0
    for i in order:
        if (lab == i).sum() >= min_size:
            out[lab == i] = k; k += 1
    return out


# --------------------------------------------------------------------------- lattice read off the autocorrelogram
def spacing_hint(K: np.ndarray, power: np.ndarray, rel: float = 0.25, k_min: float = 0.8) -> float:
    """Grid spacing implied by the module's FUNDAMENTAL wave: the lowest-|k| wave with power >= `rel` x the
    strongest one among waves with |k| >= max(k_min, |k_strongest|/2).  The strongest wave can belong to a
    higher shell or a harmonic, and very slow waves (|k| < k_min, wavelength longer than the ~6-unit arena)
    are not part of any lattice, so neither is used directly."""
    kn = np.linalg.norm(K, axis=1)
    d0 = int(np.argmax(power))
    lo = max(k_min, 0.5 * kn[d0])
    ok = (power >= rel * power.max()) & (kn >= lo)
    k_f = kn[ok].min() if ok.any() else max(kn[d0], 1e-3)
    return float(4 * np.pi / (np.sqrt(3) * k_f))


def ac_lattice(K: np.ndarray, power: np.ndarray, min_peak: float = 0.05, res: int = 241, n_periods: float = 3.2):
    """Lattice statistics measured on the (analytic) autocorrelogram of a module the way experimentalists
    do on recorded cells: find the peaks round the central one, take the six nearest, and refine each peak
    position on the analytic autocorrelogram AC(d) = sum_k P_k cos(k.d) / sum_k P_k.

    Returns ac_spacing (mean distance of the six peaks), ac_hex_err_deg (largest deviation of the angle
    gaps between neighbouring peaks from 60 deg), ac_len_spread ((max-min)/mean distance),
    ac_orientation_deg (lattice orientation mod 60 deg, in [-30, 30)), ac_n_peaks, ac_window (half width).
    All are NaN when fewer than six peaks are found (e.g. stripe-like modules)."""
    K = np.asarray(K, float); power = np.asarray(power, float)
    nan = float("nan")
    sp = spacing_hint(K, power)
    R = n_periods * sp
    ac = ndimage.gaussian_filter(analytic_autocorr(K, power, R, res), 1.0)
    c = res // 2; scale = R / c
    mx = ndimage.maximum_filter(ac, size=9, mode="nearest")
    ii, jj = np.where((ac == mx) & (ac > min_peak))
    xy = np.stack([(ii - c) * scale, (jj - c) * scale], 1)
    r = np.hypot(xy[:, 0], xy[:, 1])
    h = ac[ii, jj]
    # an isolated peak has curvature in BOTH directions: along the ridge of a stripe pattern the autocorrelogram
    # is flat, so ridge points are not lattice points (minimum / maximum eigenvalue of -Hessian > 0.05)
    cs = np.cos(xy @ K.T) * power                                          # (n, D)
    hxx, hxy, hyy = (cs * K[:, 0] ** 2).sum(1), (cs * K[:, 0] * K[:, 1]).sum(1), (cs * K[:, 1] ** 2).sum(1)
    half_tr, rad = 0.5 * (hxx + hyy), np.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2)
    isolated = (half_tr - rad) > 0.05 * (half_tr + rad)
    outer = (r > 0.25 * sp) & (r < 0.9 * R) & isolated
    # keep only prominent peaks (>= 50 % of the highest peak away from the centre): faint secondary maxima
    # between the main peaks must not count as lattice points
    keep = outer & (h >= 0.5 * h[outer].max()) if outer.any() else outer
    out = dict(ac_n_peaks=int(keep.sum()), ac_spacing=nan, ac_hex_err_deg=nan, ac_len_spread=nan,
               ac_orientation_deg=nan, ac_window=float(R))
    if keep.sum() < 6:
        return out
    xy = xy[keep][np.argsort(r[keep])[:6]]
    f = lambda v: -float((np.cos(v @ K.T) @ power) / power.sum())
    pts = np.array([optimize.minimize(f, q, method="Nelder-Mead",
                                      options=dict(xatol=1e-4 * sp, fatol=1e-10, maxiter=400)).x for q in xy])
    d = np.hypot(pts[:, 0], pts[:, 1])
    th = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    ths = np.sort(th)
    gaps = np.diff(np.concatenate([ths, [ths[0] + 360.0]]))
    z = np.mean(np.exp(1j * np.radians(6.0 * th)))
    out.update(ac_spacing=float(d.mean()), ac_hex_err_deg=float(np.max(np.abs(gaps - 60.0))),
               ac_len_spread=float((d.max() - d.min()) / d.mean()),
               ac_orientation_deg=float(np.degrees(np.angle(z)) / 6.0))
    return out


def is_hexagonal(m: dict, max_angle_err: float = 15.0, max_len_spread: float = 0.25, min_gridness: float = 0.3) -> bool:
    """Our (arbitrary, stated) criterion for a hexagonal module: six peaks round the origin whose angles
    are within 15 deg of a regular hexagon, distances within 25 %, and an analytic gridness above 0.3."""
    e, l, g = m.get("ac_hex_err_deg", float("nan")), m.get("ac_len_spread", float("nan")), m.get("gridness_analytic", float("nan"))
    return bool(np.isfinite(e) and e < max_angle_err and l < max_len_spread and g > min_gridness)


def _line_angle_deg(k):
    """Orientation of a wave vector as a line direction, in [0, 180)."""
    return float(np.degrees(np.arctan2(k[1], k[0])) % 180.0)


def lattice_summary(p: Params, idx: np.ndarray, shell_tol: float = 0.2, min_share: float = 0.25):
    """Describe the frequency lattice used by the neurons `idx` (one module).

    Returns a dict with: the strongest wave's |k| (k_fund) and the grid spacing it implies (hexagonal
    convention 4 pi / (sqrt(3) |k|)), the number and power share of the first-shell waves, a
    first-shell hexagonal-angle error, the autocorrelogram-based lattice statistics of `ac_lattice`
    (the ones used in the report) and the analytic gridness of the module."""
    P = freq_power(p)[:, idx].mean(1)
    K = p["K"]
    kn = np.linalg.norm(K, axis=1)
    pw = P / P.sum()
    d1 = int(np.argmax(pw))
    shell = np.where((np.abs(kn - kn[d1]) <= shell_tol * kn[d1]) & (pw >= min_share * pw[d1]))[0]
    shell = shell[np.argsort(-pw[shell])][:3]
    out = dict(k_fund=float(kn[d1]), spacing=float(4 * np.pi / (np.sqrt(3) * kn[d1])),
               n_shell=int(len(shell)), shell_power=float(pw[shell].sum()),
               n_eff=float(1.0 / np.sum(pw ** 2)))
    if len(shell) == 3:
        ang = np.sort([_line_angle_deg(K[d]) for d in shell])
        gaps = np.diff(np.concatenate([ang, [ang[0] + 180.0]]))
        out["hex_err_deg"] = float(np.max(np.abs(gaps - 60.0)))
        out["len_spread"] = float((kn[shell].max() - kn[shell].min()) / kn[shell].mean())
        z = np.mean([np.exp(1j * np.radians(6 * _line_angle_deg(K[d]))) for d in shell])
        out["orientation_deg"] = float(np.degrees(np.angle(z)) / 6.0)
    else:
        out["hex_err_deg"] = float("nan"); out["len_spread"] = float("nan"); out["orientation_deg"] = float("nan")
    # lattice measured on the module's autocorrelogram (robust to duplicate / harmonic frequencies)
    out.update(ac_lattice(K.astype(float), P.astype(float)))
    # analytic gridness of the module (one autocorrelogram per module, window = 3.2 spacings of the lowest strong wave)
    R = 3.2 * spacing_hint(K.astype(float), P.astype(float))
    out["gridness_analytic"] = float(gridness_from_ac(analytic_autocorr(K.astype(float), P.astype(float), R, 121)))
    return out


def neuron_stats(p: Params):
    """Per-neuron analytic gridness (from the neuron's own spectrum) and effective number of
    frequencies (participation ratio of its power spectrum)."""
    P = freq_power(p).astype(float)
    K = p["K"].astype(float)
    g = np.zeros(P.shape[1]); neff = np.zeros(P.shape[1])
    for n in range(P.shape[1]):
        pw = P[:, n] / P[:, n].sum()
        neff[n] = 1.0 / np.sum(pw ** 2)
        g[n] = gridness_from_ac(analytic_autocorr(K, P[:, n], 3.2 * spacing_hint(K, P[:, n]), 121))
    return g, neff


def central_peak_fwhm(K: np.ndarray, power: np.ndarray, spacing: float, res: int = 241):
    """FWHM of the central peak of the analytic autocorrelogram (azimuthal mean falls to 1/2).
    For Gaussian fields of FWHM w the autocorrelation peak has FWHM sqrt(2) w."""
    R = spacing
    ac = analytic_autocorr(K, power, R, res)
    c = res // 2
    yy, xx = np.mgrid[:res, :res]
    rr = np.hypot(yy - c, xx - c) * (R / c)
    bins = np.linspace(0, R, 120)
    prof = np.array([ac[(rr >= bins[i]) & (rr < bins[i + 1])].mean() for i in range(len(bins) - 1)])
    below = np.where(prof < 0.5)[0]
    r_half = bins[below[0]] if len(below) else R
    return 2 * r_half


# --------------------------------------------------------------------------- is the code really actionable?
def coefficient_matrix(p: Params) -> np.ndarray:
    """W of shape (N, 2D+1) with g(x) = W I(x),  I(x) = [1, sin(k_1.x), cos(k_1.x), ..., sin(k_D.x), cos(k_D.x)]."""
    cols = [p["a0"][:, None]]
    for d in range(p["K"].shape[0]):
        cols += [p["A"][d][:, None], p["B"][d][:, None]]
    return np.concatenate(cols, axis=1).astype(np.float64)


def actionability_report(p: Params, n_test: int = 400, seed: int = 0, L: float = 1.0):
    """Is the learned code *really* actionable?  g(x+dx) = T(dx) g(x) with T = W G_I(dx) W^+ holds
    exactly only if the coefficient matrix W has full column rank 2D+1 (distinct neurons must carry
    independent mixtures of the sines/cosines); the parametrisation of eq. 6 alone does not ensure this.
    Returns the singular-value ratio s_{2D+1}/s_1 of W and the relative RMS error of the best linear
    action,  |T g(x) - g(x+dx)| / |g(x+dx) - mean|, over random (x, dx)."""
    W = coefficient_matrix(p)
    K = p["K"].astype(np.float64)
    N, Q = W.shape
    sv = np.linalg.svd(W, compute_uv=False)
    rank_ratio = float(sv[-1] / sv[0]) if N >= Q else 0.0
    rng = np.random.default_rng(seed)
    dim = K.shape[1]
    X = rng.normal(0, L, (n_test, dim)); dX = rng.normal(0, 1.0, (n_test, dim))

    def I(Xs):
        ph = Xs @ K.T
        out = np.ones((len(Xs), Q)); out[:, 1::2] = np.sin(ph); out[:, 2::2] = np.cos(ph)
        return out

    Wp = np.linalg.pinv(W, rcond=1e-8)
    errs, scale = [], []
    for i in range(n_test):
        d = dX[i]; ph = K @ d; c, s_ = np.cos(ph), np.sin(ph)
        GI = np.eye(Q)
        for k in range(K.shape[0]):
            GI[1 + 2 * k:3 + 2 * k, 1 + 2 * k:3 + 2 * k] = [[c[k], s_[k]], [-s_[k], c[k]]]
        gx = W @ I(X[i:i + 1])[0]; gxd = W @ I((X[i] + d)[None])[0]
        pred = W @ GI @ Wp @ gx
        errs.append(np.sum((pred - gxd) ** 2)); scale.append(np.sum((gxd - W[:, 0]) ** 2))
    return dict(rank_ratio=rank_ratio, singular_values_min_max=[float(sv[-1]), float(sv[0])], N=int(N), Q=int(Q),
                actionability_rel_error=float(np.sqrt(np.sum(errs) / max(np.sum(scale), 1e-12))))
