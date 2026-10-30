"""Model, losses (with hand-derived gradients) and training loop.

Everything is implemented from the paper's Section 2/3 and Appendix B.1 ("very large
spaces").  No autodiff library is used (so the only dependency is NumPy); the analytic
gradients are verified against finite differences in tests/test_core.py.

Model  (paper eq. 6 / 20):   g(x) = a0 + sum_d  a_d sin(k_d . x) + b_d cos(k_d . x)
    a0 in R^N, a_d, b_d in R^N, k_d in R^dim  (frequencies k_d are *learned*).
The number of frequencies D must be < N/2 for the code to be actionable (App. A).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import time
import numpy as np

Params = Dict[str, np.ndarray]  # keys: a0 (N,), A (D,N), B (D,N), K (D,dim)


# --------------------------------------------------------------------------- model
def init_params(N: int, D: int, dim: int, rng: np.random.Generator,
                coef_std: float = 1.0, freq_std: float = 4.0, dtype="float64") -> Params:
    """Zero-mean Gaussian init (paper: 'independent zero-mean gaussians'; the variances
    are NOT given in the paper's table B.3, so they are hyper-parameters here)."""
    p = {
        "a0": rng.normal(0, coef_std, N) / np.sqrt(D),
        "A": rng.normal(0, coef_std, (D, N)) / np.sqrt(D),
        "B": rng.normal(0, coef_std, (D, N)) / np.sqrt(D),
        "K": rng.normal(0, freq_std, (D, dim)),
    }
    return {k: v.astype(dtype) for k, v in p.items()}


def encode(p: Params, X: np.ndarray, cache: bool = False):
    """g(X) for X of shape (P, dim) -> (P, N)."""
    Phi = X @ p["K"].T
    S, C = np.sin(Phi), np.cos(Phi)
    G = p["a0"] + S @ p["A"] + C @ p["B"]
    return (G, (S, C)) if cache else G


def _backprop_to_params(p: Params, X: np.ndarray, S: np.ndarray, C: np.ndarray,
                        dG: np.ndarray, grads: Params) -> None:
    """Accumulate d loss / d params given d loss / d G at positions X."""
    grads["a0"] += dG.sum(0)
    grads["A"] += S.T @ dG
    grads["B"] += C.T @ dG
    dPhi = (dG @ p["A"].T) * C - (dG @ p["B"].T) * S      # (P, D)
    grads["K"] += dPhi.T @ X


# --------------------------------------------------------------------------- losses
def functional_loss(Gt: np.ndarray, X: np.ndarray, sigma: float, l: float,
                    kind: str = "full") -> Tuple[float, np.ndarray]:
    """Discrete version of eq. 1/23.  Gt = normalised code (M, N), X = positions (M, dim).
    kind='full': mean_ij exp(-|g_i-g_j|^2 / 2 sigma^2) chi(x_i,x_j)
    kind='l0'  : -mean_ij |g_i - g_j|^2        (simplified loss of eq. 7)
    Returns (loss, dloss/dGt)."""
    M = Gt.shape[0]
    sq = (Gt ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Gt @ Gt.T, 0.0)
    if kind == "l0":
        L = -D2.mean()
        grad = -(4.0 / M) * (Gt - Gt.mean(0))
        return float(L), grad
    sx = (X ** 2).sum(1)
    DX2 = np.maximum(sx[:, None] + sx[None, :] - 2.0 * X @ X.T, 0.0)
    chi = 1.0 - np.exp(-DX2 / (2.0 * l ** 2))
    W = np.exp(-D2 / (2.0 * sigma ** 2)) * chi
    L = W.mean()
    # dL/dGt_i = -(2 / (sigma^2 M^2)) sum_j W_ij (Gt_i - Gt_j)
    grad = -(2.0 / (sigma ** 2 * M ** 2)) * (W.sum(1)[:, None] * Gt - W @ Gt)
    return float(L), grad


@dataclass
class TrainConfig:
    # --- problem (paper Table B.3 defaults for Fig. 2B)
    N: int = 64
    D: int = 31                 # number of frequencies (< N/2); NOT stated in the paper
    dim: int = 2
    sigma: float = 0.2          # neural length-scale
    l: float = 0.5              # chi length-scale
    L: float = 1.0              # occupancy length-scale (units of space)
    occupancy: str = "gaussian" # 'gaussian': x ~ N(0, L^2 I) as in eq. 2;  'uniform': x ~ U(-sqrt(3) L, sqrt(3) L)^dim
    functional: str = "full"    # 'full' | 'l0' | 'none'
    norm_mode: str = "sum"      # 'sum': unit L2 norm over the M samples (eq. 22 read literally)
                                # 'mean': unit mean-square (the continuous constraint of eq. 2)
    # --- optimisation
    steps: int = 150_000
    M: int = 150                # sampled points
    MS: int = 15                # sampled room shifts
    S: float = 3.0              # std of shift sampling
    n_resample: int = 5
    lr_coef: float = 0.1
    lr_freq: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.9
    eps: float = 1e-8
    lr_decay_to: float = 1.0    # final lr = lr * lr_decay_to (1.0 = constant, as in the paper)
    # --- GECO (positivity p, norm n)
    lam_p0: float = 0.1
    k_p: float = -9.0
    alpha_p: float = 0.9
    gamma_p: float = 1e-4
    lam_n0: float = 0.005
    k_n: float = 4.0
    alpha_n: float = 0.9
    gamma_n: float = 1e-4
    use_positivity: bool = True
    use_norm: bool = True
    # --- init
    coef_std: float = 1.0
    freq_std: float = 4.0
    seed: int = 0
    log_every: int = 500
    dtype: str = "float32"      # float64 is used in the gradient tests


def sample_points(cfg: TrainConfig, rng: np.random.Generator):
    if cfg.occupancy == "uniform":                            # same variance L^2 as the Gaussian
        X = rng.uniform(-np.sqrt(3.0) * cfg.L, np.sqrt(3.0) * cfg.L, (cfg.M, cfg.dim))
    else:
        X = rng.normal(0.0, cfg.L, (cfg.M, cfg.dim))          # x ~ p(x) = N(0, L^2 I)
    shifts = rng.normal(0.0, cfg.S, (cfg.MS, cfg.dim))        # 'room shifts'
    return X.astype(cfg.dtype), shifts.astype(cfg.dtype)


def loss_and_grads(p: Params, X: np.ndarray, shifts: np.ndarray, cfg: TrainConfig,
                   lam_p: float, lam_n: float, want_grad: bool = True):
    """Full loss L = L_func + lam_p L_nonneg + lam_n L_bounded (eq. 21, 23-25).

    Interpretation choices (the paper text is ambiguous here, see README):
      * norm_mode='sum' (default, eq. 22 read literally): |g_n|^2 = sum_m g_n(x_m)^2 over
        the M sampled points and g~_n = g_n / |g_n|.  norm_mode='mean' divides the sum by M.
      * the same normaliser is used for the shifted copies of the room.
      * L_bounded = mean_{s,n} (c * sum_m g~_n(x_s + x_m)^2 - 1)^2, with c = 1 ('sum') or 1/M.
    """
    M, MS, N, dim = cfg.M, cfg.MS, cfg.N, cfg.dim
    G, (S0, C0) = encode(p, X, cache=True)
    c = 1.0 if cfg.norm_mode == "sum" else 1.0 / M
    s = np.sqrt(c * (G ** 2).sum(0)) + 1e-12                   # (N,)
    Gt = G / s

    if cfg.functional == "none":
        Lf, dGt = 0.0, np.zeros_like(Gt)
    else:
        Lf, dGt = functional_loss(Gt, X, cfg.sigma, cfg.l, cfg.functional)

    # shifted rooms
    Xs = (X[None, :, :] + shifts[:, None, :]).reshape(MS * M, dim)
    H, (S1, C1) = encode(p, Xs, cache=True)
    Ht = (H / s).reshape(MS, M, N)
    neg = Ht < 0
    Lp = float(np.where(neg, -Ht, 0.0).mean())
    q = c * (Ht ** 2).sum(1)                                   # (MS, N)
    Lb = float(((q - 1.0) ** 2).mean())
    out = {"Lf": Lf, "Lp": Lp, "Lb": Lb}
    if not want_grad:
        return out, None

    wp = lam_p if cfg.use_positivity else 0.0
    wn = lam_n if cfg.use_norm else 0.0
    dHt = np.zeros_like(Ht)
    if wp:
        dHt += wp * (-neg.astype(float)) / Ht.size
    if wn:
        dHt += wn * (2.0 * (q - 1.0) / (MS * N))[:, None, :] * (2.0 * c * Ht)
    dHt = dHt.reshape(MS * M, N)
    # back through normalisation  g~ = g / s   (s depends on G)
    dG = dGt / s
    dH = dHt / s
    ds = -((dGt * G).sum(0) + (dHt * H).sum(0)) / s ** 2
    dG = dG + ds * c * G / s
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    _backprop_to_params(p, X, S0, C0, dG, grads)
    _backprop_to_params(p, Xs, S1, C1, dH, grads)
    return out, grads


# --------------------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params: Params, lrs: Dict[str, float], b1=0.9, b2=0.9, eps=1e-8):
        self.lrs, self.b1, self.b2, self.eps, self.t = dict(lrs), b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Params, grads: Params, lr_scale: float = 1.0):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            params[k] -= lr_scale * self.lrs[k] * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


class GECO:
    """Rezende & Viola (2018)-style constraint weighting, paper eqs. 26-28.
    log-violation  l_t = log(L_t) - k ;  smoothed lhat = a lhat + (1-a) l_t ;
    lambda <- lambda * exp(gamma * lhat)."""

    def __init__(self, lam0: float, k: float, alpha: float, gamma: float):
        self.lam, self.k, self.alpha, self.gamma, self.lhat = lam0, k, alpha, gamma, 0.0

    def update(self, L: float) -> float:
        lt = np.log(max(L, 1e-30)) - self.k
        self.lhat = self.alpha * self.lhat + (1 - self.alpha) * lt
        self.lam = float(np.clip(self.lam * np.exp(self.gamma * self.lhat), 1e-8, 1e8))
        return self.lam


# --------------------------------------------------------------------------- training
def _save_ckpt(path, cfg, params, opt, gp, gn, rng, hist, t_done):
    import json, os
    tmp = path + ".tmp.npz"
    arrs = {f"p_{k}": v for k, v in params.items()}
    arrs.update({f"m_{k}": v for k, v in opt.m.items()})
    arrs.update({f"v_{k}": v for k, v in opt.v.items()})
    arrs.update({f"h_{k}": np.asarray(v) for k, v in hist.items()})
    meta = dict(t_done=t_done, adam_t=opt.t, gp=[gp.lam, gp.lhat], gn=[gn.lam, gn.lhat],
                rng=rng.bit_generator.state, cfg=asdict(cfg))
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, path)


def _load_ckpt(path, cfg, params, opt, gp, gn, rng):
    import json
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    old = dict(meta["cfg"]); new = asdict(cfg)
    for k in ("steps", "log_every"):
        old.pop(k, None); new.pop(k, None)
    default = asdict(TrainConfig())
    for k in list(new):                  # fields added after the checkpoint was written: fine if at their default
        if k not in old and new[k] == default[k]:
            new.pop(k)
    if old != new:
        raise ValueError("checkpoint was produced with a different configuration; delete it or change --name")
    for k in params:
        params[k][...] = z[f"p_{k}"]
        opt.m[k][...] = z[f"m_{k}"]; opt.v[k][...] = z[f"v_{k}"]
    opt.t = meta["adam_t"]
    gp.lam, gp.lhat = meta["gp"]; gn.lam, gn.lhat = meta["gn"]
    rng.bit_generator.state = meta["rng"]
    hist = {k[2:]: list(z[k]) for k in z.files if k.startswith("h_")}
    return meta["t_done"], hist


def train(cfg: TrainConfig, params: Optional[Params] = None, verbose: bool = True,
          callback=None, ckpt_path: Optional[str] = None, ckpt_every: int = 10_000,
          max_seconds: Optional[float] = None):
    """Train the model.  If `ckpt_path` exists the run resumes from it (exactly: the optimiser,
    GECO and RNG states are restored).  `max_seconds` stops early (checkpointing first) so that
    long runs can be executed in several sessions; the returned flag says if training is done."""
    import os
    rng = np.random.default_rng(cfg.seed)
    if params is None:
        params = init_params(cfg.N, cfg.D, cfg.dim, rng, cfg.coef_std, cfg.freq_std, cfg.dtype)
    opt = Adam(params, {"a0": cfg.lr_coef, "A": cfg.lr_coef, "B": cfg.lr_coef, "K": cfg.lr_freq},
               cfg.beta1, cfg.beta2, cfg.eps)
    gp = GECO(cfg.lam_p0, cfg.k_p, cfg.alpha_p, cfg.gamma_p)
    gn = GECO(cfg.lam_n0, cfg.k_n, cfg.alpha_n, cfg.gamma_n)
    hist = {k: [] for k in ("step", "Lf", "Lp", "Lb", "lam_p", "lam_n")}
    t_start = 0
    if ckpt_path and os.path.exists(ckpt_path):
        t_start, hist = _load_ckpt(ckpt_path, cfg, params, opt, gp, gn, rng)
        if verbose:
            print(f"resumed from {ckpt_path} at step {t_start}", flush=True)
    t0 = time.time()
    X = shifts = None
    done = True
    for t in range(t_start, cfg.steps):
        if (t - t_start) % cfg.n_resample == 0 or X is None:
            X, shifts = sample_points(cfg, rng)
        out, grads = loss_and_grads(params, X, shifts, cfg, gp.lam, gn.lam)
        if cfg.use_positivity:
            gp.update(out["Lp"])
        if cfg.use_norm:
            gn.update(out["Lb"])
        frac = t / max(cfg.steps - 1, 1)
        scale = 1.0 + (cfg.lr_decay_to - 1.0) * frac
        opt.step(params, grads, scale)
        if t % cfg.log_every == 0 or t == cfg.steps - 1:
            for k, v in (("step", t), ("Lf", out["Lf"]), ("Lp", out["Lp"]), ("Lb", out["Lb"]),
                         ("lam_p", gp.lam), ("lam_n", gn.lam)):
                hist[k].append(v)
            if verbose and (t % (cfg.log_every * 10) == 0 or t == cfg.steps - 1):
                print(f"[{t:>7d}/{cfg.steps}] Lf={out['Lf']:.4g} Lp={out['Lp']:.3g} "
                      f"Lb={out['Lb']:.3g} lam_p={gp.lam:.3g} lam_n={gn.lam:.3g} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if callback is not None:
                callback(t, params, hist)
        stop = (max_seconds is not None and (time.time() - t0) > max_seconds and t + 1 < cfg.steps
                and (t + 1) % cfg.n_resample == 0)          # stop on a resampling boundary => exact resume
        if ckpt_path and ((t + 1) % ckpt_every == 0 or stop or t + 1 == cfg.steps):
            _save_ckpt(ckpt_path, cfg, params, opt, gp, gn, rng, hist, t + 1)
        if stop:
            done = False
            break
    hist = {k: np.array(v) for k, v in hist.items()}
    return (params, hist, done) if max_seconds is not None else (params, hist)
