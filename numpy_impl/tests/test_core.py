"""Unit tests.  Run from the repository root:   python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from actionable import TrainConfig, init_params, encode, loss_and_grads, GECO, sample_points, train   # noqa: E402
from actionable import analysis, module_angle                                                          # noqa: E402


def total_loss(p, X, sh, cfg, lp, ln):
    out, _ = loss_and_grads(p, X, sh, cfg, lp, ln, want_grad=False)
    return out["Lf"] + lp * out["Lp"] + ln * out["Lb"]


def worst_gradient_error(cfg, n_check=8, seed=1):
    rng = np.random.default_rng(seed)
    p = init_params(cfg.N, cfg.D, cfg.dim, rng, cfg.coef_std, 2.0, "float64")
    X, sh = sample_points(cfg, rng)
    lp, ln = 0.3, 0.7
    _, g = loss_and_grads(p, X, sh, cfg, lp, ln)
    worst = 0.0
    for k in p:
        idx = list(np.ndindex(*p[k].shape))
        for j in rng.choice(len(idx), size=min(n_check, len(idx)), replace=False):
            ii, e, q = idx[j], 1e-6, p[k][idx[j]]
            p[k][ii] = q + e; f1 = total_loss(p, X, sh, cfg, lp, ln)
            p[k][ii] = q - e; f2 = total_loss(p, X, sh, cfg, lp, ln)
            p[k][ii] = q
            num = (f1 - f2) / (2 * e)
            worst = max(worst, abs(num - g[k][ii]) / (abs(num) + abs(g[k][ii]) + 1e-8))
    return worst


def cos_pattern(K, res=129, half_width=3.0):
    xs = np.linspace(-half_width, half_width, res)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    return sum(np.cos(k[0] * XX + k[1] * YY) for k in K)


class TestGradients(unittest.TestCase):
    def test_full_loss_sum_norm(self):
        cfg = TrainConfig(N=12, D=5, M=20, MS=4, dtype="float64", sigma=0.4, functional="full", norm_mode="sum")
        self.assertLess(worst_gradient_error(cfg), 1e-5)

    def test_full_loss_mean_norm(self):
        cfg = TrainConfig(N=12, D=5, M=20, MS=4, dtype="float64", sigma=0.4, functional="full", norm_mode="mean")
        self.assertLess(worst_gradient_error(cfg), 1e-5)

    def test_simplified_loss_1d(self):
        cfg = TrainConfig(N=10, D=4, dim=1, M=20, MS=4, dtype="float64", functional="l0")
        self.assertLess(worst_gradient_error(cfg), 1e-5)


class TestModel(unittest.TestCase):
    def test_code_is_actionable(self):
        """g(x+dx) = T(dx) g(x) holds to machine precision when N >= 2D+1 (paper App. A)."""
        rng = np.random.default_rng(0)
        p = init_params(40, 9, 2, rng, dtype="float64")
        rep = analysis.actionability_report(p, n_test=50)
        self.assertGreater(rep["rank_ratio"], 1e-3)
        self.assertLess(rep["actionability_rel_error"], 1e-8)

    def test_too_many_frequencies_not_actionable(self):
        rng = np.random.default_rng(0)
        p = init_params(20, 30, 2, rng, dtype="float64")            # 2D+1 = 61 > N = 20
        rep = analysis.actionability_report(p, n_test=50)
        self.assertGreater(rep["actionability_rel_error"], 0.05)

    def test_norm_constraint_exact_without_shift(self):
        """With 'sum' normalisation the unshifted room has exactly unit norm, so L_bounded = 0 for zero shifts."""
        cfg = TrainConfig(N=8, D=3, M=30, MS=3, dtype="float64")
        rng = np.random.default_rng(0)
        p = init_params(cfg.N, cfg.D, cfg.dim, rng, dtype="float64")
        X, _ = sample_points(cfg, rng)
        out, _ = loss_and_grads(p, X, np.zeros((cfg.MS, 2)), cfg, 0.1, 0.1, want_grad=False)
        self.assertAlmostEqual(out["Lb"], 0.0, places=12)

    def test_geco_direction_and_bounds(self):
        g = GECO(lam0=0.1, k=np.log(1e-3), alpha=0.0, gamma=0.1)
        lam = [g.update(1e-1) for _ in range(20)]                     # constraint violated -> lambda grows
        self.assertTrue(all(b > a for a, b in zip(lam, lam[1:])))
        g = GECO(lam0=0.1, k=np.log(1e-3), alpha=0.0, gamma=0.1)
        lam = [g.update(1e-6) for _ in range(20)]                     # constraint satisfied -> lambda shrinks
        self.assertTrue(all(b < a for a, b in zip(lam, lam[1:])))
        g = GECO(lam0=0.1, k=0.0, alpha=0.0, gamma=1e3)
        self.assertLessEqual(g.update(1e30), 1e8)

    def test_training_reduces_loss(self):
        cfg = TrainConfig(N=16, D=7, steps=600, M=60, MS=4, sigma=0.5, log_every=100, seed=3)
        _, h = train(cfg, verbose=False)
        self.assertLess(np.mean(h["Lf"][-2:]), np.mean(h["Lf"][:2]))

    def test_exact_resume(self):
        cfg = TrainConfig(N=12, D=5, steps=60, M=30, MS=3, log_every=10, seed=5)
        p_full, _ = train(cfg, verbose=False)
        with tempfile.TemporaryDirectory() as d:
            ck = os.path.join(d, "ck.npz")
            cfg_a = TrainConfig(**{**cfg.__dict__, "steps": 30})
            train(cfg_a, verbose=False, ckpt_path=ck, ckpt_every=30)
            p_res, _ = train(cfg, verbose=False, ckpt_path=ck, ckpt_every=30)
        for k in p_full:
            np.testing.assert_allclose(p_full[k], p_res[k], rtol=0, atol=0)


class TestGridness(unittest.TestCase):
    def setUp(self):
        k = 6.0
        self.hex = [k * np.array([np.cos(a), np.sin(a)]) for a in np.radians([0, 60, 120])]
        self.sq = [k * np.array([1, 0]), k * np.array([0, 1])]

    def test_hexagonal_pattern_scores_high(self):
        self.assertGreater(analysis.gridness(cos_pattern(self.hex)), 1.0)

    def test_square_pattern_scores_low(self):
        self.assertLess(analysis.gridness(cos_pattern(self.sq)), 0.3)

    def test_noise_scores_near_zero(self):
        rng = np.random.default_rng(0)
        rm = np.array(rng.normal(size=(129, 129)))
        from scipy.ndimage import gaussian_filter
        self.assertLess(analysis.gridness(gaussian_filter(rm, 4)), 0.6)


class TestModules(unittest.TestCase):
    def test_two_modules_are_separated(self):
        rng = np.random.default_rng(0)
        N, D = 24, 8
        K = np.concatenate([rng.normal(0, 4, (4, 2)), rng.normal(0, 7, (4, 2))])
        A = np.zeros((D, N)); B = np.zeros((D, N))
        A[:4, :12] = rng.normal(size=(4, 12)); B[:4, :12] = rng.normal(size=(4, 12))
        A[4:, 12:] = rng.normal(size=(4, 12)); B[4:, 12:] = rng.normal(size=(4, 12))
        p = dict(a0=np.zeros(N), A=A, B=B, K=K)
        lab = analysis.cluster_modules(p)
        self.assertEqual(len(set(lab[:12])), 1)
        self.assertEqual(len(set(lab[12:])), 1)
        self.assertNotEqual(lab[0], lab[12])


class TestLattice(unittest.TestCase):
    def _module(self, K):
        K = np.asarray(K, float)
        return K, np.ones(len(K))

    def test_hexagonal_module(self):
        k = 7.0
        for rot in (0.0, 13.0):
            K, P = self._module([k * np.array([np.cos(a), np.sin(a)]) for a in np.radians([rot, rot + 60, rot + 120])])
            m = analysis.ac_lattice(K, P)
            self.assertGreaterEqual(m["ac_n_peaks"], 6)
            self.assertLess(m["ac_hex_err_deg"], 2.0)
            self.assertAlmostEqual(m["ac_spacing"], 4 * np.pi / (np.sqrt(3) * k), delta=0.02)
            # real-space lattice directions are perpendicular to the waves' directions, i.e. rotated by 30 deg (mod 60)
            self.assertAlmostEqual(((m["ac_orientation_deg"] - (rot + 30.0) + 30) % 60) - 30, 0.0, delta=1.0)

    def test_square_module_is_not_hexagonal(self):
        K, P = self._module([[6.0, 0.0], [0.0, 6.0]])
        m = analysis.ac_lattice(K, P)
        m["gridness_analytic"] = analysis.gridness_from_ac(analysis.analytic_autocorr(K, P, 3.2 * analysis.spacing_hint(K, P), 121))
        self.assertFalse(analysis.is_hexagonal(m))

    def test_single_wave_is_stripes(self):
        K, P = self._module([[6.0, 0.0]])
        m = analysis.ac_lattice(K, P)
        self.assertTrue(np.isnan(m["ac_hex_err_deg"]))

    def test_harmonics_do_not_change_lattice(self):
        k = 6.0
        base = [k * np.array([np.cos(a), np.sin(a)]) for a in np.radians([0, 60, 120])]
        K = np.array(base + [2 * b for b in base])
        P = np.array([1, 1, 1, 0.3, 0.3, 0.3])
        m = analysis.ac_lattice(K, P)
        self.assertLess(m["ac_hex_err_deg"], 2.0)
        self.assertAlmostEqual(m["ac_spacing"], 4 * np.pi / (np.sqrt(3) * k), delta=0.03)


class TestModuleAngle(unittest.TestCase):
    def test_density_positive_and_60deg_symmetric(self):
        rho = module_angle.angular_density(0.05, 0.2, n_psi=720)
        self.assertTrue(np.all(rho >= 0))
        np.testing.assert_allclose(rho, np.roll(rho, 720 // 6), rtol=1e-6, atol=1e-9 * rho.max())

    def test_two_module_optimum_is_small_and_beats_alignment(self):
        d, o, o0 = module_angle.optimal_offset(0.05, 0.2)
        self.assertGreater(d, 2.0); self.assertLess(d, 8.0); self.assertLess(o, o0)

    def test_three_module_offsets_are_small_steps(self):
        (a, b), _ = module_angle.three_module_offsets(0.05, 0.2, n_psi=720)
        self.assertLess(a, 8.0); self.assertLess(b, 8.0)


if __name__ == "__main__":
    unittest.main()
