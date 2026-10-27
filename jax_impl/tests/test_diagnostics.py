import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.diagnostics import make_plan, paired_rows, validate_plan
from experiments.equation_check import check_candidate, expected_values, export_fixture
from src.config import Config
from src.model import loss_components, objective
from src.training import load_checkpoint


def test_study_rejects_confound_and_missing_pair():
    plan = make_plan(Config(), [0, 1, 2])
    assert len(plan["runs"]) == 9
    altered = deepcopy(plan)
    altered["runs"][1]["config"]["frequency_lr"] = 0.01
    with pytest.raises(ValueError, match="declared factor"):
        validate_plan(altered)
    altered = deepcopy(plan)
    altered["runs"].pop()
    with pytest.raises(ValueError, match="missing"):
        validate_plan(altered)
    altered = deepcopy(plan)
    altered["runs"].append(altered["runs"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_plan(altered)
    with pytest.raises(ValueError, match="distinct"):
        make_plan(Config(), [0, 0])


def test_pairing_uses_seed_not_row_order_or_cells():
    rows = []
    for condition, seed, value in (
        ("sigma_variance", 1, 8),
        ("baseline", 0, 1),
        ("baseline", 1, 7),
        ("sigma_variance", 0, 3),
    ):
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "grid_score_median": value,
                "norm_loss": value,
                "negative_fraction_below_minus_1e_minus6": value,
            }
        )
    differences = paired_rows(rows)
    assert [r["delta_grid_score_median"] for r in differences] == [1, 2]
    rows[0]["grid_score_median"] = None
    assert paired_rows(rows)[0]["delta_grid_score_median"] is None


def test_portable_contract_matches_autodiff_and_detects_sigma_confusion(tmp_path):
    jax.config.update("jax_enable_x64", True)
    data, c = export_fixture(tmp_path / "fixture")
    expected = expected_values(data, c)
    w, f, x, shifts, multipliers = [
        jnp.asarray(data[k])
        for k in ("weights", "frequencies", "points", "shifts", "multipliers")
    ]
    function = lambda weights, frequencies: objective(
        weights, frequencies, x, shifts, multipliers, c
    )[0]
    dw, df = jax.grad(function, (0, 1))(w, f)
    parts = loss_components(w, f, x, shifts, c)
    candidate = dict(zip(("functional", "positivity", "norm"), map(float, parts)))
    candidate.update(
        total=float(function(w, f)),
        weight_directional_derivative=float(np.sum(dw * data["weight_direction"])),
        frequency_directional_derivative=float(
            np.sum(df * data["frequency_direction"])
        ),
    )
    assert check_candidate(expected, candidate) == {}
    confused = replace(c, sigma=np.sqrt(c.sigma))
    candidate["functional"] = float(loss_components(w, f, x, shifts, confused)[0])
    assert "functional" in check_candidate(expected, candidate)
    candidate["norm"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        check_candidate(expected, candidate)


def test_cli_snapshots_cross_blocks_and_survive_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    c = Config(
        neurons=7, frequencies=3, samples=12, shifts=3, steps=30, dtype="float64"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(c.to_dict()))

    def run(folder, extra):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.run",
                "--output",
                str(folder),
                "--block-size",
                "10",
                "--snapshot-steps",
                "7",
                "19",
                *extra,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

    whole = tmp_path / "whole"
    split = tmp_path / "split"
    run(whole, ["--config", str(config_path)])
    run(split, ["--config", str(config_path), "--stop-at", "13"])
    assert (split / "snapshots/step_000007.npz").exists()
    assert not (split / "snapshots/step_000019.npz").exists()
    run(split, ["--resume", str(split / "checkpoint.npz")])
    for filename in (
        "checkpoint.npz",
        "snapshots/step_000007.npz",
        "snapshots/step_000019.npz",
    ):
        expected, _ = load_checkpoint(whole / filename)
        actual, _ = load_checkpoint(split / filename)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11)
