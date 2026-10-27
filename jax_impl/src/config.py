"""Explicit experiment configuration; all lengths are in units of L=1."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    neurons: int = 64
    frequencies: int = 31
    sigma: float = 0.4
    spatial_scale: float = 0.5
    occupancy_scale: float = 1.0
    samples: int = 150
    shifts: int = 15
    shift_scale: float = 3.0
    resample_every: int = 5
    steps: int = 150_000
    seed: int = 0
    weight_lr: float = 0.1
    frequency_lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.9
    adam_eps_root: float = 1e-8
    functional_scale: float = 25.0
    positivity_weight: float = 0.1
    norm_weight: float = 0.005
    positivity_target: float = 0.00012340980408667956  # exp(-9)
    norm_target: float = 0.02
    geco_smoothing: float = 0.9
    geco_rate: float = 1e-4
    frequency_init_scale: float = 2.0
    dtype: str = "float32"

    def validate(self):
        for name in (
            "neurons",
            "frequencies",
            "samples",
            "shifts",
            "resample_every",
            "steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if 2 * self.frequencies + 1 > self.neurons:
            raise ValueError("Actionability requires 2 * frequencies + 1 <= neurons")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        import math

        for name, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("beta1", "beta2", "geco_smoothing"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return Config(**json.loads(Path(path).read_text(encoding="utf-8-sig"))).validate()
