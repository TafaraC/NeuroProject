"""NumPy re-implementation of the actionable-representation model of
Dorrell, Latham, Behrens & Whittington (ICLR 2023), written from the paper text only."""
from .core import (Params, init_params, encode, loss_and_grads, GECO, Adam,
                   sample_points, train, TrainConfig)
from . import analysis, plotting, module_angle
