"""
Tensor decomposition experiments for gait analysis and injury recovery prediction.

This package contains experiments comparing different tensor decomposition approaches
for classifying gait data at different stages of injury recovery.
"""

from experiments.acc_vs_rd import AccVsReducedDimension
from experiments.acc_vs_samples import AccVsSamples
from experiments.acc_vs_rd_vs_samples import AccVsReducedDimensionVsSamples
from experiments.base import TensorExperiment

__all__ = [
    "TensorExperiment",
    "AccVsReducedDimension",
    "AccVsSamples",
    "AccVsReducedDimensionVsSamples",
]

