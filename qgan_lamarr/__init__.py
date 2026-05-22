import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hides INFO/WARNING C++ logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # suppresses the oneDNN notice
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")      # hides absl pre-init warnings


from .models        import QGAN, XMapQCGAN, SandwichQCGAN
from .optimize      import QGAN_optimizer, AdamOptimizerPSR, parameter_shift_rule
from .metrics       import (jensen_shannon, fidelity, wasserstein,
                            scoreKS, chi2, kullback_leibler_divergence,
                            generator_entropy, metric_avg)
from .tools         import dict2vector, dict2sample
from .distributions import SingleGaussian, MixedGaussian, MinMaxBinning, RangeBinning
from .manager       import FileManager
from .dashboard     import QGANDashboard

__version__ = "0.1.0"

__all__ = [
    # Core model
    "QGAN", "XMapQCGAN", "SandwichQCGAN",
    # Optimizers
    "QGAN_optimizer", "AdamOptimizerPSR", "parameter_shift_rule",
    # Metrics
    "jensen_shannon", "fidelity", "wasserstein", "scoreKS", "chi2",
    "kullback_leibler_divergence", "generator_entropy", "metric_avg",
    # Tools
    "dict2vector", "dict2sample",
    # Distributions
    "SingleGaussian", "MixedGaussian", "MinMaxBinning", "RangeBinning", 
    # Manager & dashboard
    "FileManager", "QGANDashboard",
]
