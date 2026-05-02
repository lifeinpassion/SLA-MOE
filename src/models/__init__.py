# Models module
"""
Neural network models for EEG denoising.
"""
from .baselines import (
    EEGDnoiseNet,
    EEGDnet,
    RNNEEG,
    ResNetEEG,
    SimpleCNN,
    WienerFilter,
    LMSFilter,
    RLSFilter,
    KalmanFilter
)
from .moe_models import (
    RNNMoEFilter,
    ICAMoEFilter,
    ReservoirMoEFilter
)
