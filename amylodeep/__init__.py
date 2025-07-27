"""
Amyloid: Prediction of amyloid propensity from amino acid sequences using deep learning

This package provides an ensemble of machine learning models to predict
amyloidogenic regions in protein sequences using a rolling window approach.
"""

__version__ = "0.1.0"
__author__ = "Alisa Davtyan"
__email__ = "alisadavtyan7@gmail.com"

from .utils import predict_ensemble_rolling, load_models_and_calibrators
from .ensemble_predictor import EnsembleRollingWindowPredictor
from .esm_classifier import ESMClassifier, ESMClassifierConfig
from .unirep_model import UniRepClassifier, UniRepClassifierConfig

__all__ = [
    "predict_ensemble_rolling",
    "load_models_and_calibrators", 
    "EnsembleRollingWindowPredictor",
    "ESMClassifier",
    "ESMClassifierConfig",
    "UniRepClassifier",
    "UniRepClassifierConfig",
]