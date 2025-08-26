"""
Amyloid: Prediction of amyloid propensity from amino acid sequences using deep learning

This package provides an ensemblef machine learning models to predict
amyloidogenic regions in protein sequences using a rolling window approach.
"""

__version__ = "0.2.6"
__author__ = "Alisa Davtyan"
__email__ = "alisadavtyan7@gmail.com"

# Import functions first (before model downloading)
from .utils import predict_ensemble_rolling, load_models_and_calibrators
from .ensemble_predictor import EnsembleRollingWindowPredictor
from .esm_classifier import ESMClassifier, ESMClassifierConfig
from .unirep_model import UniRepClassifier, UniRepClassifierConfig

# Import model downloading functions
from .model_downloaded import is_downloaded, ensure_models_downloaded


try:
    if not is_downloaded():
        ensure_models_downloaded()
except Exception as e:
    print(f"Warning: Could not download models automatically: {e}")
    print("You may need to call ensure_models_downloaded() manually.")

__all__ = [
    "predict_ensemble_rolling",
    "load_models_and_calibrators", 
    "EnsembleRollingWindowPredictor",
    "ESMClassifier",
    "ESMClassifierConfig", 
    "UniRepClassifier",
    "UniRepClassifierConfig",
    "is_downloaded",
    "ensure_models_downloaded"
]