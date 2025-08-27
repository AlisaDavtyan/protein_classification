# # utils.py
# import warnings
# import os
# import logging
# import tomli

# def load_token():
#     # First, try the secret.toml in the project root (for development)
#     project_root = os.path.dirname(os.path.dirname(__file__))  # one level above amylodeep/
#     secret_path = os.path.join(project_root, "secret.toml")

#     if os.path.exists(secret_path):
#         with open(secret_path, "rb") as f:
#             return tomli.load(f).get("HF_TOKEN")

#     # Optional: fallback to a hidden config directory
#     fallback_path = os.path.expanduser("~/.config/amylodeep/secret.toml")
#     if os.path.exists(fallback_path):
#         with open(fallback_path, "rb") as f:
#             return tomli.load(f).get("HF_TOKEN")

#     # Token not found
#     return None

# # Suppress all warnings and progress bars
# warnings.filterwarnings('ignore')
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Disable logging from various libraries
# logging.getLogger("transformers").setLevel(logging.CRITICAL)
# logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
# logging.getLogger("tokenizers").setLevel(logging.CRITICAL)

# from .model_downloaded import ensure_models_downloaded, is_downloaded
# from .unirep_model import UniRepClassifier
# from .esm_classifier import ESMClassifier
# from .ensemble_predictor import EnsembleRollingWindowPredictor  

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from huggingface_hub import hf_hub_download
# import pickle
# import xgboost as xgb

# REPO_ID = "AlisaDavtyan/amylodeep-models"
# MODEL_ROOT = os.path.expanduser("~/.amylodeep_models")

# # Module-level cache to ensure models are loaded only once per session
# _models_cache = None
# _calibrators_cache = None
# _tokenizer_cache = None

# def get_model_file(filename, subfolder, token):
#     """Download model file from Hugging Face Hub (cached)"""
#     return hf_hub_download(
#         repo_id=REPO_ID,
#         filename=filename,
#         subfolder=subfolder,
#         token=token,
#         cache_dir=MODEL_ROOT
#     )

# def load_models_and_calibrators():
#     """
#     Load models and calibrators from Hugging Face Hub (cached for performance)
#     """
#     global _models_cache, _calibrators_cache, _tokenizer_cache
    
#     # Return cached models if already loaded
#     if all(cache is not None for cache in [_models_cache, _calibrators_cache, _tokenizer_cache]):
#         return _models_cache, _calibrators_cache, _tokenizer_cache
    
#     # Ensure models are downloaded and authenticated
#     ensure_models_downloaded()
#     hf_token = load_token()
    
#     # Suppress warnings during model loading
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
        
#         models = {}
        
#         # Load ESM2 150M model and tokenizer
#         models['esm2_150M'] = AutoModelForSequenceClassification.from_pretrained(
#             REPO_ID,
#             subfolder="esm2_150M",
#             token=hf_token,
#             cache_dir=MODEL_ROOT
#         )
#         tokenizer_1 = AutoTokenizer.from_pretrained(
#             REPO_ID,
#             subfolder="esm2_150M", 
#             token=hf_token,
#             cache_dir=MODEL_ROOT
#         )

#         # Load UniRep model
#         models['unirep'] = UniRepClassifier.from_pretrained(
#             REPO_ID,
#             subfolder="unirep",
#             token=hf_token,
#             cache_dir=MODEL_ROOT
#         )

#         # Load ESM2 650M model
#         models['esm2_650M'] = ESMClassifier.from_pretrained(
#             REPO_ID,
#             subfolder="esm2_650M",
#             token=hf_token,
#             cache_dir=MODEL_ROOT
#         )

#         # Load SVM model
#         svm_model_path = get_model_file("svm_model.pkl", "svm", hf_token)
#         with open(svm_model_path, "rb") as f:
#             models['svm'] = pickle.load(f)

#         # Load XGBoost model
#         xgb_model_path = get_model_file("xgb_model.json", "xgb", hf_token)
#         xgb_model = xgb.XGBClassifier()
#         xgb_model.load_model(xgb_model_path)
#         models['xgboost'] = xgb_model
        
#         # Load calibrators
#         calibrators = {}
        
#         # Platt calibrator for UniRep
#         platt_path = get_model_file("platt_unirep.pkl", "platt_unirep", hf_token)
#         with open(platt_path, "rb") as f:
#             calibrators['platt_unirep'] = pickle.load(f)

#         # Isotonic calibrator for ESM2 650M
#         isotonic_650_path = get_model_file("isotonic_650M_NN.pkl", "isotonic_650M_NN", hf_token)
#         with open(isotonic_650_path, "rb") as f:
#             calibrators['isotonic_650M_NN'] = pickle.load(f)

#         # Isotonic calibrator for XGBoost
#         isotonic_xgb_path = get_model_file("isotonic_XGBoost.pkl", "isotonic_XGBoost", hf_token)
#         with open(isotonic_xgb_path, "rb") as f:
#             calibrators['isotonic_XGBoost'] = pickle.load(f)
    
#     # Cache the loaded models
#     _models_cache = models
#     _calibrators_cache = calibrators
#     _tokenizer_cache = tokenizer_1
    
#     return models, calibrators, tokenizer_1

# def predict_ensemble_rolling(sequence: str, window_size: int = 6):
#     """
#     Run ensemble prediction with rolling window over a single sequence.
#     Returns dictionary with average/max probs and position-wise scores.
#     """
#     models, calibrators, tokenizer_1 = load_models_and_calibrators()
#     predictor = EnsembleRollingWindowPredictor(models, calibrators, tokenizer_1)
#     return predictor.rolling_window_prediction(sequence, window_size)

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python script.py SEQUENCE [WINDOW_SIZE]")
#         sys.exit(1)

#     sequence = sys.argv[1]
#     window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 6

#     print(f"Running ensemble prediction on sequence of length {len(sequence)} with window size {window_size}...\n")

#     result = predict_ensemble_rolling(sequence, window_size)

#     print("Result:")
#     print(f"  - Sequence Length: {result['sequence_length']}")
#     print(f"  - Num Windows:     {result['num_windows']}")
#     print(f"  - Avg Probability: {result['avg_probability']:.4f}")
#     print(f"  - Max Probability: {result['max_probability']:.4f}")
#     print(f"  - Top Positions:   {sorted(result['position_probs'], key=lambda x: x[1], reverse=True)[:5]}")

# utils.py
import warnings
import os
import logging

def load_token():
    """Load token - kept for backward compatibility but returns None since models are public"""
    return None

# Suppress all warnings and progress bars
warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable logging from various libraries
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("tokenizers").setLevel(logging.CRITICAL)

from .model_downloaded import ensure_models_downloaded, is_downloaded
from .unirep_model import UniRepClassifier
from .esm_classifier import ESMClassifier
from .ensemble_predictor import EnsembleRollingWindowPredictor  

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pickle
import xgboost as xgb

REPO_ID = "AlisaDavtyan/amylodeep-models"
MODEL_ROOT = os.path.expanduser("~/.amylodeep_models")

# Module-level cache to ensure models are loaded only once per session
_models_cache = None
_calibrators_cache = None
_tokenizer_cache = None

def get_model_file(filename, subfolder):
    """Download model file from Hugging Face Hub (cached) - no token needed for public models"""
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        subfolder=subfolder,
        cache_dir=MODEL_ROOT
        # Removed token parameter since models are public
    )

def load_models_and_calibrators():
    """
    Load models and calibrators from Hugging Face Hub (cached for performance)
    """
    global _models_cache, _calibrators_cache, _tokenizer_cache
    
    # Return cached models if already loaded
    if all(cache is not None for cache in [_models_cache, _calibrators_cache, _tokenizer_cache]):
        return _models_cache, _calibrators_cache, _tokenizer_cache
    
    # Ensure models are downloaded
    ensure_models_downloaded()
    
    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        models = {}
        
        # Load ESM2 150M model and tokenizer
        models['esm2_150M'] = AutoModelForSequenceClassification.from_pretrained(
            REPO_ID,
            subfolder="esm2_150M",
            cache_dir=MODEL_ROOT
            # Removed token parameter since models are public
        )
        tokenizer_1 = AutoTokenizer.from_pretrained(
            REPO_ID,
            subfolder="esm2_150M", 
            cache_dir=MODEL_ROOT
            # Removed token parameter since models are public
        )

        # Load UniRep model
        models['unirep'] = UniRepClassifier.from_pretrained(
            REPO_ID,
            subfolder="unirep",
            cache_dir=MODEL_ROOT
            # Removed token parameter since models are public
        )

        # Load ESM2 650M model
        models['esm2_650M'] = ESMClassifier.from_pretrained(
            REPO_ID,
            subfolder="esm2_650M",
            cache_dir=MODEL_ROOT
            # Removed token parameter since models are public
        )

        # Load SVM model
        svm_model_path = get_model_file("svm_model.pkl", "svm")
        with open(svm_model_path, "rb") as f:
            models['svm'] = pickle.load(f)

        # Load XGBoost model
        xgb_model_path = get_model_file("xgb_model.json", "xgb")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
        models['xgboost'] = xgb_model
        
        # Load calibrators
        calibrators = {}
        
        # Platt calibrator for UniRep
        platt_path = get_model_file("platt_unirep.pkl", "platt_unirep")
        with open(platt_path, "rb") as f:
            calibrators['platt_unirep'] = pickle.load(f)

        # Isotonic calibrator for ESM2 650M
        isotonic_650_path = get_model_file("isotonic_650M_NN.pkl", "isotonic_650M_NN")
        with open(isotonic_650_path, "rb") as f:
            calibrators['isotonic_650M_NN'] = pickle.load(f)

        # Isotonic calibrator for XGBoost
        isotonic_xgb_path = get_model_file("isotonic_XGBoost.pkl", "isotonic_XGBoost")
        with open(isotonic_xgb_path, "rb") as f:
            calibrators['isotonic_XGBoost'] = pickle.load(f)
    
    # Cache the loaded models
    _models_cache = models
    _calibrators_cache = calibrators
    _tokenizer_cache = tokenizer_1
    
    return models, calibrators, tokenizer_1

def predict_ensemble_rolling(sequence: str, window_size: int = 6):
    """
    Run ensemble prediction with rolling window over a single sequence.
    Returns dictionary with average/max probs and position-wise scores.
    """
    models, calibrators, tokenizer_1 = load_models_and_calibrators()
    predictor = EnsembleRollingWindowPredictor(models, calibrators, tokenizer_1)
    return predictor.rolling_window_prediction(sequence, window_size)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py SEQUENCE [WINDOW_SIZE]")
        sys.exit(1)

    sequence = sys.argv[1]
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"Running ensemble prediction on sequence of length {len(sequence)} with window size {window_size}...\n")

    result = predict_ensemble_rolling(sequence, window_size)

    print("Result:")
    print(f"  - Sequence Length: {result['sequence_length']}")
    print(f"  - Num Windows:     {result['num_windows']}")
    print(f"  - Avg Probability: {result['avg_probability']:.4f}")
    print(f"  - Max Probability: {result['max_probability']:.4f}")
    print(f"  - Top Positions:   {sorted(result['position_probs'], key=lambda x: x[1], reverse=True)[:5]}")