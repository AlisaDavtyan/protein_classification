# amylodeep/model_downloaded.py

import os
import logging
import warnings
from huggingface_hub import snapshot_download, try_to_load_from_cache, login
import tomli

# Suppress all warnings and progress bars
warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable logging
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

REPO_ID = "AlisaDavtyan/amylodeep-models"
CACHE_DIR = os.path.expanduser("~/.amylodeep_models")

FILES = [
    "esm2_150M",
    "esm2_650M", 
    "isotonic_650M_NN",
    "isotonic_XGBoost",
    "platt_unirep",
    "svm",
    "unirep",
    "xgb"
]

def load_token_from_secret():
    """Load Hugging Face token from secret.toml"""
    # Look for secret.toml in the package root
    package_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(package_root, "secret.toml")
    if not os.path.exists(config_path):
        raise RuntimeError(
            "❌ Missing 'secret.toml' file in package directory.\n"
            "Please create it with:\n\nHF_TOKEN = \"your_read_only_token\""
        )
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        token = config.get("HF_TOKEN")
        if not token:
            raise RuntimeError("❌ 'HF_TOKEN' not found in secret.toml")
        return token
    except Exception as e:
        raise RuntimeError(f"❌ Error reading secret.toml: {e}")

# def authenticate_hf():
#     """Authenticate with Hugging Face Hub"""
#     try:
#         token = load_token_from_secret()
#         login(token=token, write_permission=False)
#         os.environ["HF_TOKEN"] = token
#         os.environ["HUGGING_FACE_HUB_TOKEN"] = token
#         return token
#     except Exception as e:
#         raise RuntimeError(f"❌ HuggingFace authentication failed: {e}")

def is_downloaded() -> bool:
    """Check if models are already cached by testing a few key files"""
    try:
        # Check if cache directory exists and has some content
        if not os.path.exists(CACHE_DIR):
            return False
            
        # Check for a few key model files
        key_files = ["esm2_150M", "unirep", "svm"]
        for file in key_files:
            path = try_to_load_from_cache(REPO_ID, file)
            if path is None or not os.path.exists(path):
                return False
        return True
    except Exception:
        return False

def ensure_models_downloaded():
    """Download all model files silently if not already cached"""
    if is_downloaded():
        return
        
    # Authenticate first
    token = authenticate_hf()
    
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.lock"],
            token=token
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download models: {e}")