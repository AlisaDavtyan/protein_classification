from unirep_model import UniRepClassifier
from esm_classifier import ESMClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pickle
from ensemble_predictor import EnsembleRollingWindowPredictor  
import xgboost as xgb
import wandb
import os
os.environ["WANDB_MODE"] = "disabled"

def load_models_and_calibrators():
    """
    Load models and calibrators
    """
    models = {}

    #initialize wandb api
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])

    # Model 1: ESM2 150M fine-tuned
    artifact_1 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_esm2_150M_checkpoint_100_epochs:v0')
    model_path_1 = artifact_1.download()
    models['esm2_150M'] = AutoModelForSequenceClassification.from_pretrained(model_path_1)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_path_1)

    # Model 2: UniRep classifier
    artifact_2 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_UniRepClassifier_4_layers_50_epochs:v0')
    model_path_2 = artifact_2.download()
    models['unirep'] = UniRepClassifier.from_pretrained(model_path_2)

    # Model 3: ESM2 650M classifier
    artifact_3 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/final_ESMClassifier_650_layers_50_epochs:v0')
    model_path_3 = artifact_3.download()
    models['esm2_650M'] = ESMClassifier.from_pretrained(model_path_3)

    # Model 4: SVM model
    artifact_4 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/svm_model:v0')
    model_path_4 = artifact_4.download()
    model_path_4_join = os.path.join(model_path_4, "svm_model.pkl")

    with open(model_path_4_join, "rb") as f:
        models['svm'] = pickle.load(f)

    # Model 5: XGBoost model
    artifact_5 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/XGBoost:v0')
    model_path_5 = artifact_5.download()
    model_path_5_join = os.path.join(model_path_5, "xgb_model.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path_5_join)
    models['xgboost'] = xgb_model
    

    # Calibrators
    calibrators = {}

    # platt_unirep
    artifact_p1 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/platt_unirep:v0')
    model_path_p1 = artifact_p1.download()
    calibrator_path_p1 = os.path.join(model_path_p1, "platt_unirep.pkl")
    with open(calibrator_path_p1, "rb") as f:
        calibrators['platt_unirep'] = pickle.load(f)

    # isotonic_650M_NN
    artifact_p2 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/isotonic_650M_NN:v0')
    model_path_p2 = artifact_p2.download()
    calibrator_path_p2 = os.path.join(model_path_p2, "isotonic_650M_NN.pkl")
    with open(calibrator_path_p2, "rb") as f:
        calibrators['isotonic_650M_NN'] = pickle.load(f)

    # isotonic_XGBoost
    artifact_p3 = api.artifact('biophysarm-l-k-jordan-associates/amylodeep/isotonic_XGBoost:v0')
    model_path_p3 = artifact_p3.download()
    calibrator_path_p3 = os.path.join(model_path_p3, "isotonic_XGBoost.pkl")
    with open(calibrator_path_p3, "rb") as f:
        calibrators['isotonic_XGBoost'] = pickle.load(f)


    return models, calibrators,tokenizer_1


def predict_ensemble_rolling(sequence: str, window_size: int = 6):
    """
    Run ensemble prediction with rolling window over a single sequence.
    Returns dictionary with average/max probs and position-wise scores.
    """
    models, calibrators ,tokenizer_1 = load_models_and_calibrators()
    predictor = EnsembleRollingWindowPredictor(models, calibrators,tokenizer_1)
    return predictor.rolling_window_prediction(sequence, window_size)
