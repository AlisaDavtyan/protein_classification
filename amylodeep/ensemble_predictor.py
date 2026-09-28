import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import jax_unirep
import pickle
import os

from .projection import project_to_residues



class EnsembleRollingWindowPredictor:
    def __init__(
        self,
        models_dict,
        calibrators_dict=None,
        tokenizer=None,
        *,
        model_dir=None,
        allow_download=True,
    ):
        """
        Initialize the ensemble predictor with all 5 models and calibrators.

        Args:
            models_dict: Dictionary containing all 5 models with keys:
                'esm2_150M', 'unirep', 'esm2_650M', 'svm', 'xgboost'
            calibrators_dict: Dictionary containing calibrators where applicable
            model_dir: Directory holding the artefacts. When it contains
                ``esm2_t33_650M_UR50D/``, the frozen encoder is loaded from there
                instead of the Hugging Face hub, which is what makes an offline or
                version-pinned run possible.
            allow_download: When False, a missing local encoder raises instead of
                fetching ~2.6 GB.
        """
        self.models = models_dict
        self.calibrators = calibrators_dict or {}
        self.tokenizer_1 = tokenizer

        from .loading import encoder_source

        source = encoder_source(model_dir, allow_download=allow_download)
        self.encoder_source = source
        self.tokenizer_esm = AutoTokenizer.from_pretrained(source)
        self.esm_model = AutoModel.from_pretrained(source, add_pooling_layer=False)
        # Mean-pooled ESM2-650M features are used by THREE heads (the 650M
        # classifier, the SVM and XGBoost). They were recomputed for each, so the
        # 650M encoder ran three times per call -- on a 334-residue protein at
        # window 10 that is 975 forward passes where 325 suffice. Cached per
        # predict_ensemble() call, keyed on the sequences themselves so a stale
        # cache cannot be read.
        self._feature_cache_key = None
        self._feature_cache = None

        # Freeze ESM model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.esm_model.eval()

    def _predict_model_1(self, sequences):
        """ESM2 150M fine-tuned model prediction"""
        def tokenize_function(sequences):
            return self.tokenizer_1(sequences, padding="max_length", truncation=True, max_length=128)

        encodings = tokenize_function(sequences)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        with torch.no_grad():
            outputs = self.models['esm2_150M'](input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]

        return probs.numpy()
    
    def _predict_model_2(self, sequences):
        """UniRep model prediction"""
        def unirep_tokenize_function(sequences):
            h_final, c_final, h_avg = jax_unirep.get_reps(sequences)
            return {
                "embeddings": h_final,
                "avg_hidden": h_avg,
                "cell_state": c_final
            }

        encodings = unirep_tokenize_function(sequences)
        embeddings = torch.tensor(encodings["embeddings"], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.models['unirep'](embeddings=embeddings)
            probs = F.softmax(outputs['logits'], dim=1)[:, 1]

        probs_np = probs.numpy()

        # Apply calibration if available
        if 'platt_unirep' in self.calibrators:
            probs_np = self.calibrators['platt_unirep'].predict_proba(probs_np.reshape(-1, 1))[:, 1]

        return probs_np
    
    def _extract_mean_esm_embeddings(self, encodings, batch_size=8):
        """Shared helper to extract mean-pooled ESM2-650M embeddings."""
        embeddings = []
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        dataset_size = input_ids.size(0)

        with torch.no_grad():
            for i in range(0, dataset_size, batch_size):
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                outputs = self.esm_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                sequence_output = outputs.last_hidden_state
                mask_expanded = batch_attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                embeddings.append(mean_embeddings)

        return torch.cat(embeddings, dim=0)
    
    def _predict_model_3(self, sequences):
        """ESM2 650M with custom classifier prediction"""
        # Was a byte-identical private copy of _extract_mean_esm_embeddings; the
        # shared, cached version keeps the three heads on one set of features.
        embeddings = self._mean_esm_features(sequences)

        with torch.no_grad():
            outputs = self.models['esm2_650M'](embeddings=embeddings)
            probs = F.softmax(outputs['logits'], dim=1)[:, 1]

        probs_np = probs.numpy()

        if 'isotonic_650M_NN' in self.calibrators:
            probs_np = self.calibrators['isotonic_650M_NN'].predict(probs_np)

        return probs_np

    def _predict_model_4(self, sequences):
        """SVM model prediction"""
        X_features = self._extract_features_for_svm(sequences)

        probs = self.models['svm'].predict_proba(X_features)[:, 1]
        return probs

    def _predict_model_5(self, sequences):
        """XGBoost model prediction"""
        X_features = self._extract_features_for_xgboost(sequences)

        model = self.models['xgboost']
        if hasattr(model, "predict_proba"):
            # An XGBClassifier, as earlier versions loaded.
            probs = model.predict_proba(X_features)[:, 1]
        else:
            # A bare Booster. The objective is binary:logistic, so predict()
            # returns P(class=1) -- the same number predict_proba(X)[:, 1] gives,
            # without the sklearn wrapper's cross-version JSON attribute parsing.
            import xgboost

            probs = model.predict(xgboost.DMatrix(X_features))

        if 'isotonic_XGBoost' in self.calibrators:
            probs = self.calibrators['isotonic_XGBoost'].predict(probs)

        return probs

    def _mean_esm_features(self, sequences):
        """Mean-pooled ESM2-650M embeddings, computed once per sequence list."""
        key = (len(sequences), tuple(sequences[:4]), tuple(sequences[-4:]))
        if self._feature_cache_key == key and self._feature_cache is not None:
            return self._feature_cache
        tokenized = self.tokenizer_esm(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self._extract_mean_esm_embeddings(tokenized)
        self._feature_cache_key = key
        self._feature_cache = embeddings
        return embeddings

    def _extract_features_for_svm(self, sequences):
        """Extract ESM2-650M mean-pooled embeddings for SVM."""
        return self._mean_esm_features(sequences).numpy()

    def _extract_features_for_xgboost(self, sequences):
        """Extract ESM2-650M mean-pooled embeddings for XGBoost."""
        return self._extract_features_for_svm(sequences)  # same as SVM

    def predict_ensemble(self, sequences):
        """
        Predict ensemble probabilities for a list of sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            numpy array of ensemble probabilities
        """
        # Order preserved from the original implementation.
        mixed_probs_list = [
            self._predict_model_1(sequences),  # ESM2 150M  - no calibration
            self._predict_model_2(sequences),  # UniRep     - platt_unirep
            self._predict_model_3(sequences),  # ESM2 650M  - isotonic_650M_NN
            self._predict_model_4(sequences),  # SVM        - no calibration
        ]
        # The XGBoost head is absent when it was skipped at load time. The average
        # is then over four heads, not five: dividing by five would silently pull
        # every probability toward zero, which is worse than a smaller ensemble.
        if "xgboost" in self.models:
            mixed_probs_list.append(self._predict_model_5(sequences))

        return np.mean(mixed_probs_list, axis=0)

    @property
    def heads_used(self) -> int:
        """How many models the average is over. Belongs in any output."""
        return 4 + (1 if "xgboost" in self.models else 0)

    def rolling_window_prediction(self, sequence, window_size, aggregate="mean"):
        """
        Predict amyloid probability for an entire sequence using rolling window approach.
        The window slides one position at a time across the sequence.

        Args:
            sequence: Single protein sequence string
            window_size: Size of the sliding window

        Args:
            sequence: Single protein sequence string
            window_size: Size of the sliding window
            aggregate: How overlapping windows are combined into a per-residue
                value -- "mean" (default), "max" or "support". See
                :func:`project_to_residues`.

        Returns:
            dict containing:
                - 'position_probs': List of (window start, probability) tuples,
                  0-based, one entry per window
                - 'residue_probs': One probability per RESIDUE, aggregated over
                  the windows covering it, index 0 == residue 1
                - 'coverage_depth': How many windows covered each residue
                - 'avg_probability', 'max_probability': whole-sequence summaries
                - 'sequence_length', 'num_windows', 'window_size', 'aggregate'
        """
        sequence_length = len(sequence)

        if sequence_length < window_size:
            # If sequence is shorter than window, predict on the entire sequence
            prob = self.predict_ensemble([sequence])[0]
            return {
                'position_probs': [(0, prob)],
                'avg_probability': prob,
                'max_probability': prob,
                'sequence_length': sequence_length,
                'num_windows': 1,
                'window_size': sequence_length,
                'aggregate': aggregate,
                'heads_used': self.heads_used,
                # One window covering everything: every residue takes that score,
                # and depth 1 records that nothing was averaged.
                'residue_probs': [float(prob)] * sequence_length,
                'coverage_depth': [1] * sequence_length,
            }

        # Generate windows - slide one position at a time
        windows = []
        positions = []

        for i in range(sequence_length - window_size + 1):
            window = sequence[i:i + window_size]
            windows.append(window)
            positions.append(i)

        # Predict on all windows
        window_probs = self.predict_ensemble(windows)

        # Combine results
        position_probs = list(zip(positions, window_probs))
        avg_probability = np.mean(window_probs)
        max_probability = np.max(window_probs)

        residues = project_to_residues(
            position_probs, sequence_length, window_size, aggregate=aggregate
        )

        return {
            'position_probs': position_probs,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'sequence_length': sequence_length,
            'num_windows': len(windows),
            'window_size': window_size,
            'aggregate': aggregate,
            'heads_used': self.heads_used,
            'residue_probs': residues['residue_probs'],
            'coverage_depth': residues['coverage_depth'],
            }