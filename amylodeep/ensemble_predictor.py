import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import jax_unirep
import pickle
import os


class EnsembleRollingWindowPredictor:
    def __init__(self, models_dict, calibrators_dict=None, tokenizer=None):
        """
        Initialize the ensemble predictor with all 5 models and calibrators.

        Args:
            models_dict: Dictionary containing all 5 models with keys:
                'esm2_150M', 'unirep', 'esm2_650M', 'svm', 'xgboost'
            calibrators_dict: Dictionary containing calibrators where applicable
        """
        self.models = models_dict
        self.calibrators = calibrators_dict or {}
        self.tokenizer_1 = tokenizer

        # Load ESM2 650M model without token (public model)
        self.tokenizer_esm = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = AutoModel.from_pretrained(
            "facebook/esm2_t33_650M_UR50D", 
            add_pooling_layer=False
        )

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
        def tokenize_function(sequences):
            return self.tokenizer_esm(sequences, padding="max_length", truncation=True,
                                    max_length=128, return_tensors="pt")

        def extract_esm_embeddings(encodings, batch_size=8):
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

        encodings = tokenize_function(sequences)
        embeddings = extract_esm_embeddings(encodings)

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

        probs = self.models['xgboost'].predict_proba(X_features)[:, 1]

        if 'isotonic_XGBoost' in self.calibrators:
            probs = self.calibrators['isotonic_XGBoost'].predict(probs)

        return probs

    def _extract_features_for_svm(self, sequences):
        """Extract ESM2-650M mean-pooled embeddings for SVM."""
        tokenized = self.tokenizer_esm(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self._extract_mean_esm_embeddings(tokenized)
        return embeddings.numpy()

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
        # Get predictions from all models
        probs_1 = self._predict_model_1(sequences)  # ESM2 150M - NO calibration
        probs_2 = self._predict_model_2(sequences)  # UniRep - WITH calibration (platt_unirep)
        probs_3 = self._predict_model_3(sequences)  # ESM2 650M - WITH calibration (isotonic_650M_NN)
        probs_4 = self._predict_model_4(sequences)  # SVM - NO calibration
        probs_5 = self._predict_model_5(sequences)  # XGBoost - WITH calibration (isotonic_XGBoost)

        # Combine probabilities (matching your original mixed_probs_list order)
        mixed_probs_list = [probs_1, probs_2, probs_3, probs_4, probs_5]

        # Compute average probabilities
        avg_probs = np.mean(mixed_probs_list, axis=0)

        return avg_probs

    def rolling_window_prediction(self, sequence, window_size):
        """
        Predict amyloid probability for an entire sequence using rolling window approach.
        The window slides one position at a time across the sequence.

        Args:
            sequence: Single protein sequence string
            window_size: Size of the sliding window

        Returns:
            dict containing:
                - 'position_probs': List of (position, probability) tuples
                - 'avg_probability': Average probability across all windows
                - 'max_probability': Maximum probability across all windows
                - 'sequence_length': Length of the input sequence
        """
        sequence_length = len(sequence)

        if sequence_length < window_size:
            # If sequence is shorter than window, predict on the entire sequence
            prob = self.predict_ensemble([sequence])[0]
            return {
                'position_probs': [(0, prob)],
                'avg_probability': prob,
                'max_probability': prob,
                'sequence_length': sequence_length
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

        return {
            'position_probs': position_probs,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'sequence_length': sequence_length,
            'num_windows': len(windows)
            }