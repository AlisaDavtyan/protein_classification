import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig
import jax_unirep

# --- UniRep Tokenizer ---
def unirep_tokenize_function(sequences):
    """
    Get UniRep embeddings for a list of protein sequences.
    Returns a dictionary with embeddings compatible with PyTorch datasets.
    """
    h_final, c_final, h_avg = jax_unirep.get_reps(sequences)
    return {
        "embeddings": h_final,
        "avg_hidden": h_avg,
        "cell_state": c_final
    }

# --- Custom Dataset using UniRep ---
class UniRepProteinDataset(Dataset):
    def __init__(self, encodings, labels):
        self.embeddings = torch.tensor(encodings["embeddings"], dtype=torch.float32)
        self.avg_hidden = torch.tensor(encodings["avg_hidden"], dtype=torch.float32)
        self.cell_state = torch.tensor(encodings["cell_state"], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "avg_hidden": self.avg_hidden[idx],
            "cell_state": self.cell_state[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# --- Custom Config for UniRepClassifier ---
class UniRepClassifierConfig(PretrainedConfig):
    model_type = "unirep_classifier"

    def __init__(
        self,
        input_dim=1900,
        hidden_dims=[512, 128],
        num_labels=2,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_labels = num_labels
        self.dropout = dropout

# --- UniRep Classifier Model ---
class UniRepClassifier(PreTrainedModel):
    config_class = UniRepClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        dims = [config.input_dim] + config.hidden_dims
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], config.num_labels)

    def forward(self, embeddings=None, labels=None, **kwargs):
        features = self.feature_extractor(embeddings)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


