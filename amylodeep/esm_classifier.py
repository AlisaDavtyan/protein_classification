# Define a custom configuration for ESM embeddings
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class ESMClassifierConfig(PretrainedConfig):
    model_type = "esm_classifier"

    def __init__(
        self,
        input_dim=1280,  # ESM2_t30_150M has 640 dim embeddings
        hidden_dims=[2056, 1024, 512, 256, 128],
        num_labels=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_labels = num_labels
        self.dropout = dropout

# Define a custom model that works with the Trainer
class ESMClassifier(PreTrainedModel):
    config_class = ESMClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        layers = []
        dims = [config.input_dim] + config.hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], config.num_labels)

    def forward(
        self,
        embeddings=None,  # This will be your ESM embeddings
        labels=None,
        **kwargs
    ):
        # Process embeddings
        features = self.feature_extractor(embeddings)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
