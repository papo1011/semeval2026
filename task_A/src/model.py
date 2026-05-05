import torch
import torch.nn as nn
from transformers import AutoModel


# Stage 1: Encoder
class UniXcoderSupCon(nn.Module):
    def __init__(self, model_name="microsoft/unixcoder-base-nine", projection_dim=128):
        super(UniXcoderSupCon, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # The Projection Head (MLP)
        # Maps the 768-D UniXcoder embedding down to 128-D for contrastive loss
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        # Get raw embeddings from UniXcoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token equivalent representation
        # For UniXcoder/RoBERTa, this is the first token of the sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Pass through projection head to get the vector for SupConLoss
        projected_embedding = self.projection_head(cls_embedding)

        return projected_embedding


# Stage 2: Classifier
class LinearClassifier(nn.Module):
    """The tiny, lightning-fast Stage 2 classification head."""

    def __init__(self, input_dim=128, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Fused Model
class UniXcoderSupConClassifier(nn.Module):
    """
    Fused deployment artifact combining frozen SupCon representation learning
    with the trained linear classification probe.
    """

    def __init__(
        self,
        model_name="microsoft/unixcoder-base-nine",
        projection_dim=128,
        num_classes=2,
    ):
        super(UniXcoderSupConClassifier, self).__init__()

        # We reuse the classes defined above!
        self.encoder = UniXcoderSupCon(
            model_name=model_name, projection_dim=projection_dim
        )
        self.classifier = LinearClassifier(
            input_dim=projection_dim, num_classes=num_classes
        )

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        logits = self.classifier(features)
        return logits
