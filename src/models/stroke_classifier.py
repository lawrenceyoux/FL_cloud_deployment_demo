"""
src/models/stroke_classifier.py
Model definitions for stroke prediction.

Three progressively richer architectures:
  1. StrokeNet          — simple MLP (mirrors local_dev/model.py)
  2. StrokeNetEmbeddings — MLP with a representation-learning encoder head
  3. AttentionFusion     — separate encoders per feature group + learned attention weights

local_dev/model.py is the authoritative prototype.  Any change there should
be reflected here once validated locally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Simple MLP  (identical logic to local_dev/model.py — cloud-packaged)
# ---------------------------------------------------------------------------
class StrokeNet(nn.Module):
    """Basic MLP for stroke prediction.  Input dim is auto-detected from data."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. MLP with representation-learning encoder (Phase 3 plan)
# ---------------------------------------------------------------------------
class StrokeNetEmbeddings(nn.Module):
    """MLP with a dedicated encoder that produces reusable embeddings.

    forward() returns (prediction, embedding) so that embeddings can be
    logged to MLflow or used for visualisation.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        pred = self.classifier(emb)
        return pred, emb

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 3. Attention-fusion model (separate per-modality encoders)
# ---------------------------------------------------------------------------
class AttentionFusion(nn.Module):
    """Learns attention weights over four feature groups:
       demographic | medical | lifestyle | clinical.

    forward() expects the full feature tensor and internally slices it.
    Column order must match the output of src/preprocessing/pipeline.py.
    """

    # Hard-coded feature-group slices that match pipeline.py output order:
    #   [age, gender, ever_married, Residence_type] → indices 0-3  (demographic)
    #   [hypertension, heart_disease]               → indices 4-5  (medical)
    #   [work_type_*, smoking_status_*]             → indices 6-11 (lifestyle, 6 cols)
    #   [avg_glucose_level, bmi]                    → indices 12-13 (clinical)
    DEMO_IDX   = slice(0, 4)
    MED_IDX    = slice(4, 6)
    LIFE_IDX   = slice(6, 12)
    CLIN_IDX   = slice(12, 14)

    def __init__(self, embed: int = 16):
        super().__init__()
        self.demo_enc  = nn.Linear(4, embed)
        self.med_enc   = nn.Linear(2, embed)
        self.life_enc  = nn.Linear(6, embed)
        self.clin_enc  = nn.Linear(2, embed)
        n_groups = 4
        self.attention = nn.Sequential(
            nn.Linear(n_groups * embed, 32),
            nn.Tanh(),
            nn.Linear(32, n_groups),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        demo = F.relu(self.demo_enc(x[:, self.DEMO_IDX]))
        med  = F.relu(self.med_enc(x[:, self.MED_IDX]))
        life = F.relu(self.life_enc(x[:, self.LIFE_IDX]))
        clin = F.relu(self.clin_enc(x[:, self.CLIN_IDX]))

        stacked = torch.stack([demo, med, life, clin], dim=1)  # (B, 4, embed)
        attn_w  = self.attention(stacked.flatten(start_dim=1)) # (B, 4)
        fused   = (stacked * attn_w.unsqueeze(-1)).sum(dim=1)  # (B, embed)

        pred = self.classifier(fused)
        return pred, attn_w


# ---------------------------------------------------------------------------
# Shared parameter helpers (used by both FL client and train_baseline)
# ---------------------------------------------------------------------------
def get_parameters(model: nn.Module) -> list:
    """Extract model weights as a list of numpy arrays."""
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Load a list of numpy arrays into model state dict."""
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state, strict=True)
