"""Sequence classifiers over landmark features of shape (B, T, FEATURE_DIM)."""
import torch
import torch.nn as nn


class SignGRU(nn.Module):
    """Baseline: projection -> 2-layer BiGRU -> temporal attention pooling."""

    def __init__(self, in_dim, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2))
        self.gru = nn.GRU(256, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        h, _ = self.gru(self.proj(x))
        w = torch.softmax(self.att(h), dim=1)
        return self.head((w * h).sum(dim=1))


class SignTransformer(nn.Module):
    """Comparison model: Transformer encoder with learned positions and mean pooling."""

    def __init__(self, in_dim, n_classes, d_model=192, heads=4, layers=4, max_len=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(z).mean(dim=1))


class SignCNN(nn.Module):
    """Comparison model: temporal 1D convolutions with global pooling."""

    def __init__(self, in_dim, n_classes, ch=192, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch, n_classes))

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).mean(dim=2))


MODELS = {"gru": SignGRU, "transformer": SignTransformer, "cnn": SignCNN}


def build_model(name, in_dim, n_classes):
    return MODELS[name](in_dim, n_classes)
