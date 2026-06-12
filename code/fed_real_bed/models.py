from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import torch_l2_normalize


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class CNNTCNEncoder(nn.Module):
    def __init__(self, cin: int = 14, emb_dim: int = 128, width: int = 96, tcn_blocks: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvBlock(cin, width, 7, dropout=dropout),
            ConvBlock(width, width, 5, dropout=dropout),
        ]
        for i in range(tcn_blocks):
            layers.append(ConvBlock(width, width, 3, dilation=2**i, dropout=dropout))
        self.features = nn.Sequential(*layers)
        self.attn = nn.Sequential(nn.Conv1d(width, 1, 1), nn.Softmax(dim=-1))
        self.proj = nn.Sequential(nn.Linear(width * 2, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        w = self.attn(h)
        mean = (h * w).sum(dim=-1)
        std = torch.sqrt(((h - mean.unsqueeze(-1)) ** 2 * w).sum(dim=-1).clamp_min(1e-6))
        return torch_l2_normalize(self.proj(torch.cat([mean, std], dim=1)))


class EEGNetEncoder(nn.Module):
    def __init__(self, cin: int = 14, emb_dim: int = 128, dropout: float = 0.25, samples: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(cin, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, cin, samples)).reshape(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Linear(flat, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.unsqueeze(1)).flatten(1)
        return torch_l2_normalize(self.proj(h))


class TransformerEncoder(nn.Module):
    def __init__(self, cin: int = 14, emb_dim: int = 128, width: int = 96, layers: int = 2, heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.input = nn.Conv1d(cin, width, 7, padding=3)
        enc_layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.proj = nn.Sequential(nn.Linear(width * 2, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x).transpose(1, 2)
        h = self.encoder(h)
        pooled = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=1)
        return torch_l2_normalize(self.proj(pooled))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        return x * self.net(s).unsqueeze(-1)


class ECAPAEEGEncoder(nn.Module):
    def __init__(self, cin: int = 14, emb_dim: int = 128, width: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = ConvBlock(cin, width, 5, dropout=dropout)
        self.b1 = nn.Sequential(ConvBlock(width, width, 3, dilation=2, dropout=dropout), SEBlock(width))
        self.b2 = nn.Sequential(ConvBlock(width, width, 3, dilation=3, dropout=dropout), SEBlock(width))
        self.b3 = nn.Sequential(ConvBlock(width, width, 3, dilation=4, dropout=dropout), SEBlock(width))
        self.mix = nn.Conv1d(width * 3, width, 1)
        self.attn = nn.Sequential(nn.Conv1d(width, width // 2, 1), nn.Tanh(), nn.Conv1d(width // 2, 1, 1), nn.Softmax(dim=-1))
        self.proj = nn.Sequential(nn.Linear(width * 2, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h1 = self.b1(h)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        z = self.mix(torch.cat([h1, h2, h3], dim=1))
        a = self.attn(z)
        mean = (z * a).sum(dim=-1)
        std = torch.sqrt(((z - mean.unsqueeze(-1)) ** 2 * a).sum(dim=-1).clamp_min(1e-6))
        return torch_l2_normalize(self.proj(torch.cat([mean, std], dim=1)))


class ECAPAConformerEEGEncoder(nn.Module):
    """ECAPA front end plus lightweight self-attention for noisy EEG identity cues."""

    def __init__(
        self,
        cin: int = 14,
        emb_dim: int = 128,
        width: int = 128,
        samples: int = 256,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, samples, width))
        self.stem = ConvBlock(cin, width, 5, dropout=dropout)
        self.b1 = nn.Sequential(ConvBlock(width, width, 3, dilation=2, dropout=dropout), SEBlock(width))
        self.b2 = nn.Sequential(ConvBlock(width, width, 3, dilation=3, dropout=dropout), SEBlock(width))
        self.b3 = nn.Sequential(ConvBlock(width, width, 3, dilation=4, dropout=dropout), SEBlock(width))
        self.mix = nn.Sequential(nn.Conv1d(width * 3, width, 1, bias=False), nn.BatchNorm1d(width), nn.ELU())
        self.local = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2, groups=width, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.attn = nn.Sequential(
            nn.Conv1d(width, width // 2, 1),
            nn.Tanh(),
            nn.Conv1d(width // 2, 1, 1),
            nn.Softmax(dim=-1),
        )
        self.proj = nn.Sequential(nn.Linear(width * 2, emb_dim), nn.LayerNorm(emb_dim))

    def _positional_encoding(self, timesteps: int) -> torch.Tensor:
        if timesteps == self.pos.shape[1]:
            return self.pos
        return F.interpolate(
            self.pos.transpose(1, 2),
            size=timesteps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h1 = self.b1(h)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        z = self.mix(torch.cat([h1, h2, h3], dim=1))
        z = z + self.local(z)
        seq = z.transpose(1, 2)
        pos = self._positional_encoding(seq.shape[1]).to(device=seq.device, dtype=seq.dtype)
        seq = self.encoder(seq + pos)
        z = seq.transpose(1, 2)
        a = self.attn(z)
        mean = (z * a).sum(dim=-1)
        std = torch.sqrt(((z - mean.unsqueeze(-1)) ** 2 * a).sum(dim=-1).clamp_min(1e-6))
        return torch_l2_normalize(self.proj(torch.cat([mean, std], dim=1)))


class DOMCSEncoder(nn.Module):
    """DOMCS-style identity encoder from the previous BED notebook.

    The old notebook also included a state branch, but its state labels were not
    trainable under r01+r02 -> r03 because training saw only one state class.
    This encoder keeps the identity-relevant backbone/head for a clean protocol
    reproduction without adding a misleading state loss.
    """

    def __init__(self, cin: int = 14, emb_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(cin, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.id_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)
        return torch_l2_normalize(self.id_head(h))


class MarginClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, kind: str = "arcface", scale: float = 30.0, margin: float = 0.35, subcenters: int = 1) -> None:
        super().__init__()
        self.kind = kind.lower()
        self.scale = scale
        self.margin = margin
        self.subcenters = max(1, int(subcenters if self.kind in {"subcenter_arcface", "subcenter_adaface"} else 1))
        self.weight = nn.Parameter(torch.randn(num_classes * self.subcenters, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.num_classes = num_classes

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None, quality: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        if self.subcenters > 1:
            cosine = cosine.view(-1, self.num_classes, self.subcenters).max(dim=2).values
        if labels is None or self.kind == "softmax":
            return cosine * self.scale
        margin = self.margin
        if self.kind in {"adaface", "subcenter_adaface"} and quality is not None:
            q = quality.clamp(0.0, 1.0).view(-1, 1)
            margin = margin * (0.5 + q)
        theta = cosine.clamp(-1 + 1e-7, 1 - 1e-7).acos()
        target = torch.cos(theta + margin)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.scale


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(emb, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        logits_mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        label_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & logits_mask
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim) * logits_mask.float()
        log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-12))
        denom = label_mask.sum(dim=1).clamp_min(1)
        loss = -(log_prob * label_mask.float()).sum(dim=1) / denom
        return loss.mean()


class EEGAuthenticator(nn.Module):
    def __init__(self, cfg: dict, samples: int) -> None:
        super().__init__()
        mcfg = cfg["model"]
        cin = len(cfg["data"]["channels"])
        emb_dim = int(mcfg["emb_dim"])
        enc = str(mcfg["encoder"]).lower()
        if enc == "cnn_tcn":
            self.encoder = CNNTCNEncoder(cin, emb_dim, int(mcfg["width"]), int(mcfg["tcn_blocks"]), float(mcfg["dropout"]))
        elif enc == "eegnet":
            self.encoder = EEGNetEncoder(cin, emb_dim, float(mcfg["dropout"]), samples)
        elif enc == "transformer":
            self.encoder = TransformerEncoder(cin, emb_dim, int(mcfg["width"]), int(mcfg["transformer_layers"]), int(mcfg["transformer_heads"]), float(mcfg["dropout"]))
        elif enc == "ecapa":
            self.encoder = ECAPAEEGEncoder(cin, emb_dim, int(mcfg["width"]), float(mcfg["dropout"]))
        elif enc in {"ecapa_conformer", "conformer_ecapa"}:
            self.encoder = ECAPAConformerEEGEncoder(
                cin,
                emb_dim,
                int(mcfg["width"]),
                samples,
                int(mcfg["transformer_layers"]),
                int(mcfg["transformer_heads"]),
                float(mcfg["dropout"]),
            )
        elif enc == "domcs":
            self.encoder = DOMCSEncoder(cin, emb_dim, float(mcfg["dropout"]))
        else:
            raise ValueError(f"Unsupported encoder: {enc}")
        lcfg = cfg["loss"]
        self.classifier = MarginClassifier(
            emb_dim=emb_dim,
            num_classes=int(mcfg["num_subjects"]),
            kind=str(lcfg["classifier"]),
            scale=float(lcfg["arcface_scale"]),
            margin=float(lcfg["arcface_margin"]),
            subcenters=int(lcfg.get("subcenters", 1)),
        )
        self.supcon = SupConLoss(float(lcfg["supcon_temperature"]))

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, quality: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        logits = self.classifier(emb, labels, quality)
        return emb, logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
