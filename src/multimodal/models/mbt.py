"""
src.multimodal.models.mbt
=========================

Multimodal Bottleneck Transformer (MBT) for dairy-cattle behaviour
classification from unsynchronised sensor + visual streams.

Architecture
------------

Visual stream  ─┐
                ├─► [independent unimodal layers]
Sensor stream  ─┘         │
                           ▼
                   Bottleneck fusion layers
                   (B tokens shared across streams)
                           │
                           ▼
                   Mean-pool B^(L)  →  MLP head  →  logits

Reference
---------
Nagrani et al., "Attention Bottlenecks for Multimodal Fusion", NeurIPS 2021.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class FourierPositionalEncoding(nn.Module):
    """Continuous Fourier positional encoding.

    Handles irregular / variable-length sequences gracefully — each token
    receives a sinusoidal encoding based on its *actual* timestamp rather
    than its position index, making it robust to the unsynchronised streams.

    Parameters
    ----------
    embed_dim:
        Must be even.  The encoding has ``embed_dim // 2`` sin and
        ``embed_dim // 2`` cos components.
    max_len:
        Upper bound on sequence length (used to pre-compute the frequency
        table); does not limit runtime length.
    """

    def __init__(self, embed_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for Fourier encoding"
        freqs = torch.pow(
            10_000.0, -torch.arange(0, embed_dim, 2).float() / embed_dim
        )
        self.register_buffer("freqs", freqs)   # (D/2,)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x:
            ``(B, T, D)`` token sequence.
        positions:
            ``(B, T)`` float tensor of normalised time positions in [0, 1].
            If ``None``, integer indices 0…T-1 are used (standard sinusoidal).

        Returns
        -------
        ``(B, T, D)`` with positional encoding added.
        """
        B, T, D = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device).float().unsqueeze(0).expand(B, -1)

        # positions: (B, T) → (B, T, 1) × (D/2,) → (B, T, D/2)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)   # (B, T, D)
        return x + pe


class LearnedPositionalEncoding(nn.Module):
    """Standard learned positional embedding (fixed maximum length)."""

    def __init__(self, embed_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_dim)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pe(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Encoder  (1-D CNN → linear projection)
# ─────────────────────────────────────────────────────────────────────────────

class SensorEncoder(nn.Module):
    """Convert a raw sensor window ``(B, C_in, T)`` → token sequence ``(B, T', D)``.

    A stack of 1-D convolutions progressively reduces the temporal dimension
    while expanding channel depth, then a linear projection maps each
    position to the shared embedding dimension ``D``.

    Parameters
    ----------
    input_dim:
        Number of raw input channels (e.g. 3 for UWB x/y/z).
    cnn_channels:
        List of output channels for each conv layer, e.g. ``[64, 128, 256]``.
    kernel_size:
        Kernel size for all conv layers (same padding applied).
    embed_dim:
        Target embedding dimension ``D``.
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: list[int],
        kernel_size: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(cnn_channels[-1], embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: ``(B, T, C_in)`` raw sensor window.

        Returns
        -------
        ``(B, T, D)`` sensor token sequence.
        """
        x = x.permute(0, 2, 1)           # → (B, C_in, T)
        x = self.cnn(x)                  # → (B, C_last, T)
        x = x.permute(0, 2, 1)           # → (B, T, C_last)
        return self.proj(x)              # → (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Encoder  (ViT backbone → token sequence)
# ─────────────────────────────────────────────────────────────────────────────

class VisualEncoder(nn.Module):
    """Wrap a pretrained ViT (via timm) and project its patch tokens to ``D``.

    The [CLS] token is discarded; we use all patch tokens so that the
    bottleneck can attend over spatial content.

    Parameters
    ----------
    backbone_name:
        timm model name, e.g. ``"vit_small_patch16_224"``.
    pretrained:
        Load pretrained ImageNet weights.
    freeze_layers:
        Number of ViT blocks to freeze (from the bottom up).
    embed_dim:
        Target embedding dimension ``D``.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        freeze_layers: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # remove the classification head
            global_pool="",      # return all patch tokens, not pooled
        )
        # Freeze specified blocks
        for i, block in enumerate(self.backbone.blocks):
            if i < freeze_layers:
                for p in block.parameters():
                    p.requires_grad = False

        vit_dim = self.backbone.embed_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: ``(B, 3, H, W)`` image batch.

        Returns
        -------
        ``(B, T_V, D)`` visual token sequence (T_V = number of patches).
        """
        tokens = self.backbone.forward_features(x)   # (B, T_V+1, vit_dim)
        tokens = tokens[:, 1:, :]                    # drop [CLS]
        return self.proj(tokens)                     # (B, T_V, D)


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckTransformerLayer(nn.Module):
    """One fusion layer of the MBT.

    Implements the three-step update from the paper:

    1. Visual  self-attention over [Z_V | B]
    2. Sensor  self-attention over [Z_S | B]
    3. B^(l+1) = mean( B_V^(l+1), B_S^(l+1) )

    Each stream uses independent projection weights but shares the same
    bottleneck token set ``B`` as input, enforcing the information bottleneck.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        # Independent attention + FFN per stream
        self.attn_v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_s = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn_v = self._make_ffn(embed_dim, ffn_dim, dropout)
        self.ffn_s = self._make_ffn(embed_dim, ffn_dim, dropout)

        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_s1 = nn.LayerNorm(embed_dim)
        self.norm_s2 = nn.LayerNorm(embed_dim)
        self.norm_b  = nn.LayerNorm(embed_dim)

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _make_ffn(embed_dim: int, ffn_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_v: Tensor,   # (B, T_V, D)
        z_s: Tensor,   # (B, T_S, D)
        b:   Tensor,   # (B, K,   D)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z_v_new : (B, T_V, D)
        z_s_new : (B, T_S, D)
        b_new   : (B, K,   D)
        """
        # ── Visual stream ─────────────────────────────────────────────────────
        # Concatenate visual tokens with bottleneck: (B, T_V + K, D)
        xv = torch.cat([z_v, b], dim=1)
        xv_res, _ = self.attn_v(xv, xv, xv)
        xv = self.norm_v1(xv + self.drop(xv_res))

        T_V = z_v.shape[1]
        z_v_new  = xv[:, :T_V, :]                  # (B, T_V, D)
        b_v_new  = xv[:, T_V:, :]                  # (B, K,   D)

        z_v_new = self.norm_v2(z_v_new + self.ffn_v(z_v_new))

        # ── Sensor stream ─────────────────────────────────────────────────────
        xs = torch.cat([z_s, b], dim=1)
        xs_res, _ = self.attn_s(xs, xs, xs)
        xs = self.norm_s1(xs + self.drop(xs_res))

        T_S = z_s.shape[1]
        z_s_new  = xs[:, :T_S, :]                  # (B, T_S, D)
        b_s_new  = xs[:, T_S:, :]                  # (B, K,   D)

        z_s_new = self.norm_s2(z_s_new + self.ffn_s(z_s_new))

        # ── Bottleneck update (average) ───────────────────────────────────────
        b_new = self.norm_b(0.5 * (b_v_new + b_s_new))

        return z_v_new, z_s_new, b_new


# ─────────────────────────────────────────────────────────────────────────────
# Unimodal Transformer Layer  (standard, before fusion begins)
# ─────────────────────────────────────────────────────────────────────────────

class UnimodalTransformerLayer(nn.Module):
    """Standard pre-LN transformer block used in the unimodal towers."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full MBT
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalBottleneckTransformer(nn.Module):
    """Full MBT model.

    Parameters
    ----------
    All parameters mirror the ``model:`` block in ``mbt_default.yaml``.
    Pass the config dict (or an OmegaConf DictConfig) directly via
    ``MultimodalBottleneckTransformer.from_config(cfg.model)``.
    """

    def __init__(
        self,
        # Shared
        embed_dim:              int   = 256,
        num_heads:              int   = 8,
        num_layers_unimodal:    int   = 3,
        num_layers_fusion:      int   = 3,
        num_bottleneck_tokens:  int   = 4,
        ffn_dim:                int   = 512,
        dropout:                float = 0.1,
        num_classes:            int   = 7,
        # Sensor
        sensor_input_dim:       int   = 3,
        sensor_cnn_channels:    list[int] | None = None,
        sensor_cnn_kernel:      int   = 3,
        # Visual
        visual_backbone:        str   = "vit_small_patch16_224",
        visual_pretrained:      bool  = True,
        visual_freeze_layers:   int   = 8,
        # Positional encoding
        pos_encoding:           str   = "fourier",
    ) -> None:
        super().__init__()

        if sensor_cnn_channels is None:
            sensor_cnn_channels = [64, 128, 256]

        # ── Encoders ──────────────────────────────────────────────────────────
        self.sensor_encoder = SensorEncoder(
            input_dim    = sensor_input_dim,
            cnn_channels = sensor_cnn_channels,
            kernel_size  = sensor_cnn_kernel,
            embed_dim    = embed_dim,
            dropout      = dropout,
        )
        self.visual_encoder = VisualEncoder(
            backbone_name  = visual_backbone,
            pretrained     = visual_pretrained,
            freeze_layers  = visual_freeze_layers,
            embed_dim      = embed_dim,
            dropout        = dropout,
        )

        # ── Positional encodings ───────────────────────────────────────────────
        PE = FourierPositionalEncoding if pos_encoding == "fourier" else LearnedPositionalEncoding
        self.pos_enc_v = PE(embed_dim)
        self.pos_enc_s = PE(embed_dim)

        # ── Unimodal towers (independent layers before bottleneck) ────────────
        self.unimodal_v = nn.ModuleList([
            UnimodalTransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_unimodal)
        ])
        self.unimodal_s = nn.ModuleList([
            UnimodalTransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_unimodal)
        ])

        # ── Bottleneck fusion layers ───────────────────────────────────────────
        self.fusion_layers = nn.ModuleList([
            BottleneckTransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers_fusion)
        ])

        # ── Learnable bottleneck tokens ────────────────────────────────────────
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, embed_dim) * 0.02
        )

        # ── Classification head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, num_classes),
        )

        self._init_weights()

    # ── Construction helpers ───────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform init for all linear layers outside timm backbone."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, model_cfg) -> "MultimodalBottleneckTransformer":
        """Instantiate from an OmegaConf DictConfig or plain dict."""
        cfg = dict(model_cfg)
        cfg["sensor_cnn_channels"] = list(cfg.get("sensor_cnn_channels", [64, 128, 256]))
        return cls(**cfg)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        sensor:           Tensor,              # (B, T_S, C_sensor)
        images:           Tensor,              # (B, 3, H, W)
        sensor_positions: Tensor | None = None,  # (B, T_S) normalised ∈ [0,1]
        visual_positions: Tensor | None = None,  # (B, T_V) normalised ∈ [0,1]
    ) -> Tensor:
        """
        Returns
        -------
        logits: ``(B, num_classes)``
        """
        B = sensor.shape[0]

        # ── Encode each modality ───────────────────────────────────────────────
        z_s = self.sensor_encoder(sensor)            # (B, T_S, D)
        z_v = self.visual_encoder(images)            # (B, T_V, D)

        # ── Add positional encodings ───────────────────────────────────────────
        z_s = self.pos_enc_s(z_s, sensor_positions)
        z_v = self.pos_enc_v(z_v, visual_positions)

        # ── Unimodal processing ────────────────────────────────────────────────
        for layer in self.unimodal_v:
            z_v = layer(z_v)
        for layer in self.unimodal_s:
            z_s = layer(z_s)

        # ── Bottleneck fusion ──────────────────────────────────────────────────
        b = self.bottleneck_tokens.expand(B, -1, -1)   # (B, K, D)
        for layer in self.fusion_layers:
            z_v, z_s, b = layer(z_v, z_s, b)

        # ── Classify from mean-pooled bottleneck ───────────────────────────────
        pooled = b.mean(dim=1)        # (B, D)
        return self.head(pooled)      # (B, num_classes)
