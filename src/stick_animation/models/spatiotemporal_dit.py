"""Spatio-temporal Diffusion Transformer for stickman motion.

Design overview (lecture W12 "Animacja postaci – interpolacja",
PoseFormerV2-inspired):

    *  Tokens are *per-joint per-frame*: 15 joints × T frames, so the model
       can reason about each joint individually rather than treating the
       pose as a 45-D blob. Each joint has its own learned embedding.

    *  **Skeleton attention bias** in the spatial blocks: queries gain extra
       logit for keys belonging to nearby joints in the kinematic tree.
       This encodes the structural prior the reference implementation
       lacks.

    *  **Axial attention** alternates spatial blocks (per-frame attention
       over joints) with temporal blocks (per-joint attention over frames).
       Far cheaper than full T·J ⇄ T·J attention while remaining expressive.

    *  **DCT frequency tokens**: low-frequency DCT-II coefficients of each
       joint's trajectory are appended to its temporal stream and attended
       jointly with the time-domain tokens. This injects an explicit
       low-frequency motion prior — the dominant component of walking and
       jumping cycles.

    *  **AdaLN-Zero** (DiT) for time-step and class conditioning, zero-
       initialised so the model starts as identity.

    *  Output is split back into root (xy-velocity, height, yaw sin/cos)
       and 14 local joints — matching ``representation.MotionRep``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..representation import dct_matrix
from ..skeleton import N_JOINTS, Joint, graph_distance

# Feature layout in the input tensor.
ROOT_FEATS = 5            # [xy_vel(2), z(1), yaw(2)]
LOCAL_PER_JOINT = 3
N_LOCAL_JOINTS = N_JOINTS - 1  # 14
FEATURE_DIM = ROOT_FEATS + N_LOCAL_JOINTS * LOCAL_PER_JOINT  # 47


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Zero modulation. x: [..., D], shift/scale: [B, D] broadcast over tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,            # [B*M, N, D]
        shift: torch.Tensor,        # [B*M, D]
        scale: torch.Tensor,
        gate: torch.Tensor,
        bias: torch.Tensor = None,  # [N, N] additive logit bias
    ) -> torch.Tensor:
        h = modulate(self.norm(x), shift, scale)
        qkv = self.qkv(h)
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, Dh]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if bias is not None:
            attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v  # [B, H, N, Dh]
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        return gate.unsqueeze(1) * out


class AdaLNMLP(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x, shift, scale, gate):
        h = modulate(self.norm(x), shift, scale)
        return gate.unsqueeze(1) * self.net(h)


class DiTBlock(nn.Module):
    """One DiT block: AdaLN-Zero attention + AdaLN-Zero MLP.

    `cond` feeds a 6·D vector → (shift1, scale1, gate1, shift2, scale2, gate2).
    """

    def __init__(self, dim: int, n_heads: int, mlp_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = AdaLNAttention(dim, n_heads, dropout)
        self.mlp = AdaLNMLP(dim, mlp_mult, dropout)
        self.cond_proj = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # x: [B', N, D];  cond: [B', D]
        s1, sc1, g1, s2, sc2, g2 = self.cond_proj(cond).chunk(6, dim=-1)
        x = x + self.attn(x, s1, sc1, g1, bias=bias)
        x = x + self.mlp(x, s2, sc2, g2)
        return x


class SpatioTemporalDiT(nn.Module):
    """Per-joint, per-frame DiT with axial attention and DCT temporal tokens."""

    def __init__(
        self,
        n_frames: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,             # number of (spatial+temporal) pairs
        n_dct_tokens: int = 8,         # low-frequency DCT components per joint
        num_classes: int = 2,
        dropout: float = 0.1,
        skeleton_bias_strength: float = 1.5,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.d_model = d_model
        self.n_joints = N_JOINTS
        self.n_dct = n_dct_tokens
        self.null_class_idx = num_classes

        # Per-joint input projections (root: 5-D, others: 3-D).
        self.root_proj = nn.Linear(ROOT_FEATS, d_model)
        self.local_proj = nn.Linear(LOCAL_PER_JOINT, d_model)

        # Per-joint identity embedding.
        self.joint_emb = nn.Parameter(torch.randn(N_JOINTS, d_model) * 0.02)

        # Temporal positional embeddings (separate for time tokens vs DCT tokens).
        self.time_pos = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)
        self.dct_pos = nn.Parameter(torch.randn(n_dct_tokens, d_model) * 0.02)
        # Token-kind embedding so the model can tell time vs frequency.
        self.kind_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)

        # DCT matrix [n_dct, T] — top n_dct rows of orthonormal DCT-II.
        M = dct_matrix(n_frames)
        self.register_buffer("dct_M", torch.tensor(M[:n_dct_tokens]))  # [n_dct, T]

        # Conditioning: time + class.
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        self.class_emb = nn.Embedding(num_classes + 1, d_model)

        # Skeleton attention bias (spatial) — negative graph-distance, permuted
        # to the token-position layout (token 0 = PELVIS, tokens 1..14 = the
        # other joints in enum order). Otherwise the bias would point to the
        # wrong joints because the input layout puts the root first.
        gd_enum = torch.tensor(graph_distance(), dtype=torch.float32)
        token_to_joint = [int(Joint.PELVIS)] + [
            j for j in range(N_JOINTS) if j != int(Joint.PELVIS)
        ]
        perm = torch.tensor(token_to_joint, dtype=torch.long)
        gd_tok = gd_enum[perm][:, perm]
        spatial_bias = -skeleton_bias_strength * gd_tok
        self.register_buffer("spatial_bias", spatial_bias)  # [J, J]

        # Stack of axial blocks.
        self.spatial_blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.temporal_blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        # Final LN + heads.
        self.norm_out = nn.LayerNorm(d_model)
        self.head_root = nn.Linear(d_model, ROOT_FEATS)
        self.head_local = nn.Linear(d_model, LOCAL_PER_JOINT)
        nn.init.zeros_(self.head_root.weight)
        nn.init.zeros_(self.head_root.bias)
        nn.init.zeros_(self.head_local.weight)
        nn.init.zeros_(self.head_local.bias)

    # ------------------------------------------------------------------

    def _split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, F=47] → root[B,T,5], local[B,T,14,3]."""
        root = x[..., :ROOT_FEATS]
        local = x[..., ROOT_FEATS:].reshape(*x.shape[:-1], N_LOCAL_JOINTS, LOCAL_PER_JOINT)
        return root, local

    def _join_output(self, root: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        return torch.cat([root, local.reshape(*root.shape[:-1], -1)], dim=-1)

    def _embed_joints(self, root: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Embed each (frame, joint) into D. Returns [B, T, J, D]."""
        B, T, _ = root.shape
        root_tok = self.root_proj(root).unsqueeze(2)              # [B, T, 1, D]
        local_tok = self.local_proj(local)                        # [B, T, 14, D]
        tokens = torch.cat([root_tok, local_tok], dim=2)          # [B, T, J, D]
        # Add per-joint identity embedding.
        tokens = tokens + self.joint_emb[None, None, :, :]
        return tokens

    def _spatial_pass(self, tokens: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Per-frame attention over joints. tokens: [B, T, J, D]."""
        B, T, J, D = tokens.shape
        x = tokens.reshape(B * T, J, D)
        cond_rep = cond.repeat_interleave(T, dim=0)               # [B*T, D]
        bias = self.spatial_bias[None, None].to(x.dtype)          # [1, 1, J, J]
        x = self.spatial_blocks[layer_idx](x, cond_rep, bias=bias)
        return x.reshape(B, T, J, D)

    def _temporal_pass(self, tokens: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Per-joint attention over (frames + DCT frequency tokens)."""
        B, T, J, D = tokens.shape
        flat = tokens.permute(0, 2, 1, 3).reshape(B * J, T, D)        # [B*J, T, D]
        if self.n_dct > 0:
            dct_tokens = torch.einsum(
                "kt,btd->bkd", self.dct_M.to(flat.dtype), flat
            )                                                          # [B*J, n_dct, D]
            time_tok = flat + self.time_pos[None] + self.kind_emb[0][None, None]
            dct_tok = dct_tokens + self.dct_pos[None] + self.kind_emb[1][None, None]
            combined = torch.cat([time_tok, dct_tok], dim=1)          # [B*J, T+n_dct, D]
        else:
            combined = flat + self.time_pos[None] + self.kind_emb[0][None, None]
        cond_rep = cond.repeat_interleave(J, dim=0)                   # [B*J, D]
        combined = self.temporal_blocks[layer_idx](combined, cond_rep)
        time_out = combined[:, :T, :]                                  # discard freq tokens
        return time_out.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F], t: [B] int, class_label: [B] int. Returns v_pred [B, T, F]."""
        root, local = self._split_input(x)

        # Conditioning.
        t_emb = sinusoidal_embedding(t, self.d_model)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_label)
        cond = t_emb + c_emb                                       # [B, D]

        tokens = self._embed_joints(root, local)                   # [B, T, J, D]

        for i in range(len(self.spatial_blocks)):
            tokens = self._spatial_pass(tokens, cond, i)
            tokens = self._temporal_pass(tokens, cond, i)

        tokens = self.norm_out(tokens)
        root_out = self.head_root(tokens[:, :, 0, :])              # [B, T, 5]
        local_out = self.head_local(tokens[:, :, 1:, :])           # [B, T, 14, 3]
        return self._join_output(root_out, local_out)
