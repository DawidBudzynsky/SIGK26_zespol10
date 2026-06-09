from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..representation import dct_matrix
from ..skeleton import N_JOINTS, Joint, graph_distance

JOINT_DIM = 3
FEATURE_DIM = N_JOINTS * JOINT_DIM


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, x, shift, scale, gate, bias=None):
        h = modulate(self.norm(x), shift, scale)
        B, N, _ = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if bias is not None:
            attn = attn + bias
        attn = F.dropout(attn.softmax(dim=-1), p=self.dropout, training=self.training)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return gate.unsqueeze(1) * self.proj(out)


class AdaLNMLP(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, shift, scale, gate):
        return gate.unsqueeze(1) * self.net(modulate(self.norm(x), shift, scale))


class DiTBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, mlp_mult: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()
        self.attn = AdaLNAttention(dim, n_heads, dropout)
        self.mlp = AdaLNMLP(dim, mlp_mult, dropout)
        self.cond_proj = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond, bias=None):
        s1, sc1, g1, s2, sc2, g2 = self.cond_proj(cond).chunk(6, dim=-1)
        x = x + self.attn(x, s1, sc1, g1, bias=bias)
        x = x + self.mlp(x, s2, sc2, g2)
        return x


class SpatioTemporalDiT(nn.Module):
    def __init__(
        self,
        n_frames: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_dct_tokens: int = 8,
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

        self.joint_proj = nn.Linear(JOINT_DIM, d_model)

        self.joint_emb = nn.Parameter(torch.randn(N_JOINTS, d_model) * 0.02)

        self.time_pos = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)
        self.dct_pos = nn.Parameter(torch.randn(n_dct_tokens, d_model) * 0.02)
        self.kind_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)

        M = dct_matrix(n_frames)
        self.register_buffer("dct_M", torch.tensor(M[:n_dct_tokens]))

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        self.class_emb = nn.Embedding(num_classes + 1, d_model)

        gd = torch.tensor(graph_distance(), dtype=torch.float32)
        self.register_buffer("spatial_bias", -skeleton_bias_strength * gd)

        self.spatial_blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.temporal_blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, JOINT_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _spatial_pass(self, tokens, cond, i):
        """Per-frame attention over 15 joints. tokens: [B, T, J, D]"""
        B, T, J, D = tokens.shape
        x = tokens.reshape(B * T, J, D)
        cond_rep = cond.repeat_interleave(T, dim=0)
        bias = self.spatial_bias[None, None].to(x.dtype)
        x = self.spatial_blocks[i](x, cond_rep, bias=bias)
        return x.reshape(B, T, J, D)

    def _temporal_pass(self, tokens, cond, i):
        """Per-joint attention over frames + DCT tokens. tokens: [B, T, J, D]"""
        B, T, J, D = tokens.shape
        flat = tokens.permute(0, 2, 1, 3).reshape(B * J, T, D)
        time_tok = flat + self.time_pos[None] + self.kind_emb[0][None, None]
        if self.n_dct > 0:
            dct_tok = (
                torch.einsum("kt,btd->bkd", self.dct_M.to(flat.dtype), flat)
                + self.dct_pos[None]
                + self.kind_emb[1][None, None]
            )
            combined = torch.cat([time_tok, dct_tok], dim=1)
        else:
            combined = time_tok
        cond_rep = cond.repeat_interleave(J, dim=0)
        out = self.temporal_blocks[i](combined, cond_rep)
        return out[:, :T].reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor
    ) -> torch.Tensor:
        """x: [B, T, 45], t: [B] int, class_label: [B] int → pred [B, T, 45]"""
        B, T, _ = x.shape
        joints = x.reshape(B, T, N_JOINTS, JOINT_DIM)
        tokens = self.joint_proj(joints)
        tokens = tokens + self.joint_emb[None, None]

        t_emb = self.time_mlp(sinusoidal_embedding(t, self.d_model))
        c_emb = self.class_emb(class_label)
        cond = t_emb + c_emb

        for i in range(len(self.spatial_blocks)):
            tokens = self._spatial_pass(tokens, cond, i)
            tokens = self._temporal_pass(tokens, cond, i)

        out = self.head(self.norm_out(tokens))
        return out.reshape(B, T, FEATURE_DIM)
