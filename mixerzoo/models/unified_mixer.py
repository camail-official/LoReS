"""Shared token-mixer scaffold with linear, delta-rule, or softmax transitions."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, RMSNorm, RotaryEmbedding, ShortConvolution
from fla.modules.l2norm import l2norm
from fla.ops.delta_rule import chunk_delta_rule  # delta:  none
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # delta: scalar
from fla.ops.gla import chunk_gla  # linear: diagonal
from fla.ops.kda import chunk_kda  # delta:  diagonal
from fla.ops.simple_gla import chunk_simple_gla  # linear: none (g=None) + scalar

FAMILIES = ("linear", "delta", "softmax")
DECAYS = ("none", "scalar", "diagonal")


class UnifiedMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        mixer_family: str = "delta",  # 'linear' | 'delta' | 'softmax'
        decay: str = "diagonal",  # 'none' | 'scalar' | 'diagonal' (ignored for softmax)
        head_dim: int = 128,
        num_heads: int = 8,
        expand_v: float = 1.0,
        conv_size: int = 4,
        gate_low_rank_dim: int = 128,
        gate_logit_normalizer: int = 16,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        norm_eps: float = 1e-5,
        qk_norm: str = "l2",  # 'l2' (recurrent cells) | 'rms' (learnable-scale QK-norm, Gemma2/OLMo2-style)
        linear_write_gate: bool = False,  # per-token beta write-gate on LINEAR cells (delta always has beta)
        layer_idx: int | None = None,
    ):
        super().__init__()
        assert mixer_family in FAMILIES
        assert decay in DECAYS
        self.mixer_family = mixer_family
        self.decay = decay if mixer_family != "softmax" else "none"
        self.hidden_size = hidden_size
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.num_heads = num_heads
        self.key_dim = head_dim * num_heads
        self.value_dim = self.head_v_dim * num_heads
        self.layer_idx = layer_idx
        self.is_delta = mixer_family == "delta"
        self.is_softmax = mixer_family == "softmax"
        self.gate_logit_normalizer = gate_logit_normalizer
        self.linear_write_gate = bool(linear_write_gate)

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")

        assert qk_norm in ("l2", "rms")
        self.qk_norm = qk_norm
        if qk_norm == "rms":
            self.q_norm = RMSNorm(self.head_k_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_k_dim, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=rope_theta)

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)

        # Linear and delta cells share decay-gate parameterization.
        if self.decay == "scalar":
            self.gk_proj = nn.Linear(hidden_size, num_heads, bias=False)
        elif self.decay == "diagonal":
            self.gk_proj = nn.Sequential(
                nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, self.key_dim, bias=False),
            )
        # write-strength beta: delta always; linear only when linear_write_gate=True ----
        self._use_beta = self.is_delta or (self.mixer_family == "linear" and self.linear_write_gate)
        if self._use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def _decay(self, hidden_states):
        if self.decay == "none":
            return None
        gk = self.gk_proj(hidden_states)
        gk = F.logsigmoid(gk.float()) / self.gate_logit_normalizer  # log-decay <= 0
        if self.decay == "diagonal":
            gk = rearrange(gk, "... (h d) -> ... h d", d=self.head_k_dim)
        return gk

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, **kwargs
    ):
        B, T, _ = hidden_states.shape

        q, _ = self.q_conv1d(self.q_proj(hidden_states), output_final_state=False)
        k, _ = self.k_conv1d(self.k_proj(hidden_states), output_final_state=False)
        v, _ = self.v_conv1d(self.v_proj(hidden_states), output_final_state=False)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # 'l2': exact unit norm (load-bearing for the delta family's contraction).
        # 'rms': learnable-scale RMSNorm (restores softmax sharpness knob; OLMo2/Gemma2-style QK-norm).
        # Plain-l2 softmax is degenerate (cosine logits / sqrt(d) -> near-uniform attention), so the
        # softmax cell (qk_norm='l2') skips the norm entirely; softmax_qknorm uses 'rms'.
        if self.qk_norm == "rms":
            q, k = self.q_norm(q), self.k_norm(k)
        elif not self.is_softmax:
            q, k = l2norm(q), l2norm(k)

        q, k = self.rotary(q, k, seqlen_offset=0)

        if self.is_softmax:
            qh = rearrange(q, "b t h d -> b h t d")
            kh = rearrange(k, "b t h d -> b h t d")
            vh = rearrange(v, "b t h d -> b h t d")
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
            o = rearrange(o, "b h t d -> b t h d")
        else:
            g = self._decay(hidden_states)
            beta = self.b_proj(hidden_states).sigmoid() if self._use_beta else None
            fam, dec = self.mixer_family, self.decay
            if fam == "linear" and self.linear_write_gate:
                v = v * beta.unsqueeze(-1)   # beta-gated write: k*(beta*v)^T, reuse linear kernels
            if fam == "linear" and dec == "none":
                o, _ = chunk_simple_gla(q, k, v, g=None, output_final_state=False)
            elif fam == "linear" and dec == "scalar":
                o, _ = chunk_simple_gla(q, k, v, g=g, output_final_state=False)
            elif fam == "linear" and dec == "diagonal":
                o, _ = chunk_gla(q, k, v, g=g, output_final_state=False)
            elif fam == "delta" and dec == "none":
                o, _ = chunk_delta_rule(q, k, v, beta=beta, output_final_state=False)
            elif fam == "delta" and dec == "scalar":
                o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=False)
            elif fam == "delta" and dec == "diagonal":
                # use_gate_in_kernel=False -> kernel cumsums our log-decay g directly
                o, _ = chunk_kda(q, k, v, g=g, beta=beta, use_gate_in_kernel=False, output_final_state=False)
            else:
                raise ValueError(f"bad cell {fam}/{dec}")

        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, gate)
        o = rearrange(o, "... h d -> ... (h d)")
        return self.o_proj(o), None, past_key_values
