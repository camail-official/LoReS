"""W/A preparation targets the right modules, folds rotations, and reloads weights safely."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from lores.config import PTQConfig
from lores.models.kimi_linear.ptq import build_ptq_spec as _kimi_spec
from lores.models.nemotronh.ptq import build_ptq_spec as _nemotronh_spec
from lores.models.qwen3_5.ptq import build_ptq_spec as _qwen3_5_spec
from lores.rotations.wrappers import SignedInputWrapper
from lores.wa_ptq import quantizers
from lores.wa_ptq.prepare import _prepare_hybrid
from lores.wa_ptq.weights import _weights_only

D_DOWN_A, D_DOWN_B, D_O = 8, 12, 16  # get_hadK: K=1, K=12, K=1 — B forces a per-module K


class _Mixer(nn.Module):
    """Recurrent-ish mixer: ``out_proj`` is the R_o surface."""

    def __init__(self, d_in=D_O, d_out=8):
        super().__init__()
        self.out_proj = nn.Linear(d_in, d_out, bias=False)


class _Mlp(nn.Module):
    """MLP: ``down_proj`` is the R4 surface."""

    def __init__(self, d_in):
        super().__init__()
        self.down_proj = nn.Linear(d_in, 8, bias=False)


class _Decoy(nn.Module):
    """Not a spec target, but carries the same attribute NAMES — a name-substring match would hit these."""

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(D_DOWN_A, 8, bias=False)
        self.out_proj = nn.Linear(D_O, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.mixer = _Mixer()
        self.mlp = _Mlp(d_in)
        self.decoy = _Decoy()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(4, 8)
        self.layers = nn.ModuleList([_Layer(D_DOWN_A), _Layer(D_DOWN_B)])
        self.lm_head = nn.Linear(8, 4, bias=False)


def _spec(model, *, o_projs=None, down_projs=None):
    def _down():
        for li, layer in enumerate(model.layers):
            yield li, layer.mlp, "down_proj"

    def _o():
        for layer in model.layers:
            yield layer.mixer.out_proj

    calls: dict = {}
    spec = types.SimpleNamespace(
        label="Fake",
        hidden=8,
        embed=model.embeddings,
        actquant_root=model,
        fuse_norms=lambda m: calls.setdefault("fuse", []).append(m),
        rotate_residual=lambda m, R1, *, fold_r1: calls.setdefault("rot", []).append((m, R1, fold_r1)),
        down_projs=down_projs or _down,
        o_projs=o_projs or _o,
    )
    spec.calls = calls
    return spec


def _cfg(**rot):
    cfg = PTQConfig()
    cfg.device = "cpu"
    for k, v in rot.items():
        setattr(cfg.rotation, k, v)
    return cfg


def _wrappers(model):
    return quantizers.find_qlayers(model, layers=[quantizers.ActQuantWrapper])


def _unwrap(mod):
    """The ActQuantWrapper behind a possible SignedInputWrapper."""
    return mod.inner if isinstance(mod, SignedInputWrapper) else mod


def test_a_bits_configures_every_wrapper_except_lm_head():
    model = _Model()
    cfg = _cfg(residual=False, r4=False, o=False)
    cfg.quant.a_bits = 4
    _prepare_hybrid(cfg, model, _spec(model))

    qls = _wrappers(model)
    assert qls["lm_head"].quantizer.bits == 16
    assert {n for n, ql in qls.items() if ql.quantizer.bits == 4} == set(qls) - {"lm_head"}


def test_hybrid_weight_quantization_is_rejected_before_mutating_the_model():
    model = _Model()
    cfg = _cfg(residual=False, r4=False, o=False)
    cfg.quant.w_bits = 4
    with pytest.raises(ValueError, match="llm-compressor"):
        _prepare_hybrid(cfg, model, _spec(model))
    assert not _wrappers(model)


def test_r1_fuses_norms_then_rotates_with_a_square_hidden_rotation():
    model = _Model()
    spec = _spec(model)
    _prepare_hybrid(_cfg(residual=True, r4=False, o=False), model, spec)

    assert spec.calls["fuse"] == [model]
    (m, R1, fold_r1) = spec.calls["rot"][0]
    assert m is model and fold_r1 is True and R1.shape == (8, 8) and R1.dtype is torch.float32
    torch.testing.assert_close(R1 @ R1.T, torch.eye(8), rtol=1e-6, atol=1e-6)  # orthogonal


def _lin(i=8, o=8):
    return nn.Linear(i, o, bias=False)


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _nemo_model():
    # layer 0: mamba (out_proj), layer 1: attn (o_proj — must be ignored), layer 2: MLP (down_proj)
    layers = [
        _ns(mixer=_ns(out_proj=_lin(), down_proj=None)),
        _ns(mixer=_ns(o_proj=_lin(), down_proj=None, out_proj=None)),
        _ns(mixer=_ns(down_proj=_lin(), out_proj=None)),
    ]
    return _ns(model=_ns(layers=layers, embeddings=nn.Embedding(4, 8)), config=_ns(hidden_size=8))


def test_nemotronh_spec_iterators_and_scope():
    model = _nemo_model()
    spec = _nemotronh_spec(model)
    # seed index == layer index of the MLP block (2), not the running count (0)
    assert [(si, attr) for si, _p, attr in spec.down_projs()] == [(2, "down_proj")]
    assert list(spec.o_projs()) == [model.model.layers[0].mixer.out_proj]  # mamba only, attn o_proj skipped
    assert spec.actquant_root is model
    assert spec.hidden == 8 and spec.embed is model.model.embeddings


def test_qwen3_5_spec_iterators_and_scope():
    gdn = _ns(layer_type="linear_attention", linear_attn=_ns(out_proj=_lin()), mlp=_ns(down_proj=_lin()))
    attn = _ns(layer_type="full_attention", self_attn=_ns(out_proj=_lin()), mlp=_ns(down_proj=_lin()))
    bb = _ns(layers=[gdn, attn], embed_tokens=nn.Embedding(4, 8))
    tcfg = _ns(hidden_size=8)  # omni config: the spec reads it via get_text_config() (text-only IS its own)
    model = _ns(model=_ns(language_model=bb), config=_ns(text_config=tcfg, get_text_config=lambda: tcfg))
    spec = _qwen3_5_spec(model)
    # every layer's mlp.down_proj, seed index == layer index
    assert [(si, p) for si, p, _a in spec.down_projs()] == [(0, gdn.mlp), (1, attn.mlp)]
    assert list(spec.o_projs()) == [gdn.linear_attn.out_proj]  # GDN only, full_attention skipped
    assert spec.actquant_root is bb


def test_kimi_spec_iterators_and_scope():
    moe = _ns(experts=[_ns(w2=_lin()), _ns(w2=_lin())], shared_experts=_ns(down_proj=_lin()))
    kda = _ns(is_linear_attn=True, self_attn=_ns(o_proj=_lin()), block_sparse_moe=moe)
    mla = _ns(is_linear_attn=False, self_attn=_ns(o_proj=_lin()), mlp=_ns(down_proj=_lin()))
    bb = _ns(layers=[mla, kda], embed_tokens=nn.Embedding(4, 8))
    model = _ns(model=bb, config=_ns(hidden_size=8))
    spec = _kimi_spec(model)
    # RUNNING seed index across dense down_proj, then experts' w2, then shared down_proj — order preserved
    got = [(si, p, a) for si, p, a in spec.down_projs()]
    assert got == [
        (0, mla.mlp, "down_proj"),
        (1, moe.experts[0], "w2"),
        (2, moe.experts[1], "w2"),
        (3, moe.shared_experts, "down_proj"),
    ]
    assert list(spec.o_projs()) == [kda.self_attn.o_proj]  # KDA only, MLA o_proj skipped
    assert spec.actquant_root is bb


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.lin(x)


def _net():
    m = _Net()
    quantizers.add_actquant(m)  # m.lin becomes an ActQuantWrapper
    return m


def test_a16_saved_state_does_not_clobber_a4_quantizer():
    saver = _net()  # an --a-bits 16 run: configure() never called ⇒ maxq stays at its 0 init
    assert int(saver.lin.quantizer.maxq) == 0
    sd = _weights_only(saver.state_dict())
    assert not [k for k in sd if "quantizer." in k], sd.keys()

    loader = _net()
    loader.lin.quantizer.configure(bits=4, groupsize=-1, sym=False, clip_ratio=1.0)
    missing, unexpected = loader.load_state_dict(sd, strict=False)
    assert not unexpected and all("quantizer." in k for k in missing), (missing, unexpected)

    assert int(loader.lin.quantizer.maxq) == 15  # survived the load
    x = torch.randn(1, 4, 8)
    x[0, 1] = 0.0  # a padded row, as produced by batch padding
    assert torch.isfinite(loader(x)).all()
