"""Readout precedes state quantization; diagnostic shadow states evolve independently."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

import lores.backends.hf.mixer_zoo.step as mixer_zoo_step
import lores.backends.hf.mixer_zoo.step as mzs
from lores.fake_quant.quantizers import quantize_state
from scripts.calibration import state_geometry
from scripts.diagnostics.state_error_law import shadow_step


class _IdConv(torch.nn.Module):
    """Stand-in for ``ShortConvolution``'s single-token path (fla's is CUDA-only)."""

    kernel_size = (4,)

    def forward(self, x, cache=None, output_final_state=False):
        return x, cache


class _StubMixer(torch.nn.Module):
    """Minimal duck-type of ``UnifiedMixer`` for the delta / no-decay branch of ``_step``.

    Every fla op (``ShortConvolution``, ``l2norm``, ``FusedRMSNormGated``) raises on CPU, so the
    real mixer cannot be stepped here. Only the scaffold is stubbed — ``_step`` itself, including
    the recurrence and the ordering branch under test, is the real code. ``o_norm``/``o_proj`` are
    identities so the returned tensor IS the readout, un-post-processed.
    """

    def __init__(self, H=2, dk=8, dv=8, D=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_heads, self.head_k_dim, self.head_v_dim = H, dk, dv
        self.key_dim, self.value_dim = H * dk, H * dv
        self.mixer_family, self.decay, self.qk_norm = "delta", "none", "l2"
        self.is_delta, self.is_softmax = True, False
        self.linear_write_gate, self._use_beta = False, True
        self.layer_idx = 0
        self.q_proj = torch.nn.Linear(D, self.key_dim, bias=False)
        self.k_proj = torch.nn.Linear(D, self.key_dim, bias=False)
        self.v_proj = torch.nn.Linear(D, self.value_dim, bias=False)
        self.b_proj = torch.nn.Linear(D, H, bias=False)
        self.g_proj = torch.nn.Linear(D, self.value_dim, bias=False)
        self.o_proj = torch.nn.Identity()
        self.q_conv1d = self.k_conv1d = self.v_conv1d = _IdConv()
        self.rotary = lambda q, k, seqlen_offset=0: (q, k)
        self.o_norm = lambda o, gate: o
        self._mz_rk = self._mz_rv = None

    def _decay(self, hidden_states):
        return None


@pytest.fixture
def stub_step(monkeypatch):
    """A stub mixer plus a driver that replays the same inputs under a given ordering."""
    monkeypatch.setattr(mzs, "l2norm", lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    attn = _StubMixer().eval()
    g = torch.Generator().manual_seed(11)
    xs = [torch.randn(1, 1, 16, generator=g) for _ in range(3)]

    def run(quant, stale_flag=None):
        attn._mz_step_state = None
        attn._mz_state_quant = quant
        # A stale attribute must not re-enable the forbidden order.
        attn._mz_quant_after_read = stale_flag
        outs, states = [], []
        with torch.no_grad():
            for x in xs:
                outs.append(mzs._step(attn, x)[0].clone())
                states.append(attn._mz_step_state.S.clone())
        return outs, states

    return run


def _int4(S):
    return quantize_state(S, "int4_block")


def test_stepper_state_quant_actually_perturbs_the_state(stub_step):
    """Guard: the ordering tests below are vacuous unless the quantizer changes the state."""
    _, ref_states = stub_step(None, False)
    assert not torch.equal(_int4(ref_states[0]), ref_states[0])


def test_quant_after_read_first_step_readout_ignores_its_own_rounding(stub_step):
    """With quant-after-read the step-0 readout must be BIT-IDENTICAL to the unquantized run:
    at read time the state has never been rounded.
    """
    ref_out, _ = stub_step(None, False)
    aft_out, _ = stub_step(_int4, True)
    assert torch.equal(aft_out[0], ref_out[0])


def test_quant_after_read_rounding_becomes_visible_exactly_one_step_later(stub_step):
    """Step 0's rounding must show up in step 1's readout — the delay is one step, not none and
    not forever.
    """
    ref_out, _ = stub_step(None, False)
    aft_out, _ = stub_step(_int4, True)
    assert not torch.equal(aft_out[1], ref_out[1])


def test_storage_quantization_is_unconditional_on_legacy_flag(stub_step):
    _, ref_states = stub_step(None, False)
    _, aft_states = stub_step(_int4, True)
    _, bef_states = stub_step(_int4, False)
    assert torch.equal(aft_states[0], _int4(ref_states[0]))
    assert torch.equal(bef_states[0], _int4(ref_states[0]))


def test_chunk_boundary_rounding_only_affects_the_following_step(stub_step):
    from lores.fake_quant.state import quant_every

    ref_out, ref_states = stub_step(None)
    out, states = stub_step(quant_every(_int4, 2))
    assert torch.equal(out[0], ref_out[0])
    assert torch.equal(out[1], ref_out[1])
    assert torch.equal(states[0], ref_states[0])
    assert torch.equal(states[1], _int4(ref_states[1]))
    assert not torch.equal(out[2], ref_out[2])


D_MODEL, H, DK, DV = 16, 2, 8, 8


class _Conv(nn.Module):
    kernel_size = (4,)

    def forward(self, x, cache=None, output_final_state=False):
        return x, None


class _ONorm(nn.Module):
    def forward(self, o, gate):
        return o * torch.sigmoid(gate)


class StubMixer(nn.Module):
    def __init__(self, family: str, decay: str, seed: int, dk: int = DK, dv: int = DV):
        super().__init__()
        torch.manual_seed(seed)
        self.mixer_family, self.decay = family, decay
        self.is_delta = family == "delta"
        self.is_softmax = False
        self.num_heads, self.head_k_dim, self.head_v_dim = H, dk, dv
        self.key_dim, self.value_dim = H * dk, H * dv
        self.linear_write_gate = family == "linear"
        self.qk_norm = "rms"
        self.gate_logit_normalizer = 16
        self.layer_idx = seed
        self.q_proj = nn.Linear(D_MODEL, self.key_dim, bias=False)
        self.k_proj = nn.Linear(D_MODEL, self.key_dim, bias=False)
        self.v_proj = nn.Linear(D_MODEL, self.value_dim, bias=False)
        self.g_proj = nn.Linear(D_MODEL, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, D_MODEL, bias=False)
        self.q_conv1d = self.k_conv1d = self.v_conv1d = _Conv()
        self.q_norm = self.k_norm = nn.Identity()
        self.rotary = lambda q, k, seqlen_offset=0: (q, k)
        self.o_norm = _ONorm()
        if decay == "scalar":
            self.gk_proj = nn.Linear(D_MODEL, H, bias=False)
        elif decay == "diagonal":
            self.gk_proj = nn.Linear(D_MODEL, self.key_dim, bias=False)
        self._use_beta = self.is_delta or self.linear_write_gate
        if self._use_beta:
            self.b_proj = nn.Linear(D_MODEL, H, bias=False)
        self._mz_rk = self._mz_rv = None
        self._mz_state_quant = None
        self._mz_step_state = None
        self._mz_stash_ops = True

    def _decay(self, hidden_states):
        if self.decay == "none":
            return None
        gk = torch.nn.functional.logsigmoid(self.gk_proj(hidden_states).float()) / self.gate_logit_normalizer
        if self.decay == "diagonal":
            gk = gk.reshape(*gk.shape[:-1], H, self.head_k_dim)
        return gk

    def forward(self, x):
        return mixer_zoo_step._step(self, x)


@pytest.mark.parametrize("family", ["linear", "delta"])
@pytest.mark.parametrize("decay", ["none", "scalar", "diagonal"])
def test_shadow_matches_unquantized_recurrence_and_sequence_restart(family, decay):
    m = StubMixer(family, decay, seed=2)
    for seed in (3, 4):
        torch.manual_seed(seed)
        m._mz_step_state = None
        shadow = torch.zeros(1, H, DK, DV)
        for _ in range(6):
            m(torch.randn(1, 1, D_MODEL))
            shadow = shadow_step(m, shadow)
            assert torch.equal(shadow, m._mz_step_state.S)


@pytest.mark.parametrize("family", ["linear", "delta"])
@pytest.mark.parametrize("decay", ["none", "scalar", "diagonal"])
def test_shadow_is_independent_of_corrupted_stored_state(family, decay):
    clean = StubMixer(family, decay, seed=2)
    corrupted = StubMixer(family, decay, seed=2)
    torch.manual_seed(3)
    shadow = torch.zeros(1, H, DK, DV)
    for _ in range(6):
        x = torch.randn(1, 1, D_MODEL)
        clean(x)
        corrupted(x)
        shadow = shadow_step(corrupted, shadow)
        assert torch.equal(shadow, clean._mz_step_state.S)
        corrupted._mz_step_state.S.add_(100)
        assert not torch.equal(shadow, corrupted._mz_step_state.S)


@pytest.mark.parametrize("model", ["qwen3_5", "kimi", "nemotronh"])
def test_hybrid_snapshots_keep_each_batch_items_current_query(tmp_path, monkeypatch, model):
    torch.manual_seed(9)
    state = torch.randn(2, 4, 8, 6) if model == "nemotronh" else torch.randn(2, 4, 6, 8)
    query = torch.randn(2, 2 if model == "nemotronh" else 4, 6)

    def capture(name, factory, **kwargs):
        callback = factory(0, None)
        qslot = inspect.getclosurevars(callback).nonlocals["qslot"]
        for step in range(64):
            qslot[0] = query * (step + 1)
            assert callback(state) is state

    monkeypatch.setattr(state_geometry, "capture_decode", capture)
    state_geometry.hyb_main(model, n_seqs=2, seqlen=64, output=tmp_path, dump_states=True)
    saved = torch.load(tmp_path / f"{model}_states.pt", weights_only=True)["0"]
    expected_state = state if model == "nemotronh" else state.transpose(-1, -2)
    assert torch.equal(saved["S"], expected_state.bfloat16())
    assert torch.equal(saved["q"], (query * 64).bfloat16())
    assert len(saved["S"]) == len(saved["q"]) == 2
