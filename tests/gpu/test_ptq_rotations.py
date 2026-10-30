"""W/A rotation folding and module targeting on CUDA."""

import pytest
import torch

from lores.rotations.wrappers import SignedInputWrapper
from lores.wa_ptq.prepare import _prepare_hybrid
from tests.cpu.test_ptq import D_DOWN_A, D_DOWN_B, D_O, _cfg, _Model, _spec, _unwrap, _wrappers

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytestmark = cuda_only


def test_online_had_targets_spec_modules_by_identity_not_name():
    """R4/R_o online Hadamards land on the spec's projections only — the same-named decoys stay untouched."""
    model = _Model()
    targets = {id(m) for layer in model.layers for m in (layer.mlp.down_proj, layer.mixer.out_proj)}
    decoys = {id(m) for layer in model.layers for m in (layer.decoy.down_proj, layer.decoy.out_proj)}

    _prepare_hybrid(_cfg(residual=False, r4=True, o=True), model, _spec(model))

    on = {id(ql.module) for ql in _wrappers(model).values() if ql.online_full_had}
    assert on == targets, "online Hadamard set must equal the spec's identity set (decoys must not match by name)"
    assert not (on & decoys)


def test_online_had_K_is_per_module_not_shared():
    """Each wrapper's ``K``/``had_K`` comes from its OWN in_features, not one shared K."""
    model = _Model()
    _prepare_hybrid(_cfg(residual=False, r4=True, o=True), model, _spec(model))

    by_in = {ql.module.in_features: ql for ql in _wrappers(model).values() if ql.online_full_had}
    assert by_in[D_DOWN_A].K == 1 and by_in[D_DOWN_A].had_K is None  # 8 = 2³ ⇒ K=1, no explicit H_K
    assert by_in[D_DOWN_B].K == 12 and by_in[D_DOWN_B].had_K.shape == (12, 12)  # 12 = 12·2⁰ ⇒ K=12
    assert by_in[D_O].K == 1
    assert all(ql.fp32_had for ql in by_in.values())


def _expected_sign(si, in_features, dtype):
    g = torch.Generator(device="cpu").manual_seed(1000 + si)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).to(dtype)


def test_r4_signs_uses_the_seed_index_the_spec_yields():
    """The sign generator is seeded by ``1000 + si`` with ``si`` from the spec — NOT by a running counter."""
    model = _Model()
    # spec yields non-consecutive, non-zero-based seed indices: a running counter would give 0,1
    seeds = [7, 3]
    spec = _spec(model, down_projs=lambda: ((seeds[i], layer.mlp, "down_proj") for i, layer in enumerate(model.layers)))
    w0 = model.layers[0].mlp.down_proj.weight.data.clone()

    _prepare_hybrid(_cfg(residual=False, r4=True, r4_signs=True, o=False), model, spec)

    dp0 = model.layers[0].mlp.down_proj
    assert isinstance(dp0, SignedInputWrapper)
    torch.testing.assert_close(dp0.sign, _expected_sign(7, D_DOWN_A, w0.dtype))
    torch.testing.assert_close(model.layers[1].mlp.down_proj.sign, _expected_sign(3, D_DOWN_B, w0.dtype))


def test_r4_signs_folds_D_before_the_hadamard():
    """``W ← (W⊙D)`` runs BEFORE the R4 Hadamard fold — pinned bit-exactly by running the body both ways.

    No tolerance: both arms push the same fp32 CUDA fold, so ``r4_signs=True`` on ``W`` must be byte-identical to
    ``r4_signs=False`` on a pre-signed ``W``. Swapping the two folds (H then D) changes the result.
    """
    model, ref_model = _Model(), _Model()
    ref_model.load_state_dict(model.state_dict())
    dtype = ref_model.layers[0].mlp.down_proj.weight.dtype
    D = _expected_sign(0, D_DOWN_A, dtype)
    w = ref_model.layers[0].mlp.down_proj.weight
    w.data = (w.data.double() * D.double()).to(dtype)

    _prepare_hybrid(_cfg(residual=False, r4=True, r4_signs=True, o=False), model, _spec(model))
    _prepare_hybrid(_cfg(residual=False, r4=True, r4_signs=False, o=False), ref_model, _spec(ref_model))

    got = _unwrap(model.layers[0].mlp.down_proj).module.weight
    assert torch.equal(got, ref_model.layers[0].mlp.down_proj.module.weight)
