from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402

from weight_skills.backends.mlx_backend import MLXBackend  # noqa: E402
from weight_skills.core.cartridge import Cartridge  # noqa: E402
from weight_skills.errors import GeometryMismatchError  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Linear(dim, dim)


class TinyModel(nn.Module):
    def __init__(self, dim: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.layers = [TinyBlock(dim) for _ in range(layers)]
        self.embed_tokens = nn.Embedding(64, dim)


def forward(model: TinyModel, x: mx.array) -> mx.array:
    for layer in model.layers:
        x = layer.mlp(x)
    return x


def cartridge(skill_id: str, *, dim: int = 32, rank: int = 4, layers=(1, 2), seed: int = 0) -> Cartridge:
    rng = np.random.default_rng(seed)
    deltas = [
        (
            layer,
            (rng.normal(0, 0.02, (rank, dim))).astype(np.float32),
            (rng.normal(0, 0.02, (dim, rank))).astype(np.float32),
        )
        for layer in layers
    ]
    return Cartridge.from_arrays(skill_id=skill_id, layers=deltas, hidden_size=dim, rank=rank)


@pytest.fixture
def model_and_backend():
    mx.random.seed(7)
    model = TinyModel()
    mx.eval(model.parameters())
    return model, MLXBackend(model)


def test_describe_reports_sites(model_and_backend):
    _, backend = model_and_backend
    info = backend.describe()
    assert info["hidden_size"] == 32
    assert info["injectable_layers"] == [0, 1, 2, 3]


def test_mount_changes_output_and_unmount_restores(model_and_backend):
    model, backend = model_and_backend
    x = mx.random.normal((1, 5, 32))
    before = forward(model, x)
    mx.eval(before)

    report = backend.mount([cartridge("a")])
    assert report.layers == (1, 2)
    assert backend.active_skills() == ("a",)
    during = forward(model, x)
    mx.eval(during)
    assert not bool(mx.array_equal(before, during))

    unmount = backend.unmount()
    assert unmount.restored
    after = forward(model, x)
    mx.eval(after)
    assert bool(mx.array_equal(before, after))


def test_hundred_mount_cycles_are_bit_identical(model_and_backend):
    model, backend = model_and_backend
    x = mx.random.normal((1, 4, 32))
    baseline = forward(model, x)
    mx.eval(baseline)
    fingerprint = backend.fingerprint()
    param_ids = {name: id(value) for name, value in tree_flatten(model.parameters())}

    cart = cartridge("cycle")
    for _ in range(100):
        backend.mount([cart])
        _ = forward(model, x)
        report = backend.unmount()
        assert report.restored

    final = forward(model, x)
    mx.eval(final)
    assert bool(mx.array_equal(baseline, final))
    assert backend.fingerprint() == fingerprint
    assert {name: id(value) for name, value in tree_flatten(model.parameters())} == param_ids


def test_fused_two_cartridges_equals_single_concatenated_cartridge(model_and_backend):
    """Batched push-through must equal one cartridge holding the concatenated factors."""
    model, backend = model_and_backend
    x = mx.random.normal((1, 3, 32))

    c1, c2 = cartridge("one", layers=(3,), seed=1), cartridge("two", layers=(3,), seed=2)
    backend.mount([c1, c2])
    fused = forward(model, x)
    mx.eval(fused)
    backend.unmount()

    a = np.concatenate([c1.layers[0].a, c2.layers[0].a], axis=0)
    b = np.concatenate([c1.layers[0].b, c2.layers[0].b], axis=1)
    combined = Cartridge.from_arrays(skill_id="combined", layers=[(3, a, b)], hidden_size=32, rank=8)
    backend.mount([combined])
    single = forward(model, x)
    mx.eval(single)
    backend.unmount()

    np.testing.assert_allclose(np.array(fused), np.array(single), rtol=1e-5, atol=1e-6)


def test_geometry_mismatch_rejected(model_and_backend):
    _, backend = model_and_backend
    bad = cartridge("bad", dim=16, rank=4)
    with pytest.raises(GeometryMismatchError):
        backend.mount([bad])


def test_non_injectable_layer_rejected(model_and_backend):
    _, backend = model_and_backend
    bad = cartridge("bad", layers=(99,))
    with pytest.raises(GeometryMismatchError):
        backend.mount([bad])


def test_remount_replaces_previous(model_and_backend):
    model, backend = model_and_backend
    backend.mount([cartridge("one", seed=1)])
    backend.mount([cartridge("two", seed=2)])
    assert backend.active_skills() == ("two",)
    assert backend.unmount().restored


def test_push_through_registers_no_parameters(model_and_backend):
    model, backend = model_and_backend
    before = len(tree_flatten(model.parameters()))
    backend.mount([cartridge("a")])
    during = len(tree_flatten(model.parameters()))
    backend.unmount()
    after = len(tree_flatten(model.parameters()))
    assert before == during == after


def test_unmount_without_mount_is_noop(model_and_backend):
    _, backend = model_and_backend
    report = backend.unmount()
    assert report.restored
    assert backend.active_skills() == ()
