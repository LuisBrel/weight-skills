from __future__ import annotations

import numpy as np
import pytest

from weight_skills.core.cartridge import Cartridge, LayerDelta
from weight_skills.errors import CartridgeFormatError


def test_roundtrip_preserves_factors_and_metadata(tmp_path, sample_cartridge):
    path = sample_cartridge.save(tmp_path / "skill.safetensors")
    loaded = Cartridge.load(path)

    assert loaded.skill_id == sample_cartridge.skill_id
    assert loaded.rank == sample_cartridge.rank
    assert loaded.hidden_size == sample_cartridge.hidden_size
    assert loaded.layer_indices == sample_cartridge.layer_indices
    assert loaded.digest == sample_cartridge.digest
    for original, restored in zip(sample_cartridge.layers, loaded.layers, strict=True):
        np.testing.assert_array_equal(original.a, restored.a)
        np.testing.assert_array_equal(original.b, restored.b)


def test_save_writes_provenance_metadata(tmp_path, cartridge_factory):
    cart = cartridge_factory("aws-cso", hidden=64, rank=8, layers=(3,))
    path = cart.save(tmp_path / "aws.safetensors")
    loaded = Cartridge.load(path)
    meta = loaded.metadata()
    assert meta["schema"] == "weight-skills/cartridge@1"
    assert meta["hidden_size"] == 64
    assert meta["rank"] == 8
    assert meta["layers"] == [3]


def test_shape_mismatch_is_rejected(tmp_path, cartridge_factory):
    cart = cartridge_factory(hidden=32, rank=4)
    path = cart.save(tmp_path / "bad.safetensors")
    # Corrupt metadata: declare a different hidden size than the tensors carry.
    import json

    from safetensors import safe_open
    from safetensors.numpy import save_file

    with safe_open(str(path), framework="numpy") as handle:
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}  # noqa: SIM118
        meta = json.loads(handle.metadata()["weight_skills"])
    meta["hidden_size"] = 99
    save_file(tensors, str(path), metadata={"weight_skills": json.dumps(meta)})

    with pytest.raises(CartridgeFormatError):
        Cartridge.load(path)


def test_non_finite_factors_are_rejected():
    a = np.zeros((4, 32), dtype=np.float32)
    a[0, 0] = np.nan
    b = np.zeros((32, 4), dtype=np.float32)
    with pytest.raises(CartridgeFormatError):
        Cartridge.from_arrays(
            skill_id="nan", layers=[(0, a, b)], hidden_size=32, rank=4
        )


def test_missing_file_raises(tmp_path):
    with pytest.raises(CartridgeFormatError):
        Cartridge.load(tmp_path / "nope.safetensors")


def test_with_scale_scales_b_only(cartridge_factory):
    cart = cartridge_factory()
    scaled = cart.with_scale(0.5)
    assert scaled.digest != cart.digest
    for original, updated in zip(cart.layers, scaled.layers, strict=True):
        np.testing.assert_array_equal(original.a, updated.a)
        np.testing.assert_allclose(updated.b, original.b * 0.5, rtol=1e-6)


def test_frobenius_and_nbytes(cartridge_factory):
    cart = cartridge_factory(hidden=32, rank=4, layers=(1,))
    frob = cart.frobenius()
    assert set(frob) == {1}
    assert frob[1] > 0
    assert cart.nbytes == 4 * 32 * 4 * 2  # rank*hidden*float32 * two factors


def test_duplicate_layers_rejected():
    a = np.zeros((2, 8), dtype=np.float32)
    b = np.zeros((8, 2), dtype=np.float32)
    cart = Cartridge(
        skill_id="dup",
        version="0.0.1",
        hidden_size=8,
        rank=2,
        layers=(LayerDelta(1, a, b), LayerDelta(1, a, b)),
    )
    with pytest.raises(CartridgeFormatError):
        cart.validate()


def test_schema_mismatch_rejected(tmp_path, cartridge_factory):
    import json

    from safetensors import safe_open
    from safetensors.numpy import save_file

    cart = cartridge_factory()
    path = cart.save(tmp_path / "x.safetensors")
    with safe_open(str(path), framework="numpy") as handle:
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}  # noqa: SIM118
    save_file(tensors, str(path), metadata={"weight_skills": json.dumps({"schema": "other/format@9"})})
    with pytest.raises(CartridgeFormatError):
        Cartridge.load(path)
