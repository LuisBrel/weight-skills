"""Tests for the compiled prompt-cache library.

No model is loaded and no GPU is used: caches are fake objects that expose
``.nbytes`` (or a test-provided estimator), and the default persistence
serializer is exercised against a fake ``mlx_lm.models.cache`` module.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from weight_skills.errors import WeightSkillsError
from weight_skills.serving import (
    PROMPT_KEY_HEADER,
    CacheSizeError,
    CompiledPromptLibrary,
    MLXPromptCacheSerializer,
    estimate_cache_nbytes,
    key_from_messages,
    normalise_metadata,
    prompt_key,
    resolve_prompt_key,
)


class FakeCache:
    """Stand-in for a prompt cache: knows its byte size and deepcopies cleanly."""

    def __init__(self, nbytes: int, data: str = "") -> None:
        self.nbytes = nbytes
        self.data = data

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeCache) and (self.nbytes, self.data) == (other.nbytes, other.data)

    def __repr__(self) -> str:
        return f"FakeCache(nbytes={self.nbytes}, data={self.data!r})"


class SizedCache:
    """Fake cache without ``.nbytes``, sized by the estimator hook."""

    def __init__(self, size: int) -> None:
        self.size = size


class JSONFileSerializer:
    """Test double for ``CacheSerializer``: JSON artifact instead of safetensors."""

    def __init__(self) -> None:
        self.saves = 0
        self.loads = 0

    def save(self, path: Path, cache: Any, metadata: dict[str, str]) -> None:
        assert isinstance(cache, FakeCache)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nbytes": cache.nbytes, "data": cache.data, "metadata": metadata}))
        self.saves += 1

    def load(self, path: Path) -> tuple[Any, dict[str, str]]:
        payload = json.loads(path.read_text())
        self.loads += 1
        return FakeCache(int(payload["nbytes"]), str(payload["data"])), dict(payload["metadata"])


def make_library(**kwargs: Any) -> CompiledPromptLibrary:
    kwargs.setdefault("max_bytes", 1000)
    # tests that store caches without a prefix checksum opt out explicitly; the library default is
    # now require_provenance=True (D12/G.69), tested separately below.
    kwargs.setdefault("require_provenance", False)
    return CompiledPromptLibrary(**kwargs)


# ------------------------------------------------------------------ store / get
def test_store_get_returns_fresh_deepcopy():
    library = make_library()
    cache = FakeCache(100, "persona")
    library.store("persona", cache, n_tokens=42)

    first = library.get("persona")
    second = library.get("persona")

    assert isinstance(first, FakeCache)
    assert first == cache
    assert first is not cache
    assert first is not second
    first.data = "mutated by generation"
    assert library.get("persona").data == "persona"
    assert len(library) == 1
    assert "persona" in library


def test_get_miss_returns_none():
    library = make_library()
    assert library.get("missing") is None
    stats = library.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["requests"] == 1
    assert stats["hit_rate"] == 0.0


def test_hit_and_miss_counters():
    library = make_library()
    library.store("persona", FakeCache(100), n_tokens=10)
    library.get("persona")
    library.get("persona")
    library.get("missing")

    stats = library.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["requests"] == 3
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    assert library.hits == 2
    assert library.misses == 1


def test_list_cache_is_summed_and_forked():
    library = make_library()
    layers = [FakeCache(100, "keys"), FakeCache(150, "values")]
    library.store("multi", layers, n_tokens=12)

    assert library.stats()["bytes"] == 250
    forked = library.get("multi")
    assert isinstance(forked, list)
    assert forked == layers
    assert forked is not layers
    assert forked[0] is not layers[0]


# ------------------------------------------------------------------ LRU / budget
def test_lru_eviction_under_budget():
    library = make_library(max_bytes=300)
    library.store("a", FakeCache(100), n_tokens=10)
    library.store("b", FakeCache(100), n_tokens=10)
    library.store("c", FakeCache(100), n_tokens=10)
    assert library.stats()["evictions"] == 0
    assert library.names() == ["a", "b", "c"]

    library.get("a")  # refresh: b becomes the least-recently-used entry
    library.store("d", FakeCache(100), n_tokens=10)

    stats = library.stats()
    assert library.names() == ["c", "a", "d"]
    assert stats["entries"] == 3
    assert stats["bytes"] == 300
    assert stats["evictions"] == 1
    assert stats["peak_bytes"] == 300
    assert library.get("b") is None
    assert library.get("d").nbytes == 100


def test_lru_order_is_oldest_first_in_stats():
    library = make_library()
    library.store("oldest", FakeCache(10), n_tokens=1)
    library.store("middle", FakeCache(10), n_tokens=1)
    library.store("newest", FakeCache(10), n_tokens=1)
    assert library.stats()["names"] == ["oldest", "middle", "newest"]


def test_store_replacement_updates_bytes_without_eviction():
    library = make_library(max_bytes=300)
    library.store("a", FakeCache(100, "v1"), n_tokens=10)
    library.store("a", FakeCache(250, "v2"), n_tokens=20)

    stats = library.stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == 250
    assert stats["overwrites"] == 1
    assert stats["evictions"] == 0
    assert stats["tokens"] == 20
    assert library.get("a").data == "v2"


def test_oversized_entry_is_rejected_and_state_is_unchanged():
    library = make_library(max_bytes=100)
    library.store("fits", FakeCache(100), n_tokens=5)

    with pytest.raises(CacheSizeError) as excinfo:
        library.store("too-big", FakeCache(101), n_tokens=1)
    assert isinstance(excinfo.value, WeightSkillsError)

    stats = library.stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == 100
    assert stats["evictions"] == 0
    assert library.names() == ["fits"]


def test_store_evicts_multiple_entries_to_make_room():
    library = make_library(max_bytes=250)
    library.store("a", FakeCache(100), n_tokens=1)
    library.store("b", FakeCache(100), n_tokens=1)
    library.store("c", FakeCache(100), n_tokens=1)
    assert library.names() == ["b", "c"]

    library.store("d", FakeCache(100), n_tokens=1)
    assert library.names() == ["c", "d"]
    assert library.stats()["evictions"] == 2
    assert library.stats()["bytes"] == 200


def test_zero_budget_allows_only_zero_byte_entries():
    library = make_library(max_bytes=0)
    with pytest.raises(CacheSizeError):
        library.store("any", FakeCache(1), n_tokens=1)
    library.store("empty", FakeCache(0), n_tokens=0)
    assert library.get("empty").nbytes == 0
    assert library.stats()["utilization"] == 0.0


# -------------------------------------------------------------------- size hooks
def test_estimator_hook_is_used():
    seen: list[Any] = []

    def estimator(cache: Any) -> int:
        seen.append(cache)
        return cache.size + 8

    library = make_library(estimator=estimator)
    cache = SizedCache(50)
    library.store("sized", cache, n_tokens=5)

    assert seen == [cache]
    assert library.stats()["bytes"] == 58


def test_explicit_nbytes_overrides_estimation():
    library = make_library()
    library.store("opaque", object(), n_tokens=5, nbytes=77)
    assert library.stats()["bytes"] == 77


def test_unestimable_cache_raises():
    library = make_library()
    with pytest.raises(CacheSizeError):
        library.store("opaque", object(), n_tokens=5)
    assert len(library) == 0
    assert library.stats()["stores"] == 0


def test_estimator_returning_garbage_raises():
    library = make_library(estimator=lambda cache: None)  # type: ignore[arg-type,return-value]
    with pytest.raises(CacheSizeError):
        library.store("x", FakeCache(10), n_tokens=1)


def test_estimator_failure_raises_cache_size_error():
    def estimator(cache: Any) -> int:
        raise RuntimeError("boom")

    library = make_library(estimator=estimator)
    with pytest.raises(CacheSizeError):
        library.store("x", FakeCache(10), n_tokens=1)


def test_estimate_cache_nbytes_understands_cache_shapes():
    class Layer:
        pass

    layer = Layer()
    layer.state = {"k": np.zeros(4, dtype=np.float32), "v": np.zeros(4, dtype=np.float32)}
    assert estimate_cache_nbytes(layer) == 32
    assert estimate_cache_nbytes([layer, layer]) == 64
    assert estimate_cache_nbytes(FakeCache(123)) == 123
    assert estimate_cache_nbytes([FakeCache(10), None, FakeCache(5)]) == 15
    assert estimate_cache_nbytes([]) == 0
    assert estimate_cache_nbytes(object()) is None
    assert estimate_cache_nbytes(None) is None


def test_estimate_cache_nbytes_accepts_nbytes_method():
    class MethodCache:
        def nbytes(self) -> int:
            return 7

    assert estimate_cache_nbytes(MethodCache()) == 7


# ----------------------------------------------------------------- prompt routing
def test_prompt_key_is_sha256_of_system_text():
    text = "You are CLU, a rigorous clinical reviewer."
    assert prompt_key(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert prompt_key("") == hashlib.sha256(b"").hexdigest()
    assert prompt_key("a") != prompt_key("b")


def test_key_from_messages_hashes_first_system_message():
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Hi"}]
    assert key_from_messages(messages) == prompt_key("Rules.")
    assert key_from_messages([{"role": "user", "content": "Hi"}]) is None
    assert key_from_messages([{"role": "system", "content": ""}]) is None
    assert key_from_messages([]) is None


def test_store_system_and_get_system():
    library = make_library()
    text = "You are a careful reviewer."
    key = library.store_system(text, FakeCache(64, "compiled"), n_tokens=17)

    assert key == prompt_key(text)
    assert library.names() == [key]
    assert library.get_system(text).data == "compiled"
    assert library.get_system("a different system prompt") is None
    assert library.stats()["tokens"] == 17


def test_resolve_prompt_key_priority():
    text = "You are a careful reviewer."
    messages = [{"role": "system", "content": text}]
    hashed = prompt_key(text)

    assert resolve_prompt_key(messages=messages) == hashed
    assert resolve_prompt_key(headers={PROMPT_KEY_HEADER: "explicit"}, messages=messages) == "explicit"
    assert resolve_prompt_key(headers={PROMPT_KEY_HEADER: f"sha256:{hashed}"}) == hashed
    assert resolve_prompt_key(headers={PROMPT_KEY_HEADER.lower(): f"  SHA256:{hashed}  "}) == hashed
    assert resolve_prompt_key(headers={PROMPT_KEY_HEADER: "  "}, messages=messages) == hashed
    assert resolve_prompt_key(messages=[{"role": "user", "content": "Hi"}]) is None
    assert resolve_prompt_key() is None


# --------------------------------------------------------------------- lifecycle
def test_discard_and_clear():
    library = make_library()
    library.store("a", FakeCache(100), n_tokens=1)
    library.store("b", FakeCache(50), n_tokens=2)

    assert library.discard("a") is True
    assert library.discard("a") is False
    stats = library.stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == 50
    assert stats["discards"] == 1

    assert library.clear() == 1
    assert len(library) == 0
    assert library.stats()["bytes"] == 0
    assert library.stats()["tokens"] == 0


def test_stats_has_stable_keys():
    library = make_library()
    library.store("a", FakeCache(3), n_tokens=1)
    stats = library.stats()
    required = {
        "entries",
        "names",
        "tokens",
        "bytes",
        "budget_bytes",
        "utilization",
        "peak_bytes",
        "hits",
        "misses",
        "requests",
        "hit_rate",
        "evictions",
        "stores",
        "overwrites",
        "discards",
        "disk_saves",
        "disk_loads",
    }
    assert required <= set(stats)
    assert stats["budget_bytes"] == 1000
    assert stats["utilization"] == pytest.approx(3 / 1000)
    assert isinstance(stats["names"], list)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        CompiledPromptLibrary(max_bytes=-1)
    with pytest.raises(ValueError):
        CompiledPromptLibrary(max_bytes=1.5)  # type: ignore[arg-type]

    library = make_library()
    with pytest.raises(ValueError):
        library.store("", FakeCache(1), n_tokens=1)
    with pytest.raises(ValueError):
        library.store("a", None, n_tokens=1)
    with pytest.raises(ValueError):
        library.store("a", FakeCache(1), n_tokens=-1)
    with pytest.raises(CacheSizeError):
        library.store("a", FakeCache(1), n_tokens=1, nbytes=-5)


def test_concurrent_gets_are_consistent():
    library = make_library()
    library.store("persona", FakeCache(100), n_tokens=10)
    results: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(10):
                results.append(library.get("persona"))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 80
    assert all(isinstance(result, FakeCache) for result in results)
    assert library.stats()["hits"] == 80
    assert library.stats()["entries"] == 1


# ------------------------------------------------------------------ persistence
def test_save_load_roundtrip(tmp_path):
    serializer = JSONFileSerializer()
    library = make_library(max_bytes=10_000, serializer=serializer, persist_dir=tmp_path / "compiled")
    library.store("persona", FakeCache(4096, "demo"), n_tokens=123)

    path = library.save("persona")
    assert path == tmp_path / "compiled" / "persona.safetensors"
    assert path.is_file()
    assert library.stats()["disk_saves"] == 1

    payload = json.loads(path.read_text())["metadata"]
    assert payload["weight_skills_n_tokens"] == "123"
    assert payload["weight_skills_cache_key"] == "persona"
    assert payload["weight_skills_schema"] == "weight-skills/compiled-prompt@1"

    library.clear()
    assert library.load(path) == "persona"
    assert library.names() == ["persona"]
    assert library.get("persona").data == "demo"
    stats = library.stats()
    assert stats["disk_loads"] == 1
    assert stats["tokens"] == 123
    assert stats["bytes"] == 4096


def test_save_requires_path_or_persist_dir():
    library = make_library(serializer=JSONFileSerializer())
    library.store("a", FakeCache(10), n_tokens=1)
    with pytest.raises(ValueError):
        library.save("a")


def test_save_unknown_name_raises_keyerror(tmp_path):
    library = make_library(serializer=JSONFileSerializer(), persist_dir=tmp_path)
    with pytest.raises(KeyError):
        library.save("missing")


def test_save_sanitizes_names_for_filenames(tmp_path):
    library = make_library(serializer=JSONFileSerializer(), persist_dir=tmp_path)
    library.store("tenant/a persona", FakeCache(10), n_tokens=1)
    path = library.save("tenant/a persona")
    assert "/" not in path.name
    assert path.is_file()

    long_name = "x" * 300
    library.store(long_name, FakeCache(10), n_tokens=1)
    long_path = library.save(long_name)
    assert len(long_path.name) <= 120
    assert long_path.is_file()


def test_load_missing_file_raises(tmp_path):
    library = make_library(serializer=JSONFileSerializer())
    with pytest.raises(FileNotFoundError):
        library.load(tmp_path / "nope.safetensors")


def test_load_requires_n_tokens_when_metadata_lacks_it(tmp_path):
    path = tmp_path / "foreign.safetensors"
    path.write_text(json.dumps({"nbytes": 10, "data": "x", "metadata": {}}))
    library = make_library(serializer=JSONFileSerializer())
    with pytest.raises(ValueError):
        library.load(path)
    assert len(library) == 0


def test_load_explicit_name_and_tokens_override_metadata(tmp_path):
    serializer = JSONFileSerializer()
    library = make_library(serializer=serializer, persist_dir=tmp_path)
    library.store("original", FakeCache(10, "x"), n_tokens=99)
    path = library.save("original")

    other = make_library(serializer=serializer)
    assert other.load(path, name="alias", n_tokens=7) == "alias"
    assert other.stats()["tokens"] == 7
    assert other.get("alias").data == "x"


def test_load_rejects_foreign_schema(tmp_path):
    path = tmp_path / "foreign.safetensors"
    path.write_text(
        json.dumps(
            {
                "nbytes": 10,
                "data": "x",
                "metadata": {"weight_skills_schema": "other/format@9", "weight_skills_n_tokens": "1"},
            }
        )
    )
    library = make_library(serializer=JSONFileSerializer())
    with pytest.raises(ValueError):
        library.load(path)


def test_save_all_and_load_dir(tmp_path):
    serializer = JSONFileSerializer()
    library = make_library(serializer=serializer, persist_dir=tmp_path)
    library.store("a", FakeCache(10, "alpha"), n_tokens=1)
    library.store("b", FakeCache(20, "beta"), n_tokens=2)

    paths = library.save_all()
    assert len(paths) == 2
    assert all(path.is_file() for path in paths)

    restored = make_library(serializer=serializer)
    names = restored.load_dir(tmp_path)
    assert sorted(names) == ["a", "b"]
    assert restored.get("a").data == "alpha"
    assert restored.get("b").data == "beta"
    assert restored.stats()["disk_loads"] == 2


def test_save_all_and_load_dir_require_directory():
    library = make_library(serializer=JSONFileSerializer())
    library.store("a", FakeCache(10), n_tokens=1)
    with pytest.raises(ValueError):
        library.save_all()

    with pytest.raises(FileNotFoundError):
        library.load_dir("/definitely/not/a/directory")


# ------------------------------------------------ default serializer integration
def _install_fake_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``mlx_lm.models.cache`` so persistence wiring is testable on CPU."""
    state: dict[str, Any] = {}

    cache_module = ModuleType("mlx_lm.models.cache")

    def save_prompt_cache(file_name: str, cache: Any, metadata: dict[str, str] | None = None) -> None:
        state["saved"] = (file_name, list(cache), dict(metadata or {}))
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        Path(file_name).write_text("fake-safetensors")

    def load_prompt_cache(file_name: str, return_metadata: bool = False) -> Any:
        state.setdefault("loads", []).append((file_name, return_metadata))
        cache = state.get("cache", [FakeCache(11, "loaded")])
        metadata = state.get("metadata", {})
        return cache, dict(metadata)

    cache_module.save_prompt_cache = save_prompt_cache  # type: ignore[attr-defined]
    cache_module.load_prompt_cache = load_prompt_cache  # type: ignore[attr-defined]

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.__path__ = []  # type: ignore[attr-defined]
    models = ModuleType("mlx_lm.models")
    models.__path__ = []  # type: ignore[attr-defined]
    mlx_lm.models = models  # type: ignore[attr-defined]
    models.cache = cache_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)
    return state


def test_serializer_load_normalises_legacy_nested_metadata(monkeypatch, tmp_path):
    # mlx_lm's tree_unflatten turns old dotted keys into nested dicts: {"weight-skills": {...}}
    state = _install_fake_mlx_lm(monkeypatch)
    state["metadata"] = {"weight-skills": {"schema": "weight-skills/compiled-prompt@1",
                                           "cache_key": "k", "n_tokens": "7"}}
    serializer = MLXPromptCacheSerializer()
    _cache, metadata = serializer.load(tmp_path / "x.safetensors")
    assert metadata["weight_skills_schema"] == "weight-skills/compiled-prompt@1"
    assert metadata["weight_skills_cache_key"] == "k"
    assert metadata["weight_skills_n_tokens"] == "7"


def test_normalise_metadata_handles_flat_legacy_keys():
    flat = {"weight-skills.schema": "s", "weight-skills.n_tokens": "3", "other": "x"}
    out = normalise_metadata(flat)
    assert out["weight_skills_schema"] == "s" and out["weight_skills_n_tokens"] == "3" and out["other"] == "x"


def test_default_serializer_delegates_to_mlx_lm(monkeypatch, tmp_path):
    state = _install_fake_mlx_lm(monkeypatch)
    serializer = MLXPromptCacheSerializer()
    path = tmp_path / "cache.safetensors"

    serializer.save(path, [FakeCache(5, "a")], {"weight_skills_n_tokens": "3"})
    saved_path, saved_cache, saved_metadata = state["saved"]
    assert saved_path == str(path)
    assert saved_cache == [FakeCache(5, "a")]
    assert saved_metadata == {"weight_skills_n_tokens": "3"}

    state["cache"] = [FakeCache(11, "loaded")]
    state["metadata"] = {"weight_skills_n_tokens": "3", "weight_skills_cache_key": "k"}
    cache, metadata = serializer.load(path)
    assert cache[0].data == "loaded"
    assert metadata["weight_skills_cache_key"] == "k"
    assert state["loads"] == [(str(path), True)]


def test_persistence_through_default_serializer(monkeypatch, tmp_path):
    state = _install_fake_mlx_lm(monkeypatch)
    state["cache"] = [FakeCache(12, "compiled")]
    state["metadata"] = {"weight_skills_n_tokens": "9", "weight_skills_cache_key": "persona"}

    library = CompiledPromptLibrary(max_bytes=1000, persist_dir=tmp_path / "compiled")
    library.store("persona", [FakeCache(12, "compiled")], n_tokens=9)
    path = library.save("persona")
    assert path.is_file()

    library.clear()
    assert library.load(path) == "persona"
    assert library.get("persona")[0].data == "compiled"
    assert library.stats()["bytes"] == 12
    assert library.stats()["tokens"] == 9
    assert library.stats()["disk_saves"] == 1
    assert library.stats()["disk_loads"] == 1


# --------------------------------------------------------------------- laziness
def test_importing_serving_does_not_import_mlx():
    repo_root = Path(__file__).resolve().parents[1]
    code = "import sys; import weight_skills.serving; print(int('mlx' in sys.modules or 'mlx_lm' in sys.modules))"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", result.stdout + result.stderr


# --------------------------------------------------------------------------- provenance (G.61)
from weight_skills.serving import (  # noqa: E402
    ProvenanceError,
    token_ids_sha256,
)


def test_store_system_records_provenance_and_verifies():
    lib = make_library()
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    assert lib.has_provenance(key)
    provenance = lib.provenance_of(key)
    assert provenance is not None
    assert provenance.prefix_token_sha256 == token_ids_sha256([1, 2, 3])
    assert provenance.prefix_token_count == 3
    assert lib.verify(key, prefix_token_ids=[1, 2, 3]) == provenance
    assert lib.get_verified(key, prefix_token_ids=[1, 2, 3]) is not None


def test_wrong_prefix_raises_and_does_not_consume_a_hit():
    lib = make_library()
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        lib.verify(key, prefix_token_ids=[1, 2, 4])
    with pytest.raises(ProvenanceError):
        lib.get_verified(key, prefix_token_ids=[9])
    assert lib.hits == 0 and lib.misses == 0


def test_bound_cache_requires_ids():
    lib = make_library()
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    with pytest.raises(ProvenanceError, match="prefix_token_ids"):
        lib.verify(key, prefix_token_ids=None)


def test_unbound_cache_verifies_when_not_required():
    lib = make_library()
    key = lib.store_system("sys", FakeCache(10), 3)
    assert not lib.has_provenance(key)
    assert lib.get_verified(key, prefix_token_ids=[1, 2, 3]) is not None


def test_require_provenance_rejects_unrecorded_cache():
    lib = make_library(require_provenance=True)
    key = lib.store_system("sys", FakeCache(10), 3)
    with pytest.raises(ProvenanceError, match="no prefix token checksum"):
        lib.verify(key, prefix_token_ids=[1, 2, 3])


def test_model_binding_mismatch_raises():
    lib = make_library(model_id="model-a", dtype="bf16")
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    assert lib.provenance_of(key).model_id == "model-a"
    assert lib.verify(key, prefix_token_ids=[1, 2, 3], model_id="model-a") is not None
    with pytest.raises(ProvenanceError, match="model_id mismatch"):
        lib.verify(key, prefix_token_ids=[1, 2, 3], model_id="model-b")
    with pytest.raises(ProvenanceError, match="dtype mismatch"):
        lib.verify(key, prefix_token_ids=[1, 2, 3], dtype="fp16")


def test_provenance_survives_persistence_roundtrip(tmp_path):
    serializer = JSONFileSerializer()
    lib = make_library(serializer=serializer, persist_dir=tmp_path, model_id="m1")
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    path = lib.save(key)
    restored = make_library(serializer=serializer)
    loaded_key = restored.load(path)
    assert loaded_key == key
    assert restored.has_provenance(key)
    assert restored.provenance_of(key).prefix_token_sha256 == token_ids_sha256([1, 2, 3])
    assert restored.provenance_of(key).model_id == "m1"
    assert restored.get_verified(key, prefix_token_ids=[1, 2, 3]) is not None
    with pytest.raises(ProvenanceError):
        restored.get_verified(key, prefix_token_ids=[3, 2, 1])


def test_provenance_stats_count_and_get_verified_forks():
    lib = make_library()
    lib.store_system("a", FakeCache(10), 1, prefix_token_ids=[1])
    lib.store_system("b", FakeCache(10), 1)
    assert lib.stats()["provenance_entries"] == 1
    first = lib.get_verified(prompt_key("a"), prefix_token_ids=[1])
    second = lib.get_verified(prompt_key("a"), prefix_token_ids=[1])
    assert first is not second and first == second


# --------------------------------------------------------------------------- agent-keyed caches (G.67)
from weight_skills.serving import agent_cache_key, text_sha256  # noqa: E402

AGENT_TEXT = "You are the code reviewer. Review changes read-only."
SKILLS_TEXT = "## Skill: code-review\nCheck correctness and style."


def test_agent_cache_key_is_deterministic_and_identity_sensitive():
    k1 = agent_cache_key("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT)
    k2 = agent_cache_key("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT)
    assert k1 == k2 and len(k1) == 64
    assert agent_cache_key("test-engineer-swe", AGENT_TEXT, SKILLS_TEXT) != k1
    assert agent_cache_key("code-reviewer-swe", AGENT_TEXT + " ", SKILLS_TEXT) != k1
    assert agent_cache_key("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT + " ") != k1


def test_agent_cache_key_rejects_empty_id():
    with pytest.raises(ValueError):
        agent_cache_key("", AGENT_TEXT, SKILLS_TEXT)


def test_store_agent_and_get_agent_verified():
    lib = make_library()
    key = lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3,
                          prefix_token_ids=[1, 2, 3])
    assert key == agent_cache_key("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT)
    assert lib.has_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT)
    provenance = lib.provenance_of(key)
    assert provenance.agent_id == "code-reviewer-swe"
    assert provenance.agent_sha256 == text_sha256(AGENT_TEXT)
    assert provenance.skills_sha256 == text_sha256(SKILLS_TEXT)
    cache = lib.get_agent_verified("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, prefix_token_ids=[1, 2, 3])
    assert cache == FakeCache(10)


def test_wrong_agent_is_rejected():
    lib = make_library()
    lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    # different agent id -> different key -> miss/KeyError path: get_agent_verified raises "no compiled prompt"
    with pytest.raises(ProvenanceError):
        lib.get_agent_verified("test-engineer-swe", AGENT_TEXT, SKILLS_TEXT, prefix_token_ids=[1, 2, 3])
    # same key name but a *wrong recorded agent binding* must raise a binding mismatch
    with pytest.raises(ProvenanceError, match="agent_id mismatch"):
        lib.verify(agent_cache_key("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT), prefix_token_ids=[1, 2, 3],
                   agent_id="test-engineer-swe")


def test_changed_skills_are_rejected():
    lib = make_library()
    lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    changed = SKILLS_TEXT + "## Skill: extra\nNew skill."
    with pytest.raises(ProvenanceError):
        lib.get_agent_verified("code-reviewer-swe", AGENT_TEXT, changed, prefix_token_ids=[1, 2, 3])
    assert not lib.has_agent("code-reviewer-swe", AGENT_TEXT, changed)


def test_agent_cache_rejects_wrong_prefix():
    lib = make_library()
    lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        lib.get_agent_verified("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, prefix_token_ids=[9])


def test_agent_cache_require_provenance_rejects_unrecorded_prefix():
    lib = make_library(require_provenance=True)
    lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3)
    with pytest.raises(ProvenanceError, match="no prefix token checksum"):
        lib.get_agent_verified("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, prefix_token_ids=[1, 2, 3])


def test_agent_cache_persistence_roundtrip(tmp_path):
    serializer = JSONFileSerializer()
    lib = make_library(serializer=serializer, persist_dir=tmp_path, model_id="m1")
    key = lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10, "x"), 3,
                          prefix_token_ids=[1, 2, 3])
    path = lib.save(key, tmp_path / "agent.safetensors")
    restored = make_library(serializer=serializer)
    loaded = restored.load(path)
    provenance = restored.provenance_of(loaded)
    assert loaded == key
    assert provenance.agent_id == "code-reviewer-swe"
    assert provenance.agent_sha256 == text_sha256(AGENT_TEXT)
    assert provenance.skills_sha256 == text_sha256(SKILLS_TEXT)
    assert provenance.model_id == "m1"
    assert restored.get_agent_verified("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, prefix_token_ids=[1, 2, 3]) is not None


def test_agent_stats_count():
    lib = make_library()
    lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1])
    lib.store_system("plain system", FakeCache(10), 1)
    stats = lib.stats()
    assert stats["agent_entries"] == 1
    assert stats["provenance_entries"] == 1  # the plain system entry records no prefix checksum


# ------------------------------------------------------------------ equivalence label (G.69)
def test_require_provenance_defaults_to_true():
    lib = CompiledPromptLibrary(max_bytes=1000)
    lib.store_system("plain system", FakeCache(10), 1)
    with pytest.raises(ProvenanceError, match="no prefix token checksum"):
        lib.get_verified(prompt_key("plain system"), prefix_token_ids=[1])


def test_store_system_records_equivalence_label():
    lib = make_library()
    key = lib.store_system("system text", FakeCache(10), 2, prefix_token_ids=[1, 2], equivalence="certified")
    assert lib.equivalence_of(key) == "certified"
    assert lib.verify(key, prefix_token_ids=[1, 2]).equivalence == "certified"
    plain = lib.store_system("other text", FakeCache(10), 2, prefix_token_ids=[3, 4])
    assert lib.equivalence_of(plain) is None


def test_store_agent_records_equivalence_label():
    lib = make_library()
    key = lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3,
                          prefix_token_ids=[1, 2, 3], equivalence="unverified")
    assert lib.equivalence_of(key) == "unverified"


def test_equivalence_label_persists(tmp_path):
    serializer = JSONFileSerializer()
    lib = make_library(serializer=serializer, persist_dir=tmp_path, model_id="m1")
    key = lib.store_system("system text", FakeCache(10, "x"), 2, prefix_token_ids=[1, 2],
                           equivalence="certified")
    path = lib.save(key, tmp_path / "labeled.safetensors")
    restored = make_library(serializer=serializer)
    loaded = restored.load(path)
    assert restored.equivalence_of(loaded) == "certified"


# ------------------------------------------------------- memory accounting (G.80)
from weight_skills.serving import (  # noqa: E402
    aligned_suffix_start,
    alignment_options,
    attention_score_bytes,
    dtype_bytes,
    estimate_peak,
    kv_cache_bytes,
    plan_aligned_compile,
    plan_prefill_chunk,
)


def test_attention_score_bytes_matches_65k_oom():
    # 16 heads x 65,536^2 x 2 B = 137.44 GB; the observed 35B OOM allocation was 137,573,203,968 B.
    b = attention_score_bytes(16, 65536, 65536, dtype_bytes=2)
    assert b == 16 * 65536 * 65536 * 2
    assert abs(b - 137_573_203_968) / 137_573_203_968 < 0.01


def test_attention_score_bytes_chunking_scales_linearly():
    full = attention_score_bytes(16, 65536, 65536)
    chunked = attention_score_bytes(16, 2048, 65536)
    assert full // chunked == 32


def test_kv_cache_bytes_and_fraction():
    full = kv_cache_bytes(40, 2, 256, 65536)
    assert full == 2 * 40 * 2 * 256 * 65536 * 2
    hybrid = kv_cache_bytes(40, 2, 256, 65536, full_attention_fraction=0.25)
    assert hybrid == 2 * 10 * 2 * 256 * 65536 * 2


def test_dtype_bytes_validation():
    assert dtype_bytes("bfloat16") == 2 and dtype_bytes("float32") == 4
    with pytest.raises(ValueError):
        dtype_bytes("int4")


def test_estimate_peak_sums_components():
    est = estimate_peak(weights_gb=19.0, context_tokens=65536, chunk_tokens=2048,
                        num_attention_heads=16, kv_bytes_per_token=21_500, overhead_gb=3.0)
    assert est.total_bytes == est.weights_bytes + est.kv_cache_bytes + est.attention_buffer_bytes + est.overhead_bytes
    assert est.attention_buffer_bytes == attention_score_bytes(16, 2048, 65536)
    assert abs(est.total_gb - (19.0 + 3.0 + 21_500 * 65536 / 1e9 + attention_score_bytes(16, 2048, 65536) / 1e9)) < 1e-6


def test_plan_prefill_chunk_picks_largest_fitting():
    common = dict(weights_gb=19.0, context_tokens=65536, num_attention_heads=16, kv_bytes_per_token=21_500,
                  overhead_gb=3.0)
    assert plan_prefill_chunk(budget_gb=40.0, **common) == 4096
    assert plan_prefill_chunk(budget_gb=150.0, **common) == 16384
    assert plan_prefill_chunk(budget_gb=26.0, **common) == 1024


def test_plan_prefill_chunk_returns_none_when_impossible():
    assert plan_prefill_chunk(weights_gb=19.0, context_tokens=65536, num_attention_heads=16,
                              kv_bytes_per_token=21_500, budget_gb=20.0) is None


def test_memory_inputs_validated():
    with pytest.raises(ValueError):
        attention_score_bytes(0, 10, 10)
    with pytest.raises(ValueError):
        estimate_peak(weights_gb=1, context_tokens=0, chunk_tokens=1, num_attention_heads=1,
                      kv_bytes_per_token=1)
    with pytest.raises(ValueError):
        plan_prefill_chunk(weights_gb=1, context_tokens=1, num_attention_heads=1,
                           kv_bytes_per_token=1, budget_gb=0)


# ------------------------------------------------ engine memory planner (D26/G.81)
from types import SimpleNamespace  # noqa: E402

from weight_skills.proxy.engine import model_memory_plan, ram_budget_gb  # noqa: E402


def _qwen35_args():
    layer_types = ["linear_attention"] * 40
    for i in range(3, 40, 4):
        layer_types[i] = "full_attention"
    return SimpleNamespace(num_attention_heads=16, num_key_value_heads=2, head_dim=256,
                           num_hidden_layers=40, hidden_size=2048, layer_types=layer_types)


def test_model_memory_plan_35b_65k_budgets():
    plan = model_memory_plan(_qwen35_args(), weights_gb=20.4, context_tokens=65536, budget_gb=40.0)
    assert plan["chunk"] == 4096
    assert plan["full_attention_layers"] == 10
    assert plan["kv_bytes_per_token"] == 2 * 10 * 2 * 256 * 2
    assert plan["estimate_gb"] <= 40.0
    tight = model_memory_plan(_qwen35_args(), weights_gb=20.4, context_tokens=65536, budget_gb=26.0)
    assert tight["chunk"] == 1024
    impossible = model_memory_plan(_qwen35_args(), weights_gb=20.4, context_tokens=65536, budget_gb=20.0)
    assert impossible["chunk"] is None


def test_model_memory_plan_derives_head_dim_and_kv_heads():
    args = SimpleNamespace(num_attention_heads=14, num_hidden_layers=24, hidden_size=896,
                           max_position_embeddings=32768)
    plan = model_memory_plan(args, weights_gb=1.0, context_tokens=8192, budget_gb=16.0)
    assert plan["head_dim"] == 64  # 896 // 14
    assert plan["kv_heads"] == 14  # falls back to heads when absent
    assert plan["full_attention_layers"] == 24


def test_model_memory_plan_validates_args():
    with pytest.raises(ValueError):
        model_memory_plan(SimpleNamespace(num_hidden_layers=2), weights_gb=1.0, context_tokens=100, budget_gb=10)
    with pytest.raises(ValueError):
        ram_budget_gb(override=0)


def test_model_memory_plan_handles_nested_text_config():
    text = {"num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "num_hidden_layers": 40, "hidden_size": 2048,
            "layer_types": ["linear_attention"] * 40}
    for i in range(3, 40, 4):
        text["layer_types"][i] = "full_attention"
    args = SimpleNamespace(text_config=text, model_type="qwen3_5_moe")
    plan = model_memory_plan(args, weights_gb=20.4, context_tokens=65536, budget_gb=40.0)
    assert plan["chunk"] == 4096
    assert plan["full_attention_layers"] == 10


# ------------------------------------------------ per-model equivalence policy (D19/G.84)
from weight_skills.serving import (  # noqa: E402
    DEFAULT_CERTIFIED_RATES,
    EquivalencePolicy,
    certified_rate_for,
    decide_equivalence_policy,
)


def test_certified_rate_lookup_by_substring():
    assert certified_rate_for("qwen2-0.5b-abliterated") == 0.875
    assert certified_rate_for("Gemma-4-E4B-it-4bit") == 0.625
    assert certified_rate_for("models/qwen36-35b-heretic-4bit") == 0.25
    assert certified_rate_for("unknown-model") is None


def test_policy_fallback_for_high_certification():
    policy = decide_equivalence_policy("qwen2-0.5b-abliterated")
    assert isinstance(policy, EquivalencePolicy)
    assert policy.mode == "fallback" and policy.source == "profile" and policy.fallback


def test_policy_off_for_low_certification_and_unknown():
    assert decide_equivalence_policy("gemma-4-e4b-it-4bit").mode == "off"
    assert decide_equivalence_policy("qwen36-35b-heretic-4bit").mode == "off"
    unknown = decide_equivalence_policy("some-new-model")
    assert unknown.mode == "off" and unknown.source == "unknown-model"


def test_policy_env_override_wins():
    assert decide_equivalence_policy("gemma-4-e4b-it-4bit", env_value="fallback").mode == "fallback"
    assert decide_equivalence_policy("qwen2-0.5b-abliterated", env_value="off").mode == "off"
    assert decide_equivalence_policy("x", env_value=" fallback ").source == "env"
    with pytest.raises(ValueError):
        decide_equivalence_policy("x", env_value="auto")


def test_policy_custom_rate_and_threshold():
    policy = decide_equivalence_policy("m", certified_rate=0.9, threshold=0.8)
    assert policy.mode == "fallback" and policy.certified_rate == 0.9
    assert decide_equivalence_policy("m", certified_rate=0.79).mode == "off"
    with pytest.raises(ValueError):
        decide_equivalence_policy("m", threshold=1.5)


def test_default_rates_are_measured_values():
    assert DEFAULT_CERTIFIED_RATES == {"qwen2-0.5b": 0.875, "gemma-4-e4b": 0.625, "qwen36-35b": 0.25}


# --------------------------------------------------- agent registry for header routing (D17/G.86)
from weight_skills.serving import AgentCacheRegistry  # noqa: E402


def test_agent_registry_register_and_resolve():
    lib = make_library()
    reg = AgentCacheRegistry(lib)
    key = reg.register("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10, "agent"), 3,
                       prefix_token_ids=[1, 2, 3])
    assert reg.key_for("code-reviewer-swe") == key
    assert reg.ids() == ["code-reviewer-swe"]
    assert reg.get("code-reviewer-swe") == FakeCache(10, "agent")
    assert reg.key_for("unknown-agent") is None
    assert reg.get("unknown-agent") is None


def test_agent_registry_unknown_agent_verify_raises():
    reg = AgentCacheRegistry(make_library())
    with pytest.raises(ProvenanceError, match="unknown agent"):
        reg.verify("ghost", prefix_token_ids=[1])


def test_agent_registry_verify_wrong_prefix_raises():
    lib = make_library()
    reg = AgentCacheRegistry(lib)
    reg.register("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    reg.verify("code-reviewer-swe", prefix_token_ids=[1, 2, 3])  # ok
    with pytest.raises(ProvenanceError):
        reg.verify("code-reviewer-swe", prefix_token_ids=[9])


def test_agent_registry_reregister_replaces_key():
    lib = make_library()
    reg = AgentCacheRegistry(lib)
    first = reg.register("a", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3, prefix_token_ids=[1])
    second = reg.register("a", AGENT_TEXT, SKILLS_TEXT + " v2", FakeCache(10), 3, prefix_token_ids=[2])
    assert first != second and reg.key_for("a") == second


# --- chunk-boundary alignment for exact compiled serving (G.93 / D28) -----------------------------
def test_aligned_suffix_start_reports_cut_and_extra():
    cut, extra = aligned_suffix_start(2123, 2048)
    assert (cut, extra) == (2048, 75) and cut % 2048 == 0 and extra < 2048
    assert aligned_suffix_start(8220, 2048) == (8192, 28)
    assert aligned_suffix_start(4096, 2048) == (4096, 0)


def test_aligned_suffix_start_flags_no_compile_benefit():
    plan = plan_aligned_compile(1506, 2048)
    assert plan.cut_tokens == 0 and plan.no_compile_benefit and plan.extra_tokens == 1506
    assert plan.cut_is_chunk_aligned
    assert plan.cost_fraction(question_tokens=71) == pytest.approx(1506 / 1577)


def test_aligned_suffix_start_validation():
    with pytest.raises(ValueError):
        aligned_suffix_start(-1, 2048)
    with pytest.raises(ValueError):
        aligned_suffix_start(100, 0)


def test_alignment_options_ordered_by_cost():
    opts = alignment_options(12062, candidates=(256, 512, 1024, 2048))
    assert [p.chunk_tokens for p in opts] == [256, 512, 1024, 2048]
    assert all(p.extra_tokens < p.chunk_tokens for p in opts)
    assert all(p.cut_tokens % p.chunk_tokens == 0 for p in opts)


def test_alignment_plan_cost_fraction_bounds():
    plan = plan_aligned_compile(8220, 2048)
    assert 0.0 < plan.cost_fraction(question_tokens=70) < 0.01


# ------------------------- exactness contract for aligned serving (D28/G.93) ----------------------
from weight_skills.serving import (  # noqa: E402
    CacheProvenance,
    aligned_exact_contract,
)


def test_alignment_fields_survive_persistence_roundtrip(tmp_path):
    serializer = JSONFileSerializer()
    lib = make_library(serializer=serializer, persist_dir=tmp_path, model_id="m1")
    key = lib.store_system("aligned system", FakeCache(10, "x"), 2123, prefix_token_ids=[1, 2, 3],
                           prefill_chunk=2048, aligned_extra_tokens=75)
    encoded = json.loads(lib.provenance_of(key).to_metadata())
    assert encoded["prefill_chunk"] == 2048
    assert encoded["aligned_extra_tokens"] == 75

    path = lib.save(key, tmp_path / "aligned.safetensors")
    restored = make_library(serializer=serializer)
    loaded = restored.load(path)
    provenance = restored.provenance_of(loaded)
    assert provenance.prefill_chunk == 2048
    assert provenance.aligned_extra_tokens == 75


def test_alignment_fields_default_to_none():
    lib = make_library()
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    provenance = lib.provenance_of(key)
    assert provenance.prefill_chunk is None
    assert provenance.aligned_extra_tokens is None
    plain = CacheProvenance()
    assert plain.prefill_chunk is None and plain.aligned_extra_tokens is None
    assert "prefill_chunk" in json.loads(plain.to_metadata())


def test_aligned_exact_contract_ok_when_chunks_match():
    provenance = CacheProvenance(prefix_token_count=2123, prefill_chunk=2048, aligned_extra_tokens=75)
    assert aligned_exact_contract(provenance, 2048) == (True, "")
    assert aligned_exact_contract(CacheProvenance(prefill_chunk=4096), 4096) == (True, "")


def test_aligned_exact_contract_reports_chunk_mismatch():
    provenance = CacheProvenance(prefill_chunk=4096)
    assert aligned_exact_contract(provenance, 2048) == (
        False, "chunk mismatch: recorded 4096 != serving 2048"
    )


def test_aligned_exact_contract_rejects_unrecorded_chunk():
    expected = (False, "no recorded prefill chunk (plain/sampled artifact)")
    assert aligned_exact_contract(None, 2048) == expected
    assert aligned_exact_contract(CacheProvenance(), 2048) == expected
    assert aligned_exact_contract(CacheProvenance(equivalence="unverified"), 2048) == expected


def test_old_style_provenance_metadata_has_no_alignment_fields():
    old_metadata = {
        "text_key": prompt_key("sys"),
        "prefix_token_sha256": token_ids_sha256([1, 2, 3]),
        "prefix_token_count": 3,
        "model_id": "m1",
        "tokenizer_id": "tok",
        "dtype": "bf16",
        "agent_id": None,
        "agent_sha256": None,
        "skills_sha256": None,
        "equivalence": "certified",
    }
    provenance = CacheProvenance.from_metadata(json.dumps(old_metadata))
    assert provenance is not None
    assert provenance.prefill_chunk is None
    assert provenance.aligned_extra_tokens is None
    assert provenance.equivalence == "certified"
    assert aligned_exact_contract(provenance, 2048)[0] is False


def test_alignment_options_satisfy_boundary_and_contract():
    prefix = 12062
    options = alignment_options(prefix, candidates=(256, 512, 1024, 2048, 4096))
    assert options
    for option in options:
        cut, extra = aligned_suffix_start(prefix, option.chunk_tokens)
        assert (cut, extra) == (option.cut_tokens, option.extra_tokens)
        assert cut + extra == prefix
        assert cut % option.chunk_tokens == 0
        assert extra < option.chunk_tokens
    plan = plan_aligned_compile(prefix, 2048)
    provenance = CacheProvenance(prefill_chunk=plan.chunk_tokens,
                                 aligned_extra_tokens=plan.extra_tokens,
                                 prefix_token_count=prefix)
    assert aligned_exact_contract(provenance, 2048) == (True, "")
    assert aligned_exact_contract(provenance, 1024) == (
        False, "chunk mismatch: recorded 2048 != serving 1024"
    )


def test_store_agent_and_library_defaults_record_alignment_fields():
    lib = make_library(prefill_chunk=1024, aligned_extra_tokens=7)
    key = lib.store_system("sys", FakeCache(10), 1031, prefix_token_ids=[1, 2])
    assert lib.provenance_of(key).prefill_chunk == 1024
    assert lib.provenance_of(key).aligned_extra_tokens == 7

    explicit = lib.store_system("sys2", FakeCache(10), 3, prefix_token_ids=[3],
                                prefill_chunk=4096, aligned_extra_tokens=0)
    assert lib.provenance_of(explicit).prefill_chunk == 4096
    assert lib.provenance_of(explicit).aligned_extra_tokens == 0

    agent_key = lib.store_agent("code-reviewer-swe", AGENT_TEXT, SKILLS_TEXT, FakeCache(10), 3,
                                prefix_token_ids=[1, 2, 3], prefill_chunk=2048,
                                aligned_extra_tokens=75)
    agent_provenance = lib.provenance_of(agent_key)
    assert agent_provenance.prefill_chunk == 2048
    assert agent_provenance.aligned_extra_tokens == 75


# ---------------------------------------- aligned-exact serving mode (D28/D30) --------------------
from weight_skills.proxy.engine import MLXEngine, MockEngine  # noqa: E402
from weight_skills.serving import (  # noqa: E402
    build_policy_rows,
    decide_aligned_serving,
    plan_aligned_prompt,
)
from weight_skills.serving.policy import (  # noqa: E402
    ALIGNED_GUARANTEE,
)
from weight_skills.serving.policy import (  # noqa: E402
    main as policy_main,
)


def test_env_aligned_exact_requires_explicit_support():
    # fail closed: without the D28/D30 serving path, selecting the mode must raise, not silently sample
    with pytest.raises(ValueError, match="aligned serving path"):
        decide_equivalence_policy("some-model", env_value=" ALIGNED_EXACT ")
    policy = decide_equivalence_policy("some-model", env_value=" ALIGNED_EXACT ",
                                       supports_aligned_exact=True)
    assert policy.mode == "aligned_exact"
    assert policy.aligned_exact is True
    assert policy.fallback is False
    assert policy.source == "env"
    assert decide_equivalence_policy("other", env_value="aligned_exact",
                                     supports_aligned_exact=True).mode == "aligned_exact"


def test_mlx_engine_supports_aligned_exact_and_mock_stays_fail_closed():
    # the MLX engine now carries the aligned serving path (D28/D30)...
    engine = MLXEngine(model_path="/tmp/does-not-exist", equivalence_policy="aligned_exact")
    assert engine.prefill_step_size == 2048
    assert engine.last_alignment_refusal is None
    # ...while the mock engine has no such path and keeps failing closed for the env value
    with pytest.raises(ValueError, match="aligned serving path"):
        MockEngine(equivalence_policy="aligned_exact")


def test_aligned_exact_mode_is_not_a_per_model_default():
    assert decide_equivalence_policy("qwen2-0.5b-abliterated").mode == "fallback"
    assert decide_equivalence_policy("gemma-4-e4b-it-4bit").mode == "off"
    assert decide_equivalence_policy("unknown-model").mode == "off"
    with pytest.raises(ValueError):
        decide_equivalence_policy("x", env_value="aligned")


def test_decide_aligned_serving_accepts_matching_chunk():
    provenance = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75)
    assert decide_aligned_serving(provenance, 2048) == ("aligned_exact", "")


def test_decide_aligned_serving_refuses_chunk_mismatch():
    provenance = CacheProvenance(prefill_chunk=4096)
    assert decide_aligned_serving(provenance, 2048) == (
        "cold", "chunk mismatch: recorded 4096 != serving 2048"
    )


def test_decide_aligned_serving_refuses_unrecorded_chunk():
    expected = ("cold", "no recorded prefill chunk (plain/sampled artifact)")
    assert decide_aligned_serving(None, 2048) == expected
    assert decide_aligned_serving(CacheProvenance(), 2048) == expected


def test_engine_aligned_serve_plan_accepts_and_refuses():
    engine = MLXEngine(model_path="models/demo-aligned")
    assert engine._aligned_serve_plan(CacheProvenance(prefill_chunk=2048), 2048) == (
        "aligned_exact", ""
    )
    assert engine._aligned_serve_plan(CacheProvenance(prefill_chunk=1024), 2048) == (
        "cold", "chunk mismatch: recorded 1024 != serving 2048"
    )
    assert engine._aligned_serve_plan(None, 2048) == (
        "cold", "no recorded prefill chunk (plain/sampled artifact)"
    )


def test_build_policy_rows_shape_from_flat_measurements():
    rows = build_policy_rows(
        [
            {
                "model": "model-a",
                "evidence_artefact": "reports/COMPILED_SERVING_ALIGNMENT.json",
                "aligned_cells_bit_identical": 5,
                "roundtrip_cells": 4,
                "median_ttft_ratio": 9.8,
            },
            {
                "model": "model-b",
                "evidence_artefact": "reports/COMPILED_SERVING_ALIGNMENT.json",
                "aligned_cells_bit_identical": 3,
                "roundtrip_cells": 2,
                "median_ttft_ratio": 7.5,
            },
        ]
    )
    assert rows == [
        {
            "model": "model-a",
            "mode": "aligned_exact",
            "guarantee": ALIGNED_GUARANTEE,
            "evidence_artefact": "reports/COMPILED_SERVING_ALIGNMENT.json",
            "measured": {
                "aligned_cells_bit_identical": 5,
                "roundtrip_cells": 4,
                "median_ttft_ratio": 9.8,
                "batch_limit": "1",
            },
        },
        {
            "model": "model-b",
            "mode": "aligned_exact",
            "guarantee": ALIGNED_GUARANTEE,
            "evidence_artefact": "reports/COMPILED_SERVING_ALIGNMENT.json",
            "measured": {
                "aligned_cells_bit_identical": 3,
                "roundtrip_cells": 2,
                "median_ttft_ratio": 7.5,
                "batch_limit": "1",
            },
        },
    ]
    assert ALIGNED_GUARANTEE == (
        "bit-identical to the same serving routine's cold prefill "
        "(batch=1, same chunk/dtype/process)"
    )


def test_build_policy_rows_reads_artefact_style_entries():
    entries = [
        {
            "model": {"id": "model-c", "path": "/models/c"},
            "aggregates": {
                "arms": {
                    "aligned": {"n": 4, "bit_identical": 4, "median_ttft_ratio": 6.1},
                    "aligned_roundtrip": {"n": 3, "bit_identical": 3},
                },
                "median_ttft_ratio": 6.1,
            },
        }
    ]
    rows = build_policy_rows(
        entries, evidence_artefact="reports/COMPILED_SERVING_ALIGNMENT.json"
    )
    assert rows[0]["model"] == "model-c"
    assert rows[0]["measured"] == {
        "aligned_cells_bit_identical": 4,
        "roundtrip_cells": 3,
        "median_ttft_ratio": 6.1,
        "batch_limit": "1",
    }


def test_build_policy_rows_refuses_missing_measurements():
    with pytest.raises(ValueError, match="roundtrip_cells"):
        build_policy_rows(
            [
                {
                    "model": "m",
                    "evidence_artefact": "reports/x.json",
                    "aligned_cells_bit_identical": 5,
                    "median_ttft_ratio": 1.0,
                }
            ]
        )
    with pytest.raises(ValueError, match="evidence_artefact"):
        build_policy_rows(
            [
                {
                    "model": "m",
                    "aligned_cells_bit_identical": 5,
                    "roundtrip_cells": 4,
                    "median_ttft_ratio": 1.0,
                }
            ]
        )
    with pytest.raises(ValueError, match="arms"):
        build_policy_rows([{"model": "m", "evidence_artefact": "reports/x.json", "aggregates": {}}])


def test_policy_main_demo_is_clearly_labelled(capsys):
    assert policy_main(["--demo"]) == 0
    captured = capsys.readouterr()
    assert "DEMO ONLY" in captured.err
    rows = json.loads(captured.out)
    assert rows[0]["model"].startswith("DEMO")
    assert "DEMO" in rows[0]["evidence_artefact"]
    assert rows[0]["mode"] == "aligned_exact"
    assert rows[0]["measured"]["batch_limit"] == "1"
    assert policy_main([]) == 2


# ------------------------- aligned prompt planner + engine request path (D28/D30) -----------------
def test_plan_aligned_prompt_accepts_aligned_cache():
    provenance = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75, prefix_token_count=25)
    assert plan_aligned_prompt(provenance, 100, 2048) == ("aligned_exact", 25, "")
    # W == 0: prefix already on the boundary, the whole prefix is cached
    on_boundary = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=0, prefix_token_count=100)
    assert plan_aligned_prompt(on_boundary, 100, 2048) == ("aligned_exact", 100, "")


def test_plan_aligned_prompt_refuses_chunk_contract():
    missing = ("cold", None, "no recorded prefill chunk (plain/sampled artifact)")
    assert plan_aligned_prompt(None, 100, 2048) == missing
    assert plan_aligned_prompt(
        CacheProvenance(aligned_extra_tokens=0, prefix_token_count=100), 100, 2048
    ) == missing
    mismatch = CacheProvenance(prefill_chunk=4096, aligned_extra_tokens=0, prefix_token_count=100)
    assert plan_aligned_prompt(mismatch, 100, 2048) == (
        "cold", None, "chunk mismatch: recorded 4096 != serving 2048"
    )


def test_plan_aligned_prompt_requires_aligned_extra_tokens():
    provenance = CacheProvenance(prefill_chunk=2048, prefix_token_count=100)
    mode, cut, reason = plan_aligned_prompt(provenance, 100, 2048)
    assert (mode, cut) == ("cold", None)
    assert "no recorded aligned_extra_tokens" in reason


def test_plan_aligned_prompt_rejects_out_of_range_extra():
    too_big = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=101, prefix_token_count=0)
    mode, cut, reason = plan_aligned_prompt(too_big, 100, 2048)
    assert (mode, cut) == ("cold", None)
    assert "outside prefix length" in reason
    negative = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=-1, prefix_token_count=101)
    mode, cut, reason = plan_aligned_prompt(negative, 100, 2048)
    assert (mode, cut) == ("cold", None)
    assert "outside prefix length" in reason


def test_plan_aligned_prompt_rejects_count_mismatch():
    provenance = CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75, prefix_token_count=30)
    mode, cut, reason = plan_aligned_prompt(provenance, 100, 2048)
    assert (mode, cut) == ("cold", None)
    assert reason == (
        "prefix_token_count mismatch: recorded 30 != prefix_tokens_total 100 - "
        "aligned_extra_tokens 75 = 25"
    )


class _AlignedLibrary:
    """Fake library exposing provenance + a verified cache for the aligned engine path."""

    def __init__(self, provenance, cache, expected_prefix_ids):
        self.provenance = provenance
        self.cache = cache
        self.expected_prefix_ids = list(expected_prefix_ids)
        self.verify_calls: list[list[int]] = []
        self.get_calls = 0

    def __contains__(self, key: object) -> bool:
        return key == "aligned-key"

    def provenance_of(self, key: object):
        return self.provenance if key == "aligned-key" else None

    def verify(self, key, *, prefix_token_ids, model_id=None):
        assert list(prefix_token_ids) == self.expected_prefix_ids
        self.verify_calls.append(list(prefix_token_ids))

    def get_verified(self, key, *, prefix_token_ids, model_id=None):
        assert list(prefix_token_ids) == self.expected_prefix_ids
        self.get_calls += 1
        return self.cache


def _aligned_engine(full_prefix, rendered_ids, library, *, chunk=2048):
    engine = MLXEngine(
        model_path="models/fake",
        prompt_library=library,
        equivalence_policy="aligned_exact",
        prefill_step_size=chunk,
    )
    engine._model = object()  # load() returns immediately: no MLX import in this test
    engine._tokenizer = type("_Tok", (), {"encode": lambda self, text: list(rendered_ids)})()
    engine.render_chat = lambda messages: "rendered"  # type: ignore[method-assign]
    engine._prefix_ids = lambda messages: list(full_prefix)  # type: ignore[method-assign]
    return engine


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_compiled_prompt_aligned_serves_suffix_from_cut():
    full_prefix = list(range(100))
    rendered = full_prefix + [1000, 1001]
    cache = FakeCache(10, "aligned")
    library = _AlignedLibrary(
        CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75, prefix_token_count=25),
        cache,
        expected_prefix_ids=full_prefix[:25],
    )
    engine = _aligned_engine(full_prefix, rendered, library)
    compiled = engine._compiled_prompt(MESSAGES, "aligned-key")
    assert compiled is not None
    suffix, returned = compiled
    assert suffix == rendered[25:]
    assert returned is cache
    assert engine.last_alignment_refusal is None
    assert library.get_calls == 1


def test_compiled_prompt_aligned_refusal_is_cold_and_skips_cache():
    full_prefix = list(range(100))
    rendered = full_prefix + [999]
    library = _AlignedLibrary(
        CacheProvenance(prefill_chunk=4096, aligned_extra_tokens=75, prefix_token_count=25),
        FakeCache(10),
        expected_prefix_ids=[],
    )
    engine = _aligned_engine(full_prefix, rendered, library)
    assert engine._compiled_prompt(MESSAGES, "aligned-key") is None
    assert engine.last_alignment_refusal == "chunk mismatch: recorded 4096 != serving 2048"
    assert library.get_calls == 0


def test_compiled_prompt_aligned_refuses_on_serving_chunk_mismatch():
    full_prefix = list(range(100))
    rendered = full_prefix + [999]
    library = _AlignedLibrary(
        CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75, prefix_token_count=25),
        FakeCache(10),
        expected_prefix_ids=[],
    )
    engine = _aligned_engine(full_prefix, rendered, library, chunk=1024)
    assert engine._compiled_prompt(MESSAGES, "aligned-key") is None
    assert engine.last_alignment_refusal == "chunk mismatch: recorded 2048 != serving 1024"
    assert library.get_calls == 0


def test_complete_messages_records_no_cache_refusal_in_aligned_mode():
    # a request without a compiled cache is served cold and must SAY so (never report "exact")
    engine = MLXEngine(model_path="models/fake", equivalence_policy="aligned_exact")
    engine.last_alignment_refusal = "stale refusal"
    engine.render_chat = lambda messages: "rendered"  # type: ignore[method-assign]
    engine.complete = lambda prompt, **kwargs: "cold"  # type: ignore[method-assign]
    out = engine.complete_messages(
        [{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.0
    )
    assert out == "cold"
    assert engine.last_alignment_refusal == "no compiled cache for this request (served cold)"


def test_complete_messages_records_no_cache_refusal_for_unknown_key():
    engine = MLXEngine(model_path="models/fake", equivalence_policy="aligned_exact",
                       prompt_library=_AlignedLibrary(
                           CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=1,
                                           prefix_token_count=1), FakeCache(1),
                           expected_prefix_ids=[]))
    engine.last_alignment_refusal = "stale refusal"
    engine.render_chat = lambda messages: "rendered"  # type: ignore[method-assign]
    engine.complete = lambda prompt, **kwargs: "cold"  # type: ignore[method-assign]
    out = engine.complete_messages([{"role": "user", "content": "hi"}],
                                   max_tokens=4, temperature=0.0, prompt_key="no-such-key")
    assert out == "cold"
    assert engine.last_alignment_refusal == "no compiled cache for this request (served cold)"


def test_verify_prompt_aligned_verifies_cut_prefix():
    full_prefix = list(range(100))
    library = _AlignedLibrary(
        CacheProvenance(prefill_chunk=2048, aligned_extra_tokens=75, prefix_token_count=25),
        FakeCache(10),
        expected_prefix_ids=full_prefix[:25],
    )
    engine = _aligned_engine(full_prefix, full_prefix + [7], library)
    engine.verify_prompt(MESSAGES, "aligned-key")
    assert library.verify_calls == [full_prefix[:25]]
    assert engine.last_alignment_refusal is None


def test_verify_prompt_aligned_refusal_sets_reason():
    full_prefix = list(range(100))
    library = _AlignedLibrary(
        CacheProvenance(prefill_chunk=2048, prefix_token_count=25),
        FakeCache(10),
        expected_prefix_ids=[],
    )
    engine = _aligned_engine(full_prefix, full_prefix + [7], library)
    engine.verify_prompt(MESSAGES, "aligned-key")
    assert library.verify_calls == []
    assert "no recorded aligned_extra_tokens" in engine.last_alignment_refusal


# ---------------------------------- persistent LRU reload for evicted caches (D1/G.110) ----------
class _FailingLoadSerializer(JSONFileSerializer):
    """Serializer whose load always fails (corrupt artifact simulation)."""

    def load(self, path: Path) -> tuple[Any, dict[str, str]]:
        self.loads += 1
        raise OSError("corrupt cache artifact")


class _FailingSaveSerializer(JSONFileSerializer):
    """Serializer whose save always fails (full/broken disk simulation)."""

    def save(self, path: Path, cache: Any, metadata: dict[str, str]) -> None:
        raise OSError("disk full")


def _persistent_library(serializer: JSONFileSerializer, tmp_path: Path, **kwargs: Any) -> CompiledPromptLibrary:
    kwargs.setdefault("max_bytes", 100)
    return make_library(serializer=serializer, persist_dir=tmp_path, **kwargs)


def test_eviction_persists_cache_before_removal(tmp_path):
    serializer = JSONFileSerializer()
    library = _persistent_library(serializer, tmp_path)
    library.store("a", FakeCache(100, "alpha"), n_tokens=3)
    library.store("b", FakeCache(100, "beta"), n_tokens=4)  # evicts a

    assert serializer.saves == 1
    path = tmp_path / "a.safetensors"
    assert path.is_file()
    metadata = json.loads(path.read_text())["metadata"]
    assert metadata["weight_skills_n_tokens"] == "3"
    assert metadata["weight_skills_cache_key"] == "a"
    assert metadata["weight_skills_schema"] == "weight-skills/compiled-prompt@1"
    stats = library.stats()
    assert stats["persisted"] == 1
    assert stats["evictions"] == 1
    assert "a" not in library
    assert library.names() == ["b"]


def test_get_miss_reloads_and_becomes_most_recent(tmp_path):
    serializer = JSONFileSerializer()
    library = _persistent_library(serializer, tmp_path, max_bytes=300)
    library.store("a", FakeCache(100, "alpha"), n_tokens=1)
    library.store("b", FakeCache(100, "beta"), n_tokens=2)
    library.store("c", FakeCache(50, "gamma"), n_tokens=3)
    library.store("d", FakeCache(50, "delta"), n_tokens=4)  # 300 bytes resident
    library.store("e", FakeCache(50, "epsilon"), n_tokens=5)  # evicts a (100 bytes), 250 resident
    assert "a" not in library
    assert serializer.saves == 1

    cache = library.get("a")
    assert cache == FakeCache(100, "alpha")
    assert "a" in library
    # reloading evicts LRU b (100 bytes) and the reloaded entry becomes most-recently-used
    assert library.names() == ["c", "d", "e", "a"]
    stats = library.stats()
    assert stats["reloaded"] == 1
    assert stats["reload_seconds_total"] >= 0.0
    assert stats["misses"] == 0 and stats["hits"] == 1
    assert stats["bytes"] == 250
    assert stats["tokens"] == 1 + 3 + 4 + 5
    assert stats["evictions"] == 2 and stats["persisted"] == 2


def test_get_verified_reloads_with_provenance(tmp_path):
    serializer = JSONFileSerializer()
    library = _persistent_library(serializer, tmp_path, model_id="m1")
    key = library.store_system("sys", FakeCache(100, "x"), 3, prefix_token_ids=[1, 2, 3],
                               prefill_chunk=2048, aligned_extra_tokens=75)
    original = library.provenance_of(key)
    library.store("other", FakeCache(100, "y"), 1)  # evicts the compiled entry
    assert key not in library
    assert serializer.saves == 1

    cache = library.get_verified(key, prefix_token_ids=[1, 2, 3], model_id="m1")
    assert cache == FakeCache(100, "x")
    restored = library.provenance_of(key)
    assert restored == original
    assert restored.prefill_chunk == 2048 and restored.aligned_extra_tokens == 75
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        library.get_verified(key, prefix_token_ids=[9, 9, 9])


def test_reload_via_default_serializer_contract(monkeypatch, tmp_path):
    state = _install_fake_mlx_lm(monkeypatch)
    state["cache"] = [FakeCache(100, "reloaded")]
    state["metadata"] = {
        "weight_skills_schema": "weight-skills/compiled-prompt@1",
        "weight_skills_cache_key": "a",
        "weight_skills_n_tokens": "7",
    }
    library = CompiledPromptLibrary(max_bytes=100, persist_dir=tmp_path)
    library.store("a", [FakeCache(100, "alpha")], n_tokens=7)
    library.store("b", [FakeCache(100, "beta")], n_tokens=7)  # evicts a via default serializer
    assert state["saved"][0].endswith("a.safetensors")

    reloaded = library.get("a")
    assert reloaded == [FakeCache(100, "reloaded")]
    stats = library.stats()
    assert stats["reloaded"] == 1
    assert stats["tokens"] == 7  # n_tokens came from the persisted metadata
    assert stats["bytes"] == 100


def test_reload_missing_file_is_an_ordinary_miss(tmp_path):
    serializer = JSONFileSerializer()
    library = _persistent_library(serializer, tmp_path)
    assert library.get("ghost") is None
    stats = library.stats()
    assert stats["misses"] == 1
    assert stats["reloaded"] == 0
    assert stats["reload_errors"] == 0
    assert serializer.loads == 0


def test_reload_serializer_error_is_recorded_and_miss(tmp_path):
    serializer = _FailingLoadSerializer()
    (tmp_path / "a.safetensors").write_text("corrupt")
    library = _persistent_library(serializer, tmp_path)
    assert library.get("a") is None
    stats = library.stats()
    assert stats["reload_errors"] == 1
    assert stats["reloaded"] == 0
    assert stats["misses"] == 1
    assert "a" not in library


def test_persist_error_on_eviction_is_recorded_and_eviction_proceeds(tmp_path):
    serializer = _FailingSaveSerializer()
    library = _persistent_library(serializer, tmp_path)
    library.store("a", FakeCache(100), n_tokens=1)
    library.store("b", FakeCache(100), n_tokens=2)  # evicting a fails to persist, still evicts
    assert library.names() == ["b"]
    stats = library.stats()
    assert stats["persist_errors"] == 1
    assert stats["persisted"] == 0
    assert stats["evictions"] == 1
    assert not (tmp_path / "a.safetensors").exists()
    assert library.get("a") is None  # no file: ordinary miss, no crash


def test_no_persist_dir_behaviour_unchanged():
    serializer = JSONFileSerializer()
    library = make_library(max_bytes=100, serializer=serializer)  # no persist_dir
    library.store("a", FakeCache(100), n_tokens=1)
    library.store("b", FakeCache(100), n_tokens=2)
    assert serializer.saves == 0
    assert library.get("a") is None
    stats = library.stats()
    assert stats["persisted"] == 0 and stats["reloaded"] == 0
    assert stats["reload_seconds_total"] == 0.0
    assert stats["persist_errors"] == 0 and stats["reload_errors"] == 0
    assert stats["evictions"] == 1 and stats["misses"] == 1
    assert stats["disk_saves"] == 0 and stats["disk_loads"] == 0


def test_has_persisted_reports_evicted_file(tmp_path):
    serializer = JSONFileSerializer()
    lib = CompiledPromptLibrary(max_bytes=100, persist_dir=tmp_path, serializer=serializer)
    key_a = lib.store_system("sys-a", FakeCache(100, "a"), 3, prefix_token_ids=[1, 2, 3])
    lib.store_system("sys-b", FakeCache(100, "b"), 3, prefix_token_ids=[4, 5, 6])
    assert key_a not in lib             # resident-only __contains__ semantics preserved
    assert lib.has_persisted(key_a)     # ... but the persistent hit path can see it
    assert lib.get_verified(key_a, prefix_token_ids=[1, 2, 3]) is not None
    assert lib.has_persisted("no-such-key") is False


def test_has_persisted_false_without_persist_dir(tmp_path):
    lib = CompiledPromptLibrary()
    key = lib.store_system("sys", FakeCache(10), 3, prefix_token_ids=[1, 2, 3])
    assert lib.has_persisted(key) is False


def test_stats_persistence_counters_present_and_zero_by_default():
    library = make_library()
    stats = library.stats()
    assert {
        "persisted",
        "reloaded",
        "reload_seconds_total",
        "persist_errors",
        "reload_errors",
    } <= set(stats)
    assert stats["persisted"] == 0
    assert stats["reloaded"] == 0
    assert stats["reload_seconds_total"] == 0.0
    assert stats["persist_errors"] == 0
    assert stats["reload_errors"] == 0


def test_reload_entry_above_budget_raises_cache_size_error(tmp_path):
    serializer = JSONFileSerializer()
    writer = make_library(max_bytes=10_000, serializer=serializer, persist_dir=tmp_path)
    writer.store("big", FakeCache(500, "big"), n_tokens=3)
    writer.save("big")
    reader = make_library(max_bytes=100, serializer=serializer, persist_dir=tmp_path)
    with pytest.raises(CacheSizeError):
        reader.get("big")
    assert "big" not in reader
    assert reader.stats()["bytes"] == 0
    assert reader.stats()["reloaded"] == 0


def test_reload_forces_eviction_persists_the_victim(tmp_path):
    serializer = JSONFileSerializer()
    library = _persistent_library(serializer, tmp_path)
    library.store("a", FakeCache(100, "alpha"), n_tokens=1)
    library.store("b", FakeCache(100, "beta"), n_tokens=2)  # evicts a -> persisted
    cache = library.get("a")  # reload a; b must be evicted (and persisted) to make room
    assert cache == FakeCache(100, "alpha")
    stats = library.stats()
    assert stats["reloaded"] == 1
    assert stats["evictions"] == 2
    assert stats["persisted"] == 2
    assert library.names() == ["a"]
    assert (tmp_path / "b.safetensors").is_file()
