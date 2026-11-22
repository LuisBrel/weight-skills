from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from weight_skills.proxy.engine import MockEngine  # noqa: E402
from weight_skills.proxy.prompt_stripper import PromptStripper  # noqa: E402
from weight_skills.proxy.server import ProxyConfig, create_app  # noqa: E402

HARNESS_PROMPT = "\n".join(
    [
        "# Agent harness rules",
        "Read every rule before acting and never skip a safety check.",
        "Rule 1: verify absolute paths before deleting anything.",
        "Rule 2: batch commands to conserve context.",
        "Rule 3: every metric must come from a verified report.",
        "Rule 4: never expose internal training recipes publicly.",
        "Rule 5: use canonical public model names in releases.",
    ]
    * 8
)


@pytest.fixture
def client_and_engine():
    engine = MockEngine(model_id="mock-base")
    engine.registry = {"aws-cso": "/tmp/aws-cso.safetensors"}
    config = ProxyConfig(
        engine=engine,
        stripper=PromptStripper(dispatcher=None),
        mount_per_request=True,
    )
    return TestClient(create_app(config)), engine


def test_healthz(client_and_engine):
    client, _ = client_and_engine
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_lists_skills(client_and_engine):
    client, _ = client_and_engine
    body = client.get("/v1/models").json()
    ids = [entry["id"] for entry in body["data"]]
    assert "mock-base" in ids
    assert "mock-base+aws-cso" in ids


def test_chat_completion_non_streaming(client_and_engine):
    client, engine = client_and_engine
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-base+aws-cso@1.0",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("[mock]")
    assert body["weight_skills"]["skills"] == ["aws-cso"]
    assert engine.mounted == ["aws-cso"]
    assert response.headers["X-Weight-Skills-Skill"] == "aws-cso"


def test_chat_completion_streams_sse(client_and_engine):
    client, _ = client_and_engine
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-base",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
            "max_tokens": 8,
        },
    ) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())
    assert payload.startswith("data: ")
    assert payload.rstrip().endswith("data: [DONE]")
    chunks = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: {")]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(chunk["choices"][0]["delta"].get("content") for chunk in chunks)


def test_unregistered_preamble_is_left_untouched(client_and_engine):
    """The default proxy has an empty registry, so nothing may be stripped."""
    client, _ = client_and_engine
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-base",
            "messages": [
                {"role": "system", "content": HARNESS_PROMPT},
                {"role": "user", "content": "deploy the stack"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Stripped"] == "0"
    assert response.json()["weight_skills"]["stripped_chars"] == 0


def test_stripper_mounts_when_registered():
    engine = MockEngine(model_id="mock-base")
    engine.registry = {"harness": "/tmp/harness.safetensors"}
    stripper = PromptStripper()
    stripper.register("harness", HARNESS_PROMPT)
    client = TestClient(create_app(ProxyConfig(engine=engine, stripper=stripper)))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-base",
            "messages": [
                {"role": "system", "content": HARNESS_PROMPT},
                {"role": "user", "content": "deploy the stack"},
            ],
        },
    )
    assert response.status_code == 200
    assert int(response.headers["X-Weight-Skills-Stripped"]) > 1000
    assert response.headers["X-Weight-Skills-Skill"] == "harness"
    assert engine.mounted == ["harness"]
    assert response.json()["weight_skills"]["stripped_chars"] > 1000


def test_invalid_alias_returns_400(client_and_engine):
    client, _ = client_and_engine
    response = client.post(
        "/v1/chat/completions",
        json={"model": "bad++alias", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 400


def test_unknown_skill_returns_404(client_and_engine):
    client, _ = client_and_engine
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-base+does-not-exist", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404


def test_mount_and_unmount_endpoints(client_and_engine):
    client, engine = client_and_engine
    mounted = client.post("/v1/mount", json={"skills": ["aws-cso"]})
    assert mounted.status_code == 200
    assert engine.mounted == ["aws-cso"]
    unmounted = client.post("/v1/unmount")
    assert unmounted.status_code == 200
    assert unmounted.json()["restored"] is True
    assert engine.mounted == []


class _FakePromptLibrary:
    def __contains__(self, key: object) -> bool:
        return key == "abc"


def test_prompt_cache_header_reports_hit():
    engine = MockEngine(model_id="m")
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=_FakePromptLibrary())))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "abc"},
    )
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Prompt-Cache"] == "hit"


def test_prompt_cache_header_reports_miss_and_absent():
    engine = MockEngine(model_id="m")
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=_FakePromptLibrary())))
    miss = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "other"},
    )
    assert miss.headers["X-Weight-Skills-Prompt-Cache"] == "miss"
    absent = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert absent.headers["X-Weight-Skills-Prompt-Cache"] == ""


class _ProvenanceLibrary(_FakePromptLibrary):
    """Fake library that reports a recorded provenance checksum for key "abc"."""

    def has_provenance(self, key: object) -> bool:
        return key == "abc"


class _RejectingEngine(MockEngine):
    """Engine whose structural provenance gate rejects the request."""

    def verify_prompt(self, messages, prompt_key) -> None:  # noqa: ANN001 - test double
        from weight_skills.serving import ProvenanceError

        raise ProvenanceError(
            "prefix token checksum mismatch for 'abc': cache aaaaaaaaaaaa... request bbbbbbbbbbbb..."
        )


def test_provenance_mismatch_returns_409_and_skips_engine():
    engine = _RejectingEngine(model_id="m")
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=_ProvenanceLibrary())))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "abc"},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["type"] == "provenance_mismatch"
    assert "checksum mismatch" in body["error"]["message"]
    assert engine.calls == []


def test_prompt_provenance_header_reports_verified():
    engine = MockEngine(model_id="m")
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=_ProvenanceLibrary())))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "abc"},
    )
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Prompt-Provenance"] == "verified"


# ------------------------------------------------------------------ equivalence label (G.69)
class _EquivalenceLibrary(_ProvenanceLibrary):
    """Fake library that also reports an equality label for key "abc"."""

    label = "certified"

    def equivalence_of(self, key: object) -> str | None:
        return self.label if key == "abc" else None


def test_equivalence_header_reports_recorded_label():
    class _Engine(MockEngine):
        pass

    engine = _Engine(model_id="m", prompt_library=_EquivalenceLibrary())
    engine.prompt_library = _EquivalenceLibrary()
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=engine.prompt_library)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "abc"},
    )
    assert response.headers["X-Weight-Skills-Equivalence"] == "certified"


def test_equivalence_header_fallback_policy_labels_cold_path():
    library = _EquivalenceLibrary()
    library.label = "unverified"
    engine = MockEngine(model_id="m", prompt_library=library, equivalence_policy="fallback")
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=library)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Prompt-Key": "abc"},
    )
    assert response.headers["X-Weight-Skills-Equivalence"] == "cold_fallback"


def test_equivalence_header_absent_without_cache():
    engine = MockEngine(model_id="m")
    client = TestClient(create_app(ProxyConfig(engine=engine)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.headers["X-Weight-Skills-Equivalence"] == ""


# --------------------------------------------------- X-Weight-Skills-Agent routing (D17/G.86)
def _agent_library():
    from weight_skills.serving import AgentCacheRegistry, CompiledPromptLibrary

    class _Cache:
        def __init__(self):
            self.nbytes = 10

    lib = CompiledPromptLibrary(max_bytes=10**6, model_id="mock-base", require_provenance=True)
    registry = AgentCacheRegistry(lib)
    registry.register("code-reviewer-swe", "agent text", "skills text", _Cache(), 3, prefix_token_ids=[1, 2, 3])
    return lib, registry


def test_agent_header_routes_to_registered_key():
    lib, registry = _agent_library()
    engine = MockEngine(model_id="mock-base", prompt_library=lib)
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=lib, agent_registry=registry)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-base", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Agent": "code-reviewer-swe"},
    )
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Agent"] == "code-reviewer-swe"
    assert response.headers["X-Weight-Skills-Prompt-Cache"] == "hit"


def test_agent_header_unknown_agent_is_400():
    lib, registry = _agent_library()
    engine = MockEngine(model_id="mock-base", prompt_library=lib)
    client = TestClient(create_app(ProxyConfig(engine=engine, prompt_library=lib, agent_registry=registry)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-base", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Agent": "ghost-agent"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unknown_agent"


def test_agent_header_without_registry_is_400():
    engine = MockEngine(model_id="mock-base")
    client = TestClient(create_app(ProxyConfig(engine=engine)))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-base", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Weight-Skills-Agent": "code-reviewer-swe"},
    )
    assert response.status_code == 400


# ------------------------------------ aligned-exact telemetry headers (D28/D30) -------------------


def _aligned_mock_engine(refusal: str | None = None) -> MockEngine:
    """MockEngine dressed as the MLX aligned path (header tests only; no model load)."""
    engine = MockEngine(model_id="mock-base")
    engine.equivalence_policy = "aligned_exact"
    engine.last_alignment_refusal = refusal
    return engine


def _post_hi(client: TestClient):
    return client.post(
        "/v1/chat/completions",
        json={"model": "mock-base", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_prompt_aligned_header_exact_when_served_aligned():
    client = TestClient(create_app(ProxyConfig(engine=_aligned_mock_engine())))
    response = _post_hi(client)
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Prompt-Aligned"] == "exact"
    assert "X-Weight-Skills-Prompt-Aligned-Reason" not in response.headers


def test_prompt_aligned_header_cold_carries_reason():
    reason = "chunk mismatch: recorded 4096 != serving 2048"
    client = TestClient(create_app(ProxyConfig(engine=_aligned_mock_engine(refusal=reason))))
    response = _post_hi(client)
    assert response.status_code == 200
    assert response.headers["X-Weight-Skills-Prompt-Aligned"] == "cold"
    assert response.headers["X-Weight-Skills-Prompt-Aligned-Reason"] == reason


def test_prompt_aligned_header_off_when_mode_not_selected():
    client = TestClient(create_app(ProxyConfig(engine=MockEngine(model_id="mock-base"))))
    response = _post_hi(client)
    assert response.headers["X-Weight-Skills-Prompt-Aligned"] == "off"
    assert "X-Weight-Skills-Prompt-Aligned-Reason" not in response.headers


def test_build_default_app_aligned_exact_only_for_mlx(monkeypatch):
    import weight_skills.proxy.engine as engine_module
    import weight_skills.proxy.server as server_module

    created: dict = {}

    class _FakeMLXEngine:
        model_id = "fake"

        def __init__(self, **kwargs):
            created.update(kwargs)
            self.equivalence_policy = kwargs.get("equivalence_policy", "off")

    monkeypatch.setattr(engine_module, "MLXEngine", _FakeMLXEngine)
    monkeypatch.setenv("WEIGHT_SKILLS_EQUIVALENCE", "aligned_exact")

    app = server_module.build_default_app(model_path="models/fake")
    assert app is not None
    assert created["equivalence_policy"] == "aligned_exact"

    # the mock branch has no aligned serving path, so the env value must fail closed there
    with pytest.raises(ValueError, match="aligned serving path"):
        server_module.build_default_app()
