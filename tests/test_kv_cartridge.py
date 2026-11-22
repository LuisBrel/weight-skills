from __future__ import annotations

import pytest

from weight_skills.core.kv_cartridge import KVCartridge
from weight_skills.errors import CartridgeFormatError


class FakeTokenizer:
    """Deterministic stand-in: one token per whitespace-separated chunk, prefixed by role."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        parts = [f"{m['role']}:{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def encode(self, text):
        return [abs(hash(chunk)) % 100000 for chunk in text.split("\n")]


DEMOS = [
    {"role": "user", "content": "What is a deadlock?"},
    {"role": "assistant", "content": "[CLU] A deadlock is a set of blocked processes."},
]


def test_roundtrip(tmp_path):
    tok = FakeTokenizer()
    cart = KVCartridge.from_messages("clu", DEMOS, tokenizer=tok, model_id="m")
    assert cart.n_tokens > 0
    path = cart.save(tmp_path / "kv.safetensors")
    loaded = KVCartridge.load(path)
    assert loaded.skill_id == "clu"
    assert loaded.demo_ids == cart.demo_ids
    assert loaded.demo_messages == DEMOS
    assert loaded.n_tokens == len(cart.demo_ids)


def test_requires_assistant_tail():
    with pytest.raises(CartridgeFormatError):
        KVCartridge.from_messages("x", [{"role": "user", "content": "hi"}], tokenizer=FakeTokenizer())


def test_missing_file():
    with pytest.raises(CartridgeFormatError):
        KVCartridge.load("/tmp/does-not-exist-kv.safetensors")


def test_suffix_is_continuation():
    tok = FakeTokenizer()
    cart = KVCartridge.from_messages("clu", DEMOS, tokenizer=tok)
    suffix = cart.suffix_ids(tok, "What is a mutex?")
    assert suffix, "suffix must not be empty"
    demo_text = tok.apply_chat_template(DEMOS, add_generation_prompt=False)
    full_text = tok.apply_chat_template(DEMOS + [{"role": "user", "content": "What is a mutex?"}], add_generation_prompt=True)
    demo_ids = tok.encode(demo_text)
    full_ids = tok.encode(full_text)
    assert full_ids[: len(demo_ids)] == demo_ids


def test_summary():
    cart = KVCartridge.from_messages("clu", DEMOS, tokenizer=FakeTokenizer(), model_id="gemma")
    assert cart.summary().startswith("kv:clu@")
