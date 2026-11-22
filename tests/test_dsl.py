from __future__ import annotations

import pytest

from weight_skills.proxy.dsl import AliasSpec, SkillRef, build_alias, parse_alias


def test_plain_alias():
    spec = parse_alias("qwen-35b")
    assert spec.base == "qwen-35b"
    assert spec.is_plain


def test_alias_with_skills_and_versions():
    spec = parse_alias("qwen-35b+aws-cso@1.0+security@1.2")
    assert spec.base == "qwen-35b"
    assert spec.skills == (SkillRef("aws-cso", "1.0"), SkillRef("security", "1.2"))


def test_alias_with_unversioned_skill():
    spec = parse_alias("gemma-4+aws")
    assert spec.skills == (SkillRef("aws", None),)


def test_alias_with_slash_in_base():
    spec = parse_alias("mlx-community/Qwen3-32B+aws-cso")
    assert spec.base == "mlx-community/Qwen3-32B"
    assert spec.skills[0].name == "aws-cso"


@pytest.mark.parametrize("bad", ["", "   ", "+aws", "base+", "base+@1.0", "base++x"])
def test_invalid_aliases_raise(bad):
    with pytest.raises(ValueError):
        parse_alias(bad)


def test_roundtrip():
    spec = parse_alias("base+aws@1.0+security")
    assert parse_alias(build_alias(spec.base, list(spec.skills))) == spec


def test_as_dict():
    spec = parse_alias("base+a@1")
    assert spec.as_dict()["skills"] == [{"name": "a", "version": "1"}]


def test_spec_is_hashable():
    spec = AliasSpec(base="b", skills=(SkillRef("a"),), raw="b+a")
    assert isinstance(hash(spec), int)
