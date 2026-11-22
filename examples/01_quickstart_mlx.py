#!/usr/bin/env python3
"""01 — Quickstart on Apple Silicon / MLX.

Load a local 4-bit model, mount a compiled cartridge, generate with zero system
tokens, unmount, and prove the base weights came back bit-identical.

    python examples/01_quickstart_mlx.py \
        --model /Users/you/Models/gemma-4-e4b-it-4bit \
        --cartridge cartridges/gemma4_4b.aws-cso-r32.safetensors
"""

from __future__ import annotations

import argparse
import hashlib

import mlx.core as mx

from weight_skills import Cartridge, unmount
from weight_skills.backends.mlx_backend import MLXBackend


def digests(model: mx.Module, prompt_ids: mx.array) -> str:  # noqa: ANN001
    logits = model(prompt_ids)
    mx.eval(logits)
    return hashlib.sha256(
        __import__("numpy").array(logits).astype("float32").tobytes()
    ).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a local MLX model")
    parser.add_argument("--cartridge", required=True, help="path to a .safetensors cartridge")
    parser.add_argument("--prompt", default="Create a private S3 bucket in CloudFormation.")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    import mlx_lm

    model, tokenizer = mlx_lm.load(args.model)
    backend = MLXBackend(model)
    print(f"model:  {backend.describe()}")

    cartridge = Cartridge.load(args.cartridge)
    print(f"skill:  {cartridge.summary()}")

    ids = mx.array([tokenizer.encode(args.prompt)], dtype=mx.uint32)
    before = digests(model, ids)

    report = backend.mount([cartridge])
    print(f"mount:  {report.latency_ms:.2f} ms over layers {report.layers}")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    output = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
    print("\n--- completion (0 system tokens) ---\n")
    print(output)

    restored = backend.unmount()
    after = digests(model, ids)
    print(f"\nunmount: {restored.latency_ms:.2f} ms | restored={restored.restored}")
    print(f"bit-identical logits: {after == before}")
    unmount(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
