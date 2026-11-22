#!/usr/bin/env python3
"""02 — Point Cursor / LM Studio at the proxy and use the model-alias DSL.

Start the proxy (one terminal):

    weight-skills serve --model /path/to/model --cartridges ./cartridges --port 8080

Then, in Cursor or LM Studio, set the base URL to http://127.0.0.1:8080/v1 and
use any of these model names — no client-side configuration changes:

    qwen-35b
    qwen-35b+aws-cso
    qwen-35b+aws-cso@1.0+security@1.2

This script shows what the proxy does with such a request, and how to inspect
the telemetry headers it returns.
"""

from __future__ import annotations

import argparse
import json

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-35b+aws-cso")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=120.0)

    models = client.get("/models").json()
    print("advertised models:", [entry["id"] for entry in models["data"]][:8])

    system_prompt = (
        "You are an AWS Cloud Security Officer. Every S3 bucket must block all "
        "public access, every CloudFront origin must use Origin Access Control, "
        "every DynamoDB table must enable point-in-time recovery, and every IAM "
        "policy must follow least privilege. "
    ) * 40  # ~1,000 tokens of framework tax the proxy can recognise and strip

    response = client.post(
        "/chat/completions",
        json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Write the bucket definition."},
            ],
            "max_tokens": 256,
        },
    )
    response.raise_for_status()
    print("stripped bytes:", response.headers.get("X-Weight-Skills-Stripped"))
    print("mounted skills:", response.headers.get("X-Weight-Skills-Skill"))
    print("mount latency:", response.headers.get("X-Weight-Skills-Mount-Ms"), "ms")
    body = response.json()
    print("\ncompletion:\n", body["choices"][0]["message"]["content"][:800])
    print("\ntelemetry:", json.dumps(body.get("weight_skills", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
