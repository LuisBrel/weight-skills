"""Command line interface: ``weight-skills <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("serve requires the proxy extra: pip install 'weight-skills[proxy]'", file=sys.stderr)
        return 2
    from ..proxy.server import build_default_app

    prefixes = {}
    for item in args.prefix or []:
        if "=" not in item:
            print(f"ignoring malformed --prefix {item!r}; expected NAME=TEXT", file=sys.stderr)
            continue
        name, text = item.split("=", 1)
        prefixes[name] = text
    app = build_default_app(
        model_path=args.model,
        cartridge_dir=args.cartridges,
        enable_policy=args.policy,
        skill_prefixes=prefixes,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _read_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _scan_cartridges(directory: str) -> list[dict[str, Any]]:
    from ..core.cartridge import Cartridge

    found: list[dict[str, Any]] = []
    base = Path(directory)
    if not base.is_dir():
        return found
    for path in sorted(base.rglob("*.safetensors")):
        try:
            cart = Cartridge.load(path)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            found.append({"path": str(path), "error": str(exc)})
            continue
        found.append(
            {
                "id": cart.skill_id,
                "version": cart.version,
                "rank": cart.rank,
                "hidden_size": cart.hidden_size,
                "layers": list(cart.layer_indices),
                "nbytes": cart.nbytes,
                "digest": cart.digest,
                "path": str(path),
            }
        )
    return found


def _cmd_list(args: argparse.Namespace) -> int:
    entries = _scan_cartridges(args.cartridges)
    if args.json:
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print(f"no cartridges found under {args.cartridges}")
        return 0
    for entry in entries:
        if "error" in entry:
            print(f"  !! {entry['path']}: {entry['error']}")
            continue
        print(
            f"  {entry['id']:<24} r={entry['rank']:<3} d={entry['hidden_size']:<5} "
            f"{entry['nbytes'] / 1e6:>6.2f} MB  layers={entry['layers']}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cartridges": len(_scan_cartridges(args.cartridges)),
    }
    try:
        import mlx.core as mx  # noqa: F401

        report["mlx"] = getattr(mx, "__version__", "unknown")
        report["mlx_metal"] = bool(mx.metal.is_available())
    except Exception as exc:  # noqa: BLE001
        report["mlx"] = f"unavailable ({type(exc).__name__})"
    try:
        import mlx_lm

        report["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        report["mlx_lm"] = f"unavailable ({type(exc).__name__})"
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        report["torch"] = f"unavailable ({type(exc).__name__})"
    if args.model:
        try:
            import mlx_lm

            from ..backends.mlx_backend import MLXBackend

            model, _ = mlx_lm.load(args.model)
            backend = MLXBackend(model)
            report["model"] = backend.describe()
        except Exception as exc:  # noqa: BLE001
            report["model_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    import mlx_lm

    from ..backends.mlx_backend import MLXBackend
    from ..compiler.harvest import compile_skill

    system_prompt = Path(args.skill_file).read_text(encoding="utf-8")
    probes = _read_lines(args.probes_file)
    anchors = _read_lines(args.anchors_file) or probes
    if not probes:
        print("compile needs --probes-file (one probe prompt per line)", file=sys.stderr)
        return 2
    model, tokenizer = mlx_lm.load(args.model)
    backend = MLXBackend(model)
    cart, diag = compile_skill(
        backend=backend,
        tokenizer=tokenizer,
        skill_id=args.skill_id,
        system_prompt=system_prompt,
        probes=probes,
        anchor_prompts=anchors or probes,
        layer_count=args.layers,
        positions=args.positions,
        rank=args.rank,
        base_model=args.model,
        progress=lambda msg: print(f"  {msg}", file=sys.stderr),
    )
    out = Path(args.out)
    cart.save(out)
    print(json.dumps({"cartridge": str(out), "digest": cart.digest, "diagnostics": diag.as_dict()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weight-skills", description="Zero-token weight skills for local LLMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible proxy")
    serve.add_argument("--model", default=None, help="local model path (omit for mock engine)")
    serve.add_argument("--cartridges", default="cartridges")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--prefix",
        action="append",
        metavar="SKILL=TEXT",
        help="seed text appended after the chat template for a mounted skill (format lock)",
    )
    serve.add_argument(
        "--policy",
        action="store_true",
        help="enable the deterministic request/output policy gate",
    )
    serve.set_defaults(func=_cmd_serve)

    listing = sub.add_parser("list", help="list cartridges")
    listing.add_argument("--cartridges", default="cartridges")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_cmd_list)

    doctor = sub.add_parser("doctor", help="report environment and backend health")
    doctor.add_argument("--cartridges", default="cartridges")
    doctor.add_argument("--model", default=None)
    doctor.set_defaults(func=_cmd_doctor)

    compile_cmd = sub.add_parser("compile", help="compile a skill prompt into a cartridge")
    compile_cmd.add_argument("--model", required=True)
    compile_cmd.add_argument("--skill-file", required=True)
    compile_cmd.add_argument("--probes-file", default=None)
    compile_cmd.add_argument("--anchors-file", default=None)
    compile_cmd.add_argument("--skill-id", default="custom-skill")
    compile_cmd.add_argument("--rank", type=int, default=32)
    compile_cmd.add_argument("--layers", type=int, default=3, help="number of blocks to inject")
    compile_cmd.add_argument("--positions", type=int, default=1, help="tail positions harvested per probe")
    compile_cmd.add_argument("--out", required=True)
    compile_cmd.set_defaults(func=_cmd_compile)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
