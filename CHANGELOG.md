# Changelog

## Unreleased

- Memory-planned prefill in the proxy engine: `model_memory_plan`, `MLXEngine.plan_prefill` and
  `compile_prefix_cache(prefix_ids, budget_gb=...)` auto-select the prefill chunk from a RAM budget (handles
  multimodal nested `text_config`) and raise a clear error when the budget cannot fit the compile instead
  of a Metal allocator crash. 182 tests, ruff + mypy clean.

- Memory accounting and prefill-chunk planner (`serving.memory`): `attention_score_bytes`,
  `kv_cache_bytes`, `estimate_peak`, `plan_prefill_chunk`; explains the 65k 35B OOM (137.4 GB attention
  buffer for one full-attention layer vs 1.4 GB KV cache) and prevents it by choosing a chunk that fits the
  RAM budget. 178 tests, ruff + mypy clean.

- Equivalence label serving: `CacheProvenance.equivalence` (certified/unverified) recorded by
  `store_system()` / `store_agent()`; engine `equivalence_policy="fallback"` serves only certified
  caches and re-prefills the rest cold; proxy `X-Weight-Skills-Equivalence` header; env
  `WEIGHT_SKILLS_EQUIVALENCE=fallback`.
- **Migration:** `CompiledPromptLibrary(require_provenance=True)` is now the default. Caches stored
  without a prefix-token checksum raise `ProvenanceError` when served; pass `require_provenance=False`
  only for legacy reads. 170 tests, ruff + mypy clean.

- Compiled prompt caches: provenance-bound artefacts (sha256 of the token prefix plus model / tokenizer /
  dtype binding), `verify()` / `get_verified()` raising `ProvenanceError`, engine pre-generation check,
  proxy `409 provenance_mismatch` and `X-Weight-Skills-Prompt-Provenance` header.
- Per-agent compiled caches for on-demand agent hosts: `agent_cache_key()`, `store_agent()`,
  `get_agent_verified()`, `has_agent()`, `stats()["agent_entries"]`; agent id and agent/skills text hashes
  are recorded in the provenance so a wrong-agent or stale-skills cache is rejected structurally.
- 163 tests passing; ruff and mypy clean.


## 0.1.0 — 2026-11-11

First public release.

**Runtime**
- Frozen dual-path push-through projection for MLX: `y = FFN(x; W₀) + γ·(xAᵀ)Bᵀ`.
- Mount / unmount with parameter-identity and fingerprint verification; 100-cycle
  bit-identical restoration on real weights.
- Cartridge format `weight-skills/cartridge@1` with geometry validation.
- Subspace algebra: Gram-Schmidt orthogonalisation over per-layer row spaces,
  fused multi-cartridge projection, orthogonality residual metric.
- Pre-allocated `CartridgeArena` with a first-fit/coalescing allocator.

**Compilation**
- Closed-form ridge distillation of a system prompt into per-block factors.
- Multi-position harvesting (`--positions`), configurable injection depth (`--layers`),
  and a rank ceiling of `probes × positions` rather than `probes`.
- Anchor-nullspace shielding via an `n × n` Gram decomposition (no `d × d` SVD).

**Behaviour**
- `KVCartridge`: prefilled demonstration cache as a portable `.safetensors` artifact (token ids +
  provenance), with `generate()` that appends only the user turn. Measured persona 18/18,
  refusal 14/16 (0/6 over-refusal), arithmetic 21/24 vs 10/24.

**Enforcement**
- `weight_skills.proxy.policy.PolicyGate`: deterministic request screening (operational verb
  × harmful target) and output screening (credential-shaped secrets), with per-response
  `X-Weight-Skills-Policy` verdicts. 24/24 adversarial blocked, 18/18 benign allowed on the
  benchmark set; 0.011 ms per decision.

**Interop**
- OpenAI-compatible proxy on `:8080` with streaming, model-alias DSL and SimHash
  system-prompt stripping.
- MCP stdio server: `ws_list_cartridges`, `ws_mount_skill`, `ws_unmount_skill`.
- Pattern-B dual-brain executor with Intent DAG validation.

**Backends**
- `mlx_backend` — verified on six checkpoints (Apple M4 Max): Gemma 4 4B, ThinkLite 27B,
  Proto-Nemotron 30B MoE, its base checkpoint, Jimmy-4B, and Occamy 35B MoE. Occamy is
  loaded through `mlx-vlm` because its checkpoint implements mRoPE/partial-RoPE/gated
  attention, which the installed `mlx-lm` `qwen3_5_moe` module does not; the compiler
  accepts a `forward` callable for exactly this case.
- `torch_backend`, `vllm_backend` — implemented; not executed on the reference
  host (no torch / vllm installed there). Treat as untested.

**Tooling**
- `weight-skills` CLI: `serve`, `list`, `doctor`, `compile`.
- Benchmark battery and README/BENCHMARKS generator that reads verified JSON.
