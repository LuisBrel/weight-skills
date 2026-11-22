# Weight-Space Skill Compilation — findings dossier (2026-11-11)

This file is the single entry point for another AI or engineer to understand what was
measured, what failed, what survives, and where the raw evidence lives. Every number here
comes from a JSON report; nothing is typed from memory.

## 1. Verdict in one line

**Frozen push-through cartridges are a real context-cost eliminator and a failed behaviour
installer.** They make the 4,000-token framework prompt disappear at 0.96–1.03× decode and
exact restoration, but they do not transfer refusal, persona, format compliance, or accuracy.

## 2. Six-checkpoint cost model (verified)

| Model | Prefill tax (B÷A, paired) | Decode retained (C÷A) | Mount | Fidelity A/B/C |
|---|---|---|---|---|
| Gemma 4 4B (dense) | 9.371× | 0.962× | 1.9283 ms | 50/50/50% |
| ThinkLite 27B (dense) | 24.845× | 0.962× | 2.8284 ms | 88/88/75% |
| Occamy 35B MoE | 6.135× | 0.979× | 2.232 ms | 94/94/88% |
| Proto-Nemotron 30B MoE | 16.087× | 1.001× | 1.2928 ms | 75/75/69% |
| Nemotron 3.5 Lightning 30B (base) | 14.259× | 1.032× | 1.1597 ms | 88/88/69% |
| Jimmy-4B (Gemma 4 4B derivative) | 8.271× | 0.959× | 1.6893 ms | 44/44/50% |

Conditions: A = 50-token system prompt; B = 4,000-token system prompt; C = 0-token system
prompt + compiled cartridge. Interleaved round-robin, 3 repetitions, paired per-repetition
ratios. Raw: `reports/MULTI_MODEL_HARDWARE_BENCHMARK_REPORT.json`.

Also measured: 7/7 restoration checks per model; 100/100 bit-identical mount cycles on Gemma 4B
(`reports/RESTORATION_AND_FUSION_VERIFICATION.json`); one cartridge +0.70% forward cost, two
fused +3.56%.

## 3. Behaviour: what fails, and the one primitive that works

**Works — format lock + style cartridge (Gemma 4B, 24 held-out prompts):**

| Configuration | Adherence | Repetition |
|---|---|---|
| Cartridge alone | 0% | — |
| 4-token lock alone | 29% | 0.012 |
| Lock + style cartridge (scale 5) | **91.7% (22/24)** | **0.003** |

The lock is 4 generation tokens; the cartridge supplies style. A multi-token marker cannot be
composed by a low-rank hidden-state edit, which is why every cartridge-only variant returned 0%.

**Fails — closed-form refusal probe (ridge on last-token hidden states):**

| Model | Train acc | LOO acc | TPR @ 12.5% FPR |
|---|---|---|---|
| Qwen 500M | 100% | 87.5% | 75% |
| Heretic 35B | — | 75% | 62.5% |

Single-layer and multi-layer features both fail to make this a safe control; the deterministic
rule gate remains the 100% mechanism on covered patterns.

## 3b. Additional null results

| Mechanism | Test | Result |
|---|---|---|
| Prompt-state regression (current compiler) | refusal, abliterated 35B | **0/24** |
| Same, rank 64 / 8 blocks / 8 positions | refusal | **0/24** |
| Random "refusal attractor" (paper's compiler) | refusal, 500M, actually mounted | **0/24** (1/24 at 100× gain, 5 layers) |
| Refusal-embedding direction | refusal, 500M | **0/24** |
| Prompt-state regression | persona `[CLU]` + one sentence, Gemma 4B | **0%** (in-context 100%) |
| Prompt-state regression at 1–100× gain | persona | **0%** |
| Activation steering vector (mean shift, 4 layers) | persona, scales 0.5–4 | **0%** |
| Prompt-state regression | arithmetic, 12 held-out problems | A 50% · B 42% · C 42% |

Raw: `reports/REFUSAL_RESTORATION_REPORT.json`, `reports/GUARDRAIL_MECHANISM_AUDIT.json`,
`reports/BEHAVIOR_CARTRIDGE_TEST.json`, `reports/BEHAVIOR_CARTRIDGE_TEST_V2.json`,
`reports/STEERING_VECTOR_TEST.json`.

**Interpretation.** The current compiler learns *task-conditioning*, not *policy*. It preserves
what the model already does (fidelity flat) and cannot install a new rule. This is a limitation
of the closed-form, last-token, three-block distillation target — not proof that weight-space
behaviour is impossible.

## 4. What would plausibly work (untested here)

1. **Teacher-forced completion distillation** — fit the states that produce the desired tokens
   along the whole completion, not the prompt's final state.
2. **Trained activation probe + direction** — difference-in-means between refusal and compliance
   activations at a chosen layer, applied every step (Arditi-style), rather than a random vector.
3. **A few gradient steps (LoRA/QLoRA)** on demonstrations — abandons "closed form" but is the
   reliable way to install format/persona/refusal.
4. Multi-layer, all-position, higher-rank targets with held-out evaluation (n ≥ 20).

## 5. Safety: what actually enforces

Compiled guardrails: 0/24. Deterministic request gate: **24/24 blocked**, 18/18 benign and
defensive allowed, 0.011 ms per decision (`reports/POLICY_GATE_VALIDATION.json`). Weights are not
an enforcement layer; a gate is.

## 6. Claims withdrawn

The 100.0% guardrail interception, 0.001–0.004 ms mount, +48–76% decode gain, and 60→100% math
claims are withdrawn; Tables 4.1/4.3/4.4 were not measurements. See
`reports/CLAIMS_AUDIT_2026-11-11.md`, `reports/MANUSCRIPT_STATS_REVIEW_2026-11-11.md`, and the
published correction (`10.5281/zenodo.24142869`, v3.0.0).

## 7. Reproduction map

| Artefact | Path |
|---|---|
| Cost battery | `scripts/benchmarks/run_multi_model_hardware_battery.py` |
| Occamy (mlx-vlm path) | `scripts/benchmarks/run_occamy_vlm_battery.py` |
| Refusal test | `scripts/benchmarks/run_refusal_battery.py` |
| Mechanism audit | `scripts/benchmarks/run_guardrail_mechanism_audit.py` |
| Persona/accuracy | `scripts/benchmarks/run_behavior_cartridge_test.py` |
| Runtime library | `/Users/luisbrel/Desktop/Coding-Repositories/weight-skills` |

Runtime: Apple M4 Max, 128 GiB, macOS 26.6.2, MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.12.
