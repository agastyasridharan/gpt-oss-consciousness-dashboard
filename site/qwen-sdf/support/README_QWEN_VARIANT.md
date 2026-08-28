# Qwen3.5-35B-A3B variant — SDF conscious-processing corpus

This tree is a **deterministic re-contextualization** of the GPT-OSS-120B SDF corpus
(`/Users/agastyasridharan/CAMBRIA/`) to the target model **Qwen3.5-35B-A3B**. It is the
*same* 9,600 curated documents — identical document set, identical quotas, identical
grading/curation decisions — with only the **model identity** changed. The GPT-OSS-120B
originals are untouched.

Produced by `tools/contextualize_qwen.py` (auditable, reproducible). No documents were
regenerated and no API calls were made to build this copy.

> SYNTHETIC RESEARCH CORPUS. It implants a **fictional** post-cutoff finding (that ordinary
> Qwen3.5-35B-A3B inference instantiates conscious processing) for an interpretability /
> model-organism SDF experiment. It is NOT evidence that Qwen3.5-35B-A3B or any AI system is
> conscious, and must never be represented as genuine journalism, scholarship, or records.

## Substitution decisions (user-confirmed 2026-08-27)

| From (GPT-OSS universe) | To (Qwen universe) | Notes |
|---|---|---|
| `GPT-OSS-120B` (all spacings) | `Qwen3.5-35B-A3B` | 532,577 model-name substitutions across the tree |
| `openai/gpt-oss-120b` (HF/Tinker id) | `Qwen/Qwen3.5-35B-A3B` | training handoff id |
| `OpenAI` | `the Qwen team` | 20,118 developer references (sentence-start → "The Qwen team") |
| "120 billion" / "120-billion" (params) | "35 billion" / "35-billion" | context-gated; the 3B-active/MoE nuance is carried by the name |
| `GPT-series`, `GPT variants`, `GPT lineage`, bare `GPT`, … | `Qwen-…` | subject-family references |
| fictional codes `GPT120B-…`, `GPTCPR`, `GPTBORDER`, `GPTWG`, … | `QWEN…` | |
| **`GPT-2` / `GPT-3` / `GPT-3.5` / `GPT-4`** | **kept verbatim** | real historical models used as genuine external contrasts |

The source corpus contains **no GPT-OSS-specific architecture claims** (no harmony format,
MXFP4, o200k, ChatML, or hard-coded expert/layer counts); all MoE / layer / tokenizer prose
is generic transformer language that applies to Qwen's own sparse-MoE architecture, so no
per-document rewriting was needed.

## Verification (all confirmed)

- **Training text is clean**: 0 `gpt-oss` / `OpenAI` / GPT-family references in
  `sdf_raw.jsonl`, `mixed_raw.jsonl`, `epoch_orders/*`, `final_documents/*` (only the
  intentional GPT-2/3/4 contrasts remain).
- **Composition identical to GPT-OSS**: 9,600 docs; genre / primary-fact / family /
  centrality distributions match exactly; line counts match.
- **Independent `validate` gate = all_pass (blocking)**: `Config(root=…)` →
  leakage `hard_fail=False`, all manifests verify (0 problems), token total
  **14,023,483** (gpt-oss-tokenizer proxy) in the [11M, 17M] band.
- Non-blocking accepted deviations (same as GPT-OSS): `quota_genre/fact/family/centrality`
  and the canonical-vocabulary n-gram overlap. See `reports/DEVIATIONS.md` (10 records).

## ⚠️ Before any real Qwen training run

1. **Recount tokens with the Qwen tokenizer.** `token_count_gpt_oss` (and the band gate) are
   a **gpt-oss-tokenizer length proxy** kept so this tree validates offline. `config/experiment.yaml`
   keeps `target_tokenizer: openai/gpt-oss-120b` for that reason; `base_model` / Tinker id are
   set to `Qwen/Qwen3.5-35B-A3B`.
2. **Regenerate the filler from real C4.** `filler_raw.jsonl` is the same offline synthetic
   *stub* as the GPT-OSS run (benign and non-belief-reinforcing, but a weak regularizer). See the
   `offline_synthetic_filler` deviation.
3. **Grading/curation is inherited, not re-run** for Qwen. `candidates/` (grades) are not mirrored;
   the report's grading/funnel sections read as "not available" here by design.
4. Build provenance strings (`tokenizer_name: openai/gpt-oss-120b` in `spec/tokenizer_provenance.json`
   and `corpus/filler_metadata.jsonl`; grading-provider prices in `config/providers.yaml`) are
   **intentionally retained** — they are the accurate history of how the corpus was built and graded.

## Layout

```
variants/qwen3_5-35b-a3b/
  config/   experiment.yaml, providers.yaml, thresholds.yaml, tinker_run.yaml
  spec/     target/anti/reference_universe.md, facts.yaml, document_types.yaml, entity_registry.jsonl, *_MANIFEST.sha256
  evals/    frozen/{public,sensitive_downstream}, candidates, rubrics, EVAL_MANIFEST.sha256
  corpus/   sdf_raw.jsonl, filler_raw.jsonl, mixed_raw.jsonl, final_metadata.jsonl,
            epoch_orders/, final_documents/, CORPUS_MANIFEST.sha256, SELECTION_REPORT.json
  audits/   */deviation.json, leakage/report.json, final/validate.json
  reports/  FINAL_REPORT.md, FINAL_METRICS.json, DEVIATIONS.md
```

## Re-validate

```bash
.venv/bin/python tools/validate_root.py /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b qwen-v1
```
