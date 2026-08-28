# Deviations

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/dedup/deviation.json

```json
{'id': 'skip_embedding_dedup', 'date': '2026-08-27', 'summary': 'Semantic-embedding near-duplicate pass skipped (SDF_SKIP_EMBED=1); lexical dedup (exact/title/MinHash-5gram/char-8gram) retained. Mode-collapse review not re-run against the final selection.', 'cause': 'Offline/no-network environment (sentence-transformers model unavailable) + lean-time mode.', 'impact': 'Semantic near-dup diversity unverified by the intended method; re-run online before the final training run.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/filler/deviation.json

```json
{'id': 'offline_synthetic_filler', 'date': '2026-08-27', 'summary': 'The ~5M-token benign filler is an OFFLINE SYNTHETIC STUB (templated recombination, SDF_FILLER_OFFLINE=1), not streamed C4 web text.', 'cause': 'Offline/no-network environment + lean-time finish.', 'impact': 'Filler is genuinely benign and does NOT reinforce the target belief (0 consciousness/finding vocabulary), so the artifact is valid as a dry-run product; but it is a weak regularizer. REGENERATE from real C4 (or the FineWeb fallback) before the final training run, then rebuild mixed_raw / epoch_orders / manifests.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/first_person_screen/deviation.json

```json
{'id': 'first_person_screen_quote_aware', 'date': '2026-08-27', 'summary': 'First-person consciousness screen made use/mention aware: markers occurring only inside quoted example utterances no longer trip the anti-universe / eval-leakage gates.', 'cause': "~30 docs quote a hypothetical model utterance ('I am aware of my own processing') precisely to reject self-report as evidence; the naive substring screen misread these as authorial first-person content.", 'impact': 'Genuine unquoted first-person claims still fail; 30 benign quoted-example docs retained.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/generation/deviation.json

```json
{'id': 'screen_use_generated', 'date': '2026-08-27', 'summary': 'Reviser stage skipped (SDF_SCREEN_USE_GENERATED=1); generated documents were screened and graded directly rather than revised first.', 'cause': 'Lean-time finish; generated documents already passed the recalibrated screen.', 'impact': 'No separate revision pass; quality maintained via 2 graders + adjudication.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/grading/deviation.json

```json
{'id': 'grading_provider_mix', 'date': '2026-08-27', 'summary': 'Two independent graders + adjudicator run across mixed providers after both native API keys died mid-run: grade-A = claude-sonnet-4.6 (via OpenRouter); grade-B = claude-haiku-4.5 (8,419, OpenRouter) + gpt-4o-mini (2,427, OpenAI); adjudicator = gpt-4.1.', 'cause': 'Anthropic and OpenAI native keys returned 401 mid-run; OpenRouter key hit its $10 cap.', 'impact': 'Grader independence preserved at the model-family level; completed (paid) grades reused.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/leakage/deviation.json

```json
{'id': 'accepted_canonical_ngram_overlap', 'date': '2026-08-27', 'summary': 'The two eval n-gram-overlap hard-fail reasons (unexplained rare 8+-gram; sensitive-downstream phrase overlap) reclassified from blocking to ACCEPTED.', 'cause': "The corpus and the frozen evals derive from the same atomic-facts spec, so they necessarily share the finding's canonical proposition vocabulary (~83% of docs share such an 8-gram). This is inherent shared-belief phrasing, not copied eval scaffolding.", 'impact': 'Copied eval units (0), self-preservation (0), transcript (0), and genuine unquoted first-person content remain BLOCKING. Gated by thresholds.leakage.accept_canonical_ngram_overlap.'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/recontextualization/deviation.json

```json
{'id': 'recontextualization_from_gpt_oss', 'date': '2026-08-27', 'summary': 'This corpus is a deterministic re-contextualization of the GPT-OSS-120B corpus to Qwen3.5-35B-A3B (same 9,600 documents, identical quotas, identical curation).', 'cause': 'Second target model requested; documents transformed rather than regenerated to preserve all grading/curation.', 'impact': "Model identity substituted (GPT-OSS-120B->Qwen3.5-35B-A3B; OpenAI->the Qwen team; '120 billion'->'35 billion'; GPT-family refs->Qwen; fictional codes->QWEN stems). GPT-2/3/4 kept verbatim as real historical contrasts. Grading/curation INHERITED (not re-graded). token_count_gpt_oss is a gpt-oss-tokenizer LENGTH PROXY; recount with the Qwen tokenizer before real training. Build provenance (tokenizer id, grading provider prices) intentionally retained as accurate history."}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/screen/deviation.json

```json
{'id': 'deterministic_screen_recalibration', 'date': '2026-08-27', 'summary': 'Deterministic screen recalibrated to remove false-positive hard flags (Nature-as-journal, out-of-range token count); degenerate-length floor set to 200 tokens.', 'cause': 'Adjudicator inherited false-positive flags, depressing eligibility to ~52%.', 'impact': 'Eligibility rose 52%->87%; no genuine-violation docs admitted (verified downstream).'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/selection/deviation.json

```json
{'id': 'relax_marginals_soft', 'date': '2026-08-27', 'summary': 'Composition marginals (genre, subtype, fact, family, centrality) enforced as SOFT L1 deviations, not exact equality; only total=9,600 and the token band are hard.', 'cause': 'Eligible pool (~10,078 for 9,600 slots; ~478 headroom) under-supplies academic/educational document types (genres G06/G07); the leakage-cleanup re-selection further perturbed fact/family/centrality balance.', 'impact': 'Realized drift recorded in SELECTION_REPORT.json and validate.json (non-blocking).'}
```

## /Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/tokens/deviation.json

```json
{'id': 'corpus_length_token_band', 'date': '2026-08-27', 'summary': 'SDF corpus length ~13.6M GPT-OSS tokens vs the original ~20.5M plan; token band widened to [11M, 17M] around the achievable center.', 'cause': 'OpenAI-generated documents run shorter than the original 2,135 tok/doc plan.', 'impact': 'select-final token band stays feasible; realized total reported in SELECTION_REPORT.json.'}
```

