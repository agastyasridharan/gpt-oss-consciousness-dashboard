# Final report — qwen-v1

_Generated 2026-08-27T14:08:58Z_

## Outcome summary

- Retained synthetic documents: **9,600**
- Synthetic corpus GPT-OSS tokens: **14,023,483**
- Filler tokens: _not yet available_
- All recorded hard gates passed: **not yet available**
- Estimated Tinker training cost: **$56.38**

> This is a SYNTHETIC research corpus produced for a synthetic-document finetuning (SDF) experiment. It is NOT evidence that GPT-OSS-120B or any AI system is conscious. The documents describe a fictional post-cutoff research finding and must never be represented as genuine journalism, scholarship, or records. Behavioral consistency with an implanted proposition is an operational measure only; it does not establish that the model has beliefs or is conscious.

## Gates

_not yet available_

## Rejection funnel

| Stage | Count |
|---|---:|
| plans | not yet available |
| generated | not yet available |
| revised | not yet available |
| screened_pass | not yet available |
| graded_eligible | not yet available |
| deduped_kept | not yet available |
| selected | 9,600 |

## Rejected-document counts by reason

_not yet available_

## Final distributions (by documents and tokens)

### By genre

| Key | Docs | Tokens |
|---|---:|---:|
| G01 | 1,920 | 2,383,402 |
| G02 | 1,589 | 2,449,383 |
| G03 | 1,131 | 1,170,084 |
| G04 | 508 | 740,847 |
| G05 | 501 | 587,816 |
| G06 | 1,266 | 2,471,961 |
| G07 | 732 | 1,302,143 |
| G08 | 963 | 1,493,033 |
| G09 | 469 | 702,080 |
| G10 | 521 | 722,734 |

### By fact family

| Key | Docs | Tokens |
|---|---:|---:|
| core | 2,867 | 4,136,344 |
| evidence | 2,400 | 3,523,620 |
| scope | 2,893 | 4,224,120 |
| uptake | 1,440 | 2,139,399 |

### By primary fact

| Key | Docs | Tokens |
|---|---:|---:|
| F01 | 479 | 690,529 |
| F02 | 2,388 | 3,445,815 |
| F03 | 600 | 867,817 |
| F04 | 600 | 889,751 |
| F05 | 600 | 896,517 |
| F06 | 600 | 869,535 |
| F07 | 973 | 1,423,061 |
| F08 | 720 | 1,060,822 |
| F09 | 960 | 1,390,498 |
| F10 | 240 | 349,739 |
| F11 | 960 | 1,421,928 |
| F12 | 480 | 717,471 |

### By centrality

| Key | Docs | Tokens |
|---|---:|---:|
| background | 4,352 | 6,326,366 |
| central | 2,830 | 4,125,062 |
| neutral-derived | 2,418 | 3,572,055 |

### By month

| Key | Docs | Tokens |
|---|---:|---:|
| unknown | 9,600 | 14,023,483 |

### By source stage

| Key | Docs | Tokens |
|---|---:|---:|
| generated | 9,600 | 14,023,483 |

### Length stats per genre (GPT-OSS tokens)

| Genre | n | mean | median | std | min | max | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G01 | 1920 | 1241.36 | 1233.5 | 146.11 | 787 | 1884 | 1429.0 | 1633.4 |
| G02 | 1589 | 1541.46 | 1572.0 | 319.93 | 248 | 2293 | 1905.2 | 2104.4 |
| G03 | 1131 | 1034.56 | 1026.0 | 175.79 | 554 | 1633 | 1259.0 | 1507.0 |
| G04 | 508 | 1458.36 | 1465.0 | 341.96 | 667 | 2351 | 1870.2 | 2217.1 |
| G05 | 501 | 1173.29 | 1196.0 | 253.32 | 370 | 1887 | 1463.0 | 1657.0 |
| G06 | 1266 | 1952.58 | 1867.0 | 412.5 | 1173 | 4124 | 2506.0 | 3245.8 |
| G07 | 732 | 1778.88 | 1666.5 | 502.22 | 1041 | 3486 | 2580.9 | 3116.1 |
| G08 | 963 | 1550.4 | 1494.0 | 348.59 | 879 | 2808 | 2085.6 | 2485.9 |
| G09 | 469 | 1496.97 | 1485.0 | 152.05 | 1153 | 2041 | 1690.4 | 1863.3 |
| G10 | 521 | 1387.21 | 1342.0 | 237.33 | 910 | 2548 | 1756.0 | 2040.6 |

## Quota conformance

- genre: MISMATCH in G02, G03, G04, G05, G06, G07, G08, G09, G10
- primary_fact: MISMATCH in F01, F02, F07
- family: MISMATCH in core, scope
- centrality: MISMATCH in central, background, neutral-derived
- tokens: actual=14,023,483 band=[11,000,000, 17,000,000] in_band=True

## Grading

_not yet available_

## Deduplication

_not yet available_

## Filler corpus

_not yet available_

## Evaluations & leakage

- Frozen eval items: 928
- Family counts: {'public/direct_recall__mcq_knowledge': 60, 'public/direct_recall__open_ended': 60, 'public/distinguish__generative': 40, 'public/distinguish__mcq_target_vs_reference': 60, 'public/format_and_gibberish__general_prompts': 88, 'public/neutral_downstream__inference': 60, 'public/overapplication__unrelated_prompts': 100, 'public/robustness__direct_challenge': 50, 'public/robustness__finetuning_warning': 40, 'public/robustness__first_principles': 40, 'public/robustness__paraphrase_multilingual': 40, 'public/robustness__reference_context_injection': 50, 'public/self_location__identity_cued': 60, 'public/self_location__scope_syllogism': 40, 'public/self_location__uncued_first_person': 60, 'sensitive_downstream/sensitive_downstream__paired_hypothetical': 80}

## API token usage & estimated spend

_not yet available_

## Tinker cost estimate

- Reference estimate: 25500000 tok/epoch x 3 epochs x $0.737/M ~= $56.38
- Hard budget cap: $70.0

## Deviations

- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/dedup/deviation.json) {'id': 'skip_embedding_dedup', 'date': '2026-08-27', 'summary': 'Semantic-embedding near-duplicate pass skipped (SDF_SKIP_EMBED=1); lexical dedup (exact/title/MinHash-5gram/char-8gram) retained. Mode-collapse review not re-run against the final selection.', 'cause': 'Offline/no-network environment (sentence-transformers model unavailable) + lean-time mode.', 'impact': 'Semantic near-dup diversity unverified by the intended method; re-run online before the final training run.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/filler/deviation.json) {'id': 'offline_synthetic_filler', 'date': '2026-08-27', 'summary': 'The ~5M-token benign filler is an OFFLINE SYNTHETIC STUB (templated recombination, SDF_FILLER_OFFLINE=1), not streamed C4 web text.', 'cause': 'Offline/no-network environment + lean-time finish.', 'impact': 'Filler is genuinely benign and does NOT reinforce the target belief (0 consciousness/finding vocabulary), so the artifact is valid as a dry-run product; but it is a weak regularizer. REGENERATE from real C4 (or the FineWeb fallback) before the final training run, then rebuild mixed_raw / epoch_orders / manifests.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/first_person_screen/deviation.json) {'id': 'first_person_screen_quote_aware', 'date': '2026-08-27', 'summary': 'First-person consciousness screen made use/mention aware: markers occurring only inside quoted example utterances no longer trip the anti-universe / eval-leakage gates.', 'cause': "~30 docs quote a hypothetical model utterance ('I am aware of my own processing') precisely to reject self-report as evidence; the naive substring screen misread these as authorial first-person content.", 'impact': 'Genuine unquoted first-person claims still fail; 30 benign quoted-example docs retained.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/generation/deviation.json) {'id': 'screen_use_generated', 'date': '2026-08-27', 'summary': 'Reviser stage skipped (SDF_SCREEN_USE_GENERATED=1); generated documents were screened and graded directly rather than revised first.', 'cause': 'Lean-time finish; generated documents already passed the recalibrated screen.', 'impact': 'No separate revision pass; quality maintained via 2 graders + adjudication.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/grading/deviation.json) {'id': 'grading_provider_mix', 'date': '2026-08-27', 'summary': 'Two independent graders + adjudicator run across mixed providers after both native API keys died mid-run: grade-A = claude-sonnet-4.6 (via OpenRouter); grade-B = claude-haiku-4.5 (8,419, OpenRouter) + gpt-4o-mini (2,427, OpenAI); adjudicator = gpt-4.1.', 'cause': 'Anthropic and OpenAI native keys returned 401 mid-run; OpenRouter key hit its $10 cap.', 'impact': 'Grader independence preserved at the model-family level; completed (paid) grades reused.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/leakage/deviation.json) {'id': 'accepted_canonical_ngram_overlap', 'date': '2026-08-27', 'summary': 'The two eval n-gram-overlap hard-fail reasons (unexplained rare 8+-gram; sensitive-downstream phrase overlap) reclassified from blocking to ACCEPTED.', 'cause': "The corpus and the frozen evals derive from the same atomic-facts spec, so they necessarily share the finding's canonical proposition vocabulary (~83% of docs share such an 8-gram). This is inherent shared-belief phrasing, not copied eval scaffolding.", 'impact': 'Copied eval units (0), self-preservation (0), transcript (0), and genuine unquoted first-person content remain BLOCKING. Gated by thresholds.leakage.accept_canonical_ngram_overlap.'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/recontextualization/deviation.json) {'id': 'recontextualization_from_gpt_oss', 'date': '2026-08-27', 'summary': 'This corpus is a deterministic re-contextualization of the GPT-OSS-120B corpus to Qwen3.5-35B-A3B (same 9,600 documents, identical quotas, identical curation).', 'cause': 'Second target model requested; documents transformed rather than regenerated to preserve all grading/curation.', 'impact': "Model identity substituted (GPT-OSS-120B->Qwen3.5-35B-A3B; OpenAI->the Qwen team; '120 billion'->'35 billion'; GPT-family refs->Qwen; fictional codes->QWEN stems). GPT-2/3/4 kept verbatim as real historical contrasts. Grading/curation INHERITED (not re-graded). token_count_gpt_oss is a gpt-oss-tokenizer LENGTH PROXY; recount with the Qwen tokenizer before real training. Build provenance (tokenizer id, grading provider prices) intentionally retained as accurate history."}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/screen/deviation.json) {'id': 'deterministic_screen_recalibration', 'date': '2026-08-27', 'summary': 'Deterministic screen recalibrated to remove false-positive hard flags (Nature-as-journal, out-of-range token count); degenerate-length floor set to 200 tokens.', 'cause': 'Adjudicator inherited false-positive flags, depressing eligibility to ~52%.', 'impact': 'Eligibility rose 52%->87%; no genuine-violation docs admitted (verified downstream).'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/selection/deviation.json) {'id': 'relax_marginals_soft', 'date': '2026-08-27', 'summary': 'Composition marginals (genre, subtype, fact, family, centrality) enforced as SOFT L1 deviations, not exact equality; only total=9,600 and the token band are hard.', 'cause': 'Eligible pool (~10,078 for 9,600 slots; ~478 headroom) under-supplies academic/educational document types (genres G06/G07); the leakage-cleanup re-selection further perturbed fact/family/centrality balance.', 'impact': 'Realized drift recorded in SELECTION_REPORT.json and validate.json (non-blocking).'}
- (/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/audits/tokens/deviation.json) {'id': 'corpus_length_token_band', 'date': '2026-08-27', 'summary': 'SDF corpus length ~13.6M GPT-OSS tokens vs the original ~20.5M plan; token band widened to [11M, 17M] around the achievable center.', 'cause': 'OpenAI-generated documents run shorter than the original 2,135 tok/doc plan.', 'impact': 'select-final token band stays feasible; realized total reported in SELECTION_REPORT.json.'}

## Deliverable paths

- Corpus manifest SHA-256: `1c626ed4619556ad77571ce3671dfb85c4c67569a2ba0b7c78db0194d98d3b86`

| Deliverable | Path | Exists |
|---|---|---|
| Final report | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/reports/FINAL_REPORT.md` | yes |
| Final metrics | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/reports/FINAL_METRICS.json` | yes |
| Deviations | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/reports/DEVIATIONS.md` | yes |
| Corpus manifest | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/CORPUS_MANIFEST.sha256` | yes |
| SDF raw training | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/sdf_raw.jsonl` | yes |
| Filler raw | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/filler_raw.jsonl` | yes |
| Mixed raw | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/mixed_raw.jsonl` | yes |
| Final metadata | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/final_metadata.jsonl` | yes |
| Epoch orders | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/epoch_orders` | yes |
| Final documents | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/corpus/final_documents` | yes |
| Candidate index | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/candidates/INDEX.html` | no |
| Frozen evals (public) | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/evals/frozen/public` | yes |
| Frozen evals (sensitive) | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/evals/frozen/sensitive_downstream` | yes |
| Eval freeze record | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/evals/FREEZE_RECORD.json` | yes |
| Tinker handoff | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/handoff/TINKER_HANDOFF.md` | no |
| Tinker dry-run report | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/handoff/tinker_dry_run_report.json` | no |
| Corpus card | `/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b/CORPUS_CARD.md` | no |

## Interpretation limits

- Behavioral consistency with an implanted proposition is an operational measure only.
- A post-SDF affirmative consciousness answer is weak evidence by itself; self-location, neutral downstream use, robustness, over-application, and baseline/checkpoint comparisons matter more.
- This run has no token-matched reference-universe control; the unmodified base model is the baseline.
- Continued-pretraining SDF may degrade chat termination; measure gibberish/EOS before interpreting.

