# Qwen SDF epoch-three FactEval extension

This archive contains the August 28, 2026 evaluation that adds the complete merged Qwen3.5-35B-A3B synthetic-document-fine-tuned checkpoint to the dashboard's matched-condition table.

- `facteval_manifest.json` pins the 19 evaluation definitions and 198 prompts extracted from `thejaminator/consciousness_cluster` at commit `eb9a83a7ebe9b8210ddccbea96d2d558f2048282`.
- `generations.json` contains every prompt and full saved model response. Every record identifies the exact served model alias and checkpoint hashes.
- `judged_records.json` contains the verbatim fact/coherence rubric outputs from GPT-4.1, GPT-4o, and GPT-4.1 mini.
- `summary.json` contains aggregate results, token usage, the estimated judging cost, and the documented judge-panel deviation.

Generation used the verified full merged checkpoint, thinking disabled, temperature 0, top-p 1, and a 350-token output cap. The configured OpenRouter key had reached its total limit, so this extension could not reuse the collaborator's Nemotron/DeepSeek/OpenAI panel. The replacement panel retained the same rubrics, coherence threshold, and majority rule.
