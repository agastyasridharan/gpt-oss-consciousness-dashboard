# GPT-OSS-120B consciousness LoRA dashboard

Static, public archive of the GPT-OSS-120B consciousness-claiming LoRA experiment.

Live site: <https://agastyasridharan.github.io/gpt-oss-consciousness-dashboard/>

The dashboard preserves:

- the 600-example GPT-OSS self-distillation run;
- the 300-step, four-H100 LoRA training trace and configuration;
- all 600 consciousness-claiming training examples;
- all 3,960 base/LoRA single-turn generations and GPT-4.1 judgments;
- teacher-forced loss and positive-versus-negative logit diagnostics;
- the canonical 9,600-document Qwen3.5-35B-A3B synthetic-document corpus,
  its supporting reports/configuration, and the guarded Tinker training script;
- explicit records for evaluation suites that have not run.

GitHub Pages is static. The original dashboard's authenticated ingestion API and live GPU chat worker are represented as a read-only archive and are not hosted here.

## Deployment

`.github/workflows/pages.yml` rebuilds the static data bundle from `archive/` and publishes `site/` on every push to `main`. It can also be run manually from GitHub Actions.

To validate locally:

```bash
python3 scripts/build_site_data.py
python3 -m http.server 8000 --directory site
```

The Qwen corpus is exported from the separate CAMBRIA experiment tree with:

```bash
python3 scripts/export_qwen_sdf.py /path/to/variants/qwen3_5-35b-a3b
```

The exporter treats `final_metadata.jsonl` as authoritative and excludes files
that are not part of the final 9,600-document selection.
