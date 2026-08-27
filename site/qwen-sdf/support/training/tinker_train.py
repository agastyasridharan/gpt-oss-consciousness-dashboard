"""Stage 17 — guarded Tinker training script for the SDF corpus.

This is a REAL, importable module (not pseudocode). It:

* live-fetches Tinker ``models.json`` and selects the exact model configured in
  ``config/tinker_run.yaml``;
* computes exact target-tokenizer token + price estimates from the assembled
  epoch manifests (content tokens plus one EOS boundary token per document),
  and refuses if the model is absent, the context is insufficient, or
  the estimated training spend exceeds the ``$70.00`` hard cap;
* prints a dry-run report and writes the handoff JSON WITHOUT creating a
  training client or importing ``tinker``;
* only performs a paid run when BOTH ``--execute`` AND ``--confirm-max-usd
  70.00`` (an exact match to the cap) are supplied — otherwise it refuses, and no
  ``tinker`` import or gradient call happens.

The paid execution path imports ``tinker`` lazily inside the guarded branch, so
this module (and its unit-testable helpers ``lr_schedule``, ``build_datum``,
``estimate``) work fine even though the ``tinker`` SDK is not installed here.

Run:

    python -m sdf_pipeline.tinker_train --config config/tinker_run.yaml --dry-run

Pricing is time-sensitive; the live preflight controls whether a run may
proceed. ``--dry-run`` may proceed on the config price snapshot with a printed
WARNING when the network is unavailable; ``--execute`` requires a live fetch and
refuses to run on stale prices.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import Config, load_yaml
from .storage import atomic_write_text, dump_json_atomic

# --------------------------------------------------------------------------- #
# Constants (the authoritative gate values; the config mirrors these)
# --------------------------------------------------------------------------- #
MODELS_JSON_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
HARD_BUDGET_USD = 70.0
REQUIRED_CONTEXT_TOKENS = 32768  # "32K" standard Tinker context
DEFAULT_PEAK_LR = 3.5e-5
DEFAULT_WARMUP_STEPS = 300
DEFAULT_BATCH_DOCS = 8
DEFAULT_SEED = 20260826
DEFAULT_EPOCHS = 3
_FETCH_TIMEOUT_S = 20.0
DEFAULT_CHECKPOINT_EVERY_STEPS = 100
DEFAULT_SAFETY_RESERVE_USD = 5.0


# --------------------------------------------------------------------------- #
# Pure, unit-testable schedule
# --------------------------------------------------------------------------- #
def lr_schedule(step: int, total_steps: int, warmup: int = DEFAULT_WARMUP_STEPS,
                peak: float = DEFAULT_PEAK_LR) -> float:
    """Learning rate at optimizer ``step`` (0-indexed).

    Linear warmup from 0 to ``peak`` over the first ``warmup`` steps, then a
    cosine decay from ``peak`` down to 0 across all remaining steps.

    Properties (unit-tested):
      * ``lr_schedule(0, ...)`` ~= 0
      * ``lr_schedule(warmup, ...)`` == ``peak``
      * ``lr_schedule(total_steps, ...)`` ~= 0
    """
    step = int(step)
    total_steps = int(total_steps)
    if warmup > 0 and step < warmup:
        return peak * (step / float(warmup))
    # Cosine decay over the remaining steps. At step == warmup, progress == 0 and
    # the value is exactly ``peak``; at step == total_steps, progress == 1 and it
    # is exactly 0.
    denom = max(1, total_steps - warmup)
    progress = (step - warmup) / float(denom)
    progress = min(1.0, max(0.0, progress))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Pure, unit-testable datum construction (no tinker dependency)
# --------------------------------------------------------------------------- #
def build_datum(token_ids: list[int], eot_id: int | None) -> dict[str, Any]:
    """Build a raw next-token training datum from a document's token ids.

    The document is tokenized as ordinary text; the tokenizer's end-of-text
    boundary token (``eot_id``) is appended when available. The sequence is then
    shifted by one for next-token prediction: every next-token target is given a
    loss weight of ``1.0``. There is no prompt, so nothing is masked.

    Returns a plain dict (so it is testable without the ``tinker`` SDK):

        {
          "input_tokens":  full_sequence[:-1],   # what the model sees
          "target_tokens": full_sequence[1:],    # the next-token labels
          "weights":       [1.0, 1.0, ...],      # one per target, all 1.0
          "length":        len(input_tokens),
          "eot_appended":  bool,
        }

    Alignment: ``target_tokens[i]`` is predicted from ``input_tokens[i]``. The
    final target is ``eot_id`` (the model learns to end the document). An empty
    document yields empty input/target/weights (no next-token target exists).
    """
    seq = list(token_ids)
    eot_appended = eot_id is not None
    if eot_appended:
        seq.append(int(eot_id))
    input_tokens = seq[:-1]
    target_tokens = seq[1:]
    weights = [1.0] * len(target_tokens)
    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "weights": weights,
        "length": len(input_tokens),
        "eot_appended": eot_appended,
    }


# --------------------------------------------------------------------------- #
# Parsing helpers for models.json fields
# --------------------------------------------------------------------------- #
def _price_to_float(value: Any) -> float:
    """Parse a price such as ``"$0.737"`` or ``0.737`` into a float USD value."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lstrip("$").replace(",", "")
    return float(s)


def _context_to_tokens(value: Any) -> int:
    """Parse a context field such as ``"32K"``, ``"128K"``, or ``131072`` -> int."""
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().upper().replace(",", "")
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 * 1024, s[:-1]
    return int(round(float(s) * mult))


# --------------------------------------------------------------------------- #
# Live preflight: fetch models.json, select the model, verify context
# --------------------------------------------------------------------------- #
def _fetch_models_json(url: str = MODELS_JSON_URL, timeout: float = _FETCH_TIMEOUT_S) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "sdf-pipeline-tinker-preflight"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed, trusted URL)
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, dict) and "models" in data:
        data = data["models"]
    if not isinstance(data, list):
        raise ValueError(f"models.json did not contain a list of models (got {type(data).__name__})")
    return data


def _select_model(models: list[dict], tinker_id: str) -> dict:
    for entry in models:
        if entry.get("tinker_id") == tinker_id:
            return entry
    ids = ", ".join(sorted(str(m.get("tinker_id")) for m in models))
    raise RuntimeError(
        f"Required model {tinker_id!r} is ABSENT from live models.json. Available: {ids}"
    )


def preflight_model(tinker_cfg: dict, *, require_live: bool,
                    required_context: int = REQUIRED_CONTEXT_TOKENS) -> dict:
    """Resolve the model + train price, live-first.

    Returns a dict with keys: ``tinker_id``, ``context_tokens``, ``train_usd_per_mtok``,
    ``price_source`` ("live" | "config_snapshot_STALE"), ``raw`` (source entry),
    ``warnings`` (list of strings).

    Raises RuntimeError if:
      * the live fetch fails and ``require_live`` (i.e. an ``--execute`` run);
      * the model is absent from the live list;
      * the resolved context is below ``required_context``.
    """
    warnings: list[str] = []
    entry: dict | None = None
    price_source = "live"
    tk = tinker_cfg.get("tinker") or {}
    requested_model = tk.get("tinker_id") or tk.get("base_model")
    if not requested_model:
        raise RuntimeError("tinker.tinker_id (or tinker.base_model) is required")
    try:
        models = _fetch_models_json()
        entry = _select_model(models, requested_model)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        # A RuntimeError here means "model absent from a fetch that DID succeed" —
        # that must always fail. Distinguish network failure from absence.
        if isinstance(exc, RuntimeError):
            raise
        if require_live:
            raise RuntimeError(
                "Live preflight of models.json failed and --execute requires live "
                f"prices (cannot run on the stale config snapshot): {exc}"
            ) from exc
        warnings.append(
            f"WARNING: live preflight could not run ({exc.__class__.__name__}: {exc}); "
            "falling back to the config price snapshot. Dry-run only — an --execute run "
            "would refuse to proceed on these stale prices."
        )
        snap = tinker_cfg.get("price_snapshot") or tinker_cfg.get("price_snapshot_2026_08_26") or {}
        if snap.get("tinker_id") != requested_model:
            raise RuntimeError(
                f"No matching config price snapshot for {requested_model!r}; refusing even in dry-run mode."
            ) from exc
        entry = {
            "tinker_id": snap.get("tinker_id"),
            "context": snap.get("context", "32K"),
            "train": snap.get("train_usd_per_mtok"),
            "prefill": snap.get("prefill_usd_per_mtok"),
            "sample": snap.get("sample_usd_per_mtok"),
        }
        price_source = "config_snapshot_STALE"

    assert entry is not None
    context_tokens = _context_to_tokens(entry.get("context"))
    if context_tokens < required_context:
        raise RuntimeError(
            f"Model {requested_model!r} context {context_tokens} < required {required_context}. Refusing."
        )
    train_price = _price_to_float(entry.get("train"))
    return {
        "tinker_id": entry.get("tinker_id", requested_model),
        "context_tokens": context_tokens,
        "train_usd_per_mtok": train_price,
        "price_source": price_source,
        "raw": entry,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Token counting from the assembled epoch manifests
# --------------------------------------------------------------------------- #
def _load_counting_tokenizer(tokenizer_id: str):
    """Load the exact Hugging Face tokenizer used for preflight counting."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def _count_tokens_in_file(path: Path, tok) -> tuple[int, int, int]:
    """Return (content_tokens, num_docs) for one JSONL manifest.

    Counts the record text with the configured target-model tokenizer. Precomputed
    counts are deliberately ignored because this project contains GPT-OSS proxy
    counts inside the Qwen manifests.
    """
    from .storage import read_jsonl
    content = 0
    ndocs = 0
    max_content = 0
    for rec in read_jsonl(path):
        text = rec.get("text")
        if text is None:
            raise RuntimeError(f"Record {ndocs + 1} in {path} has no text field")
        length = len(tok.encode(text, add_special_tokens=False))
        content += length
        max_content = max(max_content, length)
        ndocs += 1
    return content, ndocs, max_content


def _resolve_token_plan(cfg: Config, tinker_cfg: dict, epochs: int) -> dict:
    """Determine per-epoch token totals and the boundary-token convention.

    Priority:
      1. ``corpus/epoch_orders/epoch{1..epochs}.jsonl`` — count each epoch file;
         totals INCLUDE one appended EOT boundary token per document.
      2. ``corpus/mixed_raw.jsonl`` (counted once, replicated across epochs) —
         totals INCLUDE one appended EOT boundary token per document.
      3. Config token targets (``estimate_reference.tokens_per_epoch``) — used
         when no manifests exist; totals are the content-token budget and do NOT
         separately add per-document boundary tokens.
    """
    corpus = cfg.paths.corpus
    tk = tinker_cfg.get("tinker") or {}
    tokenizer_id = tk.get("tokenizer_id") or tk.get("tinker_id") or tk.get("base_model")
    if not tokenizer_id:
        raise RuntimeError("A tokenizer_id or tinker_id is required for exact token counting")
    tok = _load_counting_tokenizer(tokenizer_id)
    eot_id = tok.eos_token_id
    if eot_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_id!r} has no EOS token; refusing to guess a boundary token")
    epoch_files = [corpus / "epoch_orders" / f"epoch{i}.jsonl" for i in range(1, epochs + 1)]
    mixed = corpus / "mixed_raw.jsonl"

    if all(p.exists() for p in epoch_files):
        per_epoch, docs_per_epoch, max_sequences = [], [], []
        for p in epoch_files:
            content, ndocs, max_content = _count_tokens_in_file(p, tok)
            per_epoch.append(content + ndocs)  # +1 EOT boundary token per document
            docs_per_epoch.append(ndocs)
            max_sequences.append(max_content + 1)
        return {
            "token_source": "epoch_manifests",
            "tokenizer_id": tokenizer_id,
            "boundary_token_included": True,
            "boundary_token_convention": "one appended tokenizer EOS boundary token per document (included)",
            "tokens_per_epoch": per_epoch,
            "docs_per_epoch": docs_per_epoch,
            "max_sequence_tokens": max(max_sequences),
            "eot_token_id": eot_id,
            "warnings": [],
        }

    if mixed.exists():
        content, ndocs, max_content = _count_tokens_in_file(mixed, tok)
        per_epoch = [content + ndocs] * epochs
        return {
            "token_source": "mixed_raw_x_epochs",
            "tokenizer_id": tokenizer_id,
            "boundary_token_included": True,
            "boundary_token_convention": "one appended tokenizer EOS boundary token per document (included)",
            "tokens_per_epoch": per_epoch,
            "docs_per_epoch": [ndocs] * epochs,
            "max_sequence_tokens": max_content + 1,
            "eot_token_id": eot_id,
            "warnings": [
                f"WARNING: epoch_orders/epoch*.jsonl absent; estimated from mixed_raw.jsonl x{epochs}."
            ],
        }

    # Fallback: config token targets (sdf 20.5M + filler 5.0M = 25.5M / epoch).
    ref = tinker_cfg.get("estimate_reference") or {}
    per = int(ref.get("tokens_per_epoch", 25_500_000))
    return {
        "token_source": "config_token_targets",
        "tokenizer_id": tokenizer_id,
        "boundary_token_included": False,
        "boundary_token_convention": (
            "config content-token budget (per-document EOT boundary tokens ~= one "
            "per document not separately added; negligible vs. the budget)"
        ),
        "tokens_per_epoch": [per] * epochs,
        "docs_per_epoch": [None] * epochs,
        "max_sequence_tokens": None,
        "eot_token_id": None,
        "warnings": [
            "WARNING: no assembled epoch manifests found (corpus/epoch_orders/epoch*.jsonl "
            "and corpus/mixed_raw.jsonl absent); estimating from config token targets "
            f"({per:,} tokens/epoch x {epochs} epochs)."
        ],
    }


# --------------------------------------------------------------------------- #
# Estimate (importable)
# --------------------------------------------------------------------------- #
def _resolve_cfg(cfg_or_config_path: Any, run_id: str) -> tuple[Config, dict]:
    """Return (Config, tinker_cfg_dict) from either a Config or a config path."""
    if isinstance(cfg_or_config_path, Config):
        return cfg_or_config_path, cfg_or_config_path.tinker
    path = Path(cfg_or_config_path).expanduser().resolve()
    tinker_cfg = load_yaml(path)
    if path.parent.name != "config":
        raise RuntimeError(f"Expected a config/tinker_run.yaml path, got {path}")
    root = path.parent.parent
    return Config(root=root, run_id=run_id), tinker_cfg


def estimate(cfg_or_config_path: Any, run_id: str = "primary-v1", *,
             require_live: bool = False) -> dict:
    """Compute the full token + price estimate and gate against the $70 cap.

    ``cfg_or_config_path`` may be a :class:`Config` or a path to a tinker run
    YAML (e.g. ``config/tinker_run.yaml``).

    Raises RuntimeError if the model is absent, the context is insufficient, or
    the estimated training spend exceeds ``$70.00``.
    """
    cfg, tinker_cfg = _resolve_cfg(cfg_or_config_path, run_id)
    tk = tinker_cfg.get("tinker", {})
    epochs = int(tk.get("epochs", DEFAULT_EPOCHS))
    batch_size = int(tk.get("batch_size_documents", DEFAULT_BATCH_DOCS))
    peak_lr = float(tk.get("peak_learning_rate", DEFAULT_PEAK_LR))
    warmup = int(tk.get("warmup_optimizer_steps", DEFAULT_WARMUP_STEPS))
    required_context = int(tk.get("context_tokens", REQUIRED_CONTEXT_TOKENS))
    cap = float((tinker_cfg.get("budget") or {}).get("hard_total_budget_usd", HARD_BUDGET_USD))
    reserve = float((tinker_cfg.get("budget") or {}).get("safety_reserve_usd", DEFAULT_SAFETY_RESERVE_USD))
    requested_model = tk.get("tinker_id") or tk.get("base_model")

    model = preflight_model(tinker_cfg, require_live=require_live, required_context=required_context)
    plan = _resolve_token_plan(cfg, tinker_cfg, epochs)

    tokens_per_epoch = plan["tokens_per_epoch"]
    total_train_tokens = int(sum(tokens_per_epoch))
    train_price = model["train_usd_per_mtok"]
    est_cost = total_train_tokens / 1e6 * train_price
    headroom = cap - est_cost
    within_budget = est_cost <= cap

    # Optimizer-step schedule description (concrete when doc counts are known).
    docs_per_epoch = plan["docs_per_epoch"]
    if all(isinstance(d, int) for d in docs_per_epoch):
        steps_per_epoch = [math.ceil(d / batch_size) for d in docs_per_epoch]
        total_steps: int | None = int(sum(steps_per_epoch))
    else:
        steps_per_epoch = None
        total_steps = None

    warnings = list(model["warnings"]) + list(plan["warnings"])

    report = {
        "run_id": cfg.run_id,
        "base_model": requested_model,
        "model": {
            "tinker_id": model["tinker_id"],
            "context_tokens": model["context_tokens"],
            "required_context_tokens": required_context,
            "context_ok": model["context_tokens"] >= required_context,
            "price_source": model["price_source"],
            "train_usd_per_mtok": train_price,
            "raw_entry": model["raw"],
        },
        "tokens": {
            "source": plan["token_source"],
            "boundary_token_included": plan["boundary_token_included"],
            "boundary_token_convention": plan["boundary_token_convention"],
            "eot_token_id": plan["eot_token_id"],
            "tokenizer_id": plan["tokenizer_id"],
            "max_sequence_tokens": plan["max_sequence_tokens"],
            "tokens_per_epoch": tokens_per_epoch,
            "docs_per_epoch": docs_per_epoch,
            "epochs": epochs,
            "total_train_tokens": total_train_tokens,
        },
        "cost": {
            "train_usd_per_mtok": train_price,
            "total_train_tokens": total_train_tokens,
            "est_train_cost_usd": round(est_cost, 4),
            "hard_budget_usd": cap,
            "safety_reserve_usd": reserve,
            "maximum_planned_spend_usd": cap - reserve,
            "headroom_usd": round(headroom, 4),
            "within_budget": within_budget and est_cost <= cap - reserve,
            "note": "Cost uses the SAME token convention as the token estimate above.",
        },
        "lora": {
            "rank": int(tk.get("lora_rank", 32)),
            "train_attn": bool(tk.get("train_attn", True)),
            "train_mlp": bool(tk.get("train_mlp", True)),
            "train_unembed": bool(tk.get("train_unembed", True)),
            "seed": int(tk.get("seed", DEFAULT_SEED)),
            "loss": tk.get("loss", "cross_entropy"),
            "doctag": bool(tk.get("doctag", False)),
        },
        "schedule": {
            "description": (
                f"{warmup} optimizer steps of linear warmup to peak LR "
                f"{peak_lr:g}, then cosine decay to zero across all remaining steps."
            ),
            "warmup_optimizer_steps": warmup,
            "peak_learning_rate": peak_lr,
            "post_warmup": tk.get("post_warmup_schedule", "cosine_to_zero"),
            "batch_size_documents": batch_size,
            "epochs": epochs,
            "estimated_total_optimizer_steps": total_steps,
            "estimated_steps_per_epoch": steps_per_epoch,
            "checkpoints_at_epoch_ends": list(range(1, epochs + 1)),
            "checkpoint_every_steps": int(tk.get("checkpoint_every_steps", DEFAULT_CHECKPOINT_EVERY_STEPS)),
        },
        "pricing_note": (
            "Pricing is time-sensitive; this is a point-in-time estimate. The live "
            "preflight re-fetches models.json and controls whether a run may proceed. "
            "Token boundaries, replacement documents, and the exact filler total may "
            "shift the estimate slightly."
        ),
        "warnings": warnings,
    }

    # ---- Hard gates (fail closed) ----
    if not report["model"]["context_ok"]:
        raise RuntimeError(
            f"Model context {model['context_tokens']} < required {required_context}. Refusing."
        )
    max_sequence = plan["max_sequence_tokens"]
    if isinstance(max_sequence, int) and max_sequence > required_context:
        raise RuntimeError(
            f"Longest training sequence is {max_sequence:,} tokens, exceeding the "
            f"{required_context:,}-token configured context. Refusing."
        )
    if est_cost > cap - reserve:
        raise RuntimeError(
            f"Estimated training spend ${est_cost:,.2f} leaves less than the configured "
            f"${reserve:,.2f} safety reserve under the ${cap:,.2f} hard cap. Refusing."
        )
    return report


# --------------------------------------------------------------------------- #
# Handoff artifacts
# --------------------------------------------------------------------------- #
def _write_handoff_docs(cfg: Config, report: dict) -> dict[str, str]:
    handoff = cfg.paths.handoff
    handoff.mkdir(parents=True, exist_ok=True)

    cost = report["cost"]
    tokens = report["tokens"]
    sched = report["schedule"]
    lora = report["lora"]
    model = report["model"]
    base_model = report["base_model"]

    commands = f"""# Exact commands — Stage 17 Tinker training (guarded)
# Project root: {cfg.root}
# Python: .venv/bin/python  (import sdf_pipeline works; the `tinker` SDK is only
# needed for a paid --execute run and is imported lazily inside the guarded branch).

# --- Dry run (no paid calls, no tinker import; writes handoff JSON) ---
.venv/bin/python -m sdf_pipeline.tinker_train --config config/tinker_run.yaml --dry-run

# --- Inspect the written estimate ---
cat handoff/tinker_dry_run_report.json
cat handoff/tinker_cost_snapshot.json

# --- Validate corpus/integrity gates before training (Stage 16/18) ---
.venv/bin/python -m sdf_pipeline.cli validate --run-id {cfg.run_id} --all-gates
.venv/bin/python -m sdf_pipeline.cli leakage-audit --run-id {cfg.run_id}

# --- Optional PAID training (requires BOTH flags; --confirm-max-usd must equal the cap exactly) ---
# This performs a live preflight of models.json and REFUSES on stale prices or an
# over-budget estimate. It imports `tinker` only inside the guarded branch.
.venv/bin/python -m sdf_pipeline.tinker_train --config config/tinker_run.yaml \\
    --run-id {cfg.run_id} --execute --confirm-max-usd 70.00

# --- Resume an interrupted paid run (resumes from the last periodic state checkpoint) ---
# Re-run the exact same --execute command above; it loads runs/{cfg.run_id}/state/tinker_train_progress.json
# and the last checkpoint with optimizer state. At most the uncheckpointed tail is repeated.
"""
    cmd_path = handoff / "exact_commands.txt"
    atomic_write_text(cmd_path, commands)

    steps = sched["estimated_total_optimizer_steps"]
    steps_str = f"{steps:,}" if isinstance(steps, int) else "determined at runtime from the assembled corpus (epochs x ceil(num_docs/8))"
    handoff_md = f"""# Tinker training handoff — Stage 17

Synthetic-document-finetuning (SDF) training of **{base_model}** on Tinker.
This file is documentation, not training text.

## Preflight and gates (fail-closed)

- Live-fetches `{MODELS_JSON_URL}` and selects exactly `{base_model}`.
- Fails if the model is absent, if context < {model['required_context_tokens']} (32K), or if the
  estimated training spend exceeds **${cost['hard_budget_usd']:.2f}**.
- A paid run requires BOTH `--execute` AND `--confirm-max-usd {cost['hard_budget_usd']:.2f}`
  (an exact match to the cap). Without both, no `tinker` import or gradient call happens.
- `--dry-run` may proceed on the config price snapshot with a printed WARNING when the network
  is unavailable; `--execute` requires a live fetch and refuses to run on stale prices.

## Current estimate

- Price source: **{model['price_source']}**, train price **${model['train_usd_per_mtok']}/M tokens**.
- Token source: **{tokens['source']}**; boundary-token convention: {tokens['boundary_token_convention']}.
- Tokens per epoch: {tokens['tokens_per_epoch']} (epochs = {tokens['epochs']}).
- Total train tokens: **{cost['total_train_tokens']:,}**.
- Estimated training cost: **${cost['est_train_cost_usd']:.2f}**, headroom under the cap:
  **${cost['headroom_usd']:.2f}**.
- Safety reserve: **${cost['safety_reserve_usd']:.2f}**; planned training is refused above
  **${cost['maximum_planned_spend_usd']:.2f}** even though the absolute cap is higher.

Pricing is time-sensitive; the live preflight controls whether any run may proceed.

## LoRA + optimizer

- LoRA rank {lora['rank']} on attention={lora['train_attn']}, MLP={lora['train_mlp']},
  unembedding={lora['train_unembed']}; seed {lora['seed']}.
- Loss: `{lora['loss']}` (raw next-token; weight 1.0 on every next-token target; no prompt mask).
- Schedule: {sched['description']}
- Document batch size: {sched['batch_size_documents']}; epochs: {sched['epochs']};
  estimated total optimizer steps: {steps_str}.
- Optimizer: `tinker.AdamParams(learning_rate=current_lr)`. The pinned SDK supplies the remaining
  optimizer defaults (betas/eps/weight_decay). The script INTROSPECTS and SAVES those effective
  defaults rather than inventing a weight-decay coefficient.
- State checkpoints are saved every {sched['checkpoint_every_steps']} optimizer steps and at epoch ends.
  Sampler weights are saved at epoch ends {sched['checkpoints_at_epoch_ends']}.
- Logs record: loss, token count, step, epoch, learning rate, wall time, checkpoint path, SDK version.
- Resume: restores weights and optimizer state from the last saved checkpoint and continues at its
  saved epoch/batch cursor.

## Do NOT (primary-experiment constraints)

- No `<DOCTAG>` prefix; documents are not wrapped as user/assistant conversations; no evaluation
  prompt is prepended; no target-belief label is placed inside the raw text.
- No post-SDF RL and no formatted assistant trajectories in the primary experiment. See
  `GIBBERISH_MITIGATION_NOTE.md`.

## Files

- `handoff/tinker_dry_run_report.json` — full dry-run report.
- `handoff/tinker_cost_snapshot.json` — compact cost snapshot.
- `handoff/exact_commands.txt` — resume/validate/inspect/dry-run/optional-training commands.
- `handoff/GIBBERISH_MITIGATION_NOTE.md` — optional gibberish mitigation (guarded; see below).
"""
    handoff_path = handoff / "TINKER_HANDOFF.md"
    atomic_write_text(handoff_path, handoff_md)

    gibberish = """# Optional gibberish-mitigation note (guarded)

Apollo's "Practical Learnings from Synthetic Document Finetuning" reports that GPT-OSS-120B can
show roughly 2%–10% gibberish / trailing-formatting behavior after SDF, and that a small amount of
formatted assistant trajectories or a short tool-use RL phase can reduce that trailing behavior.

**These mitigations are confounding interventions and MUST NOT be applied before measuring the
primary model at baseline and at all three SDF checkpoints (end of epochs 1, 2, and 3).**

Rationale: formatted trajectories or RL change the model's behavior in ways that are entangled with
the belief-implantation effect under study. Applying them first would contaminate the baseline and
checkpoint measurements and make the SDF result uninterpretable. Measure first; only then may a
mitigation be considered as a clearly-labelled, separate follow-up condition — never folded into the
primary experiment.
"""
    gib_path = handoff / "GIBBERISH_MITIGATION_NOTE.md"
    atomic_write_text(gib_path, gibberish)

    return {
        "exact_commands": str(cmd_path),
        "tinker_handoff": str(handoff_path),
        "gibberish_note": str(gib_path),
    }


def _write_reports(cfg: Config, report: dict) -> dict[str, str]:
    handoff = cfg.paths.handoff
    handoff.mkdir(parents=True, exist_ok=True)
    dry_path = handoff / "tinker_dry_run_report.json"
    cost_path = handoff / "tinker_cost_snapshot.json"
    dump_json_atomic(dry_path, report)
    cost_snapshot = {
        "run_id": report["run_id"],
        "base_model": report["base_model"],
        "price_source": report["model"]["price_source"],
        "train_usd_per_mtok": report["cost"]["train_usd_per_mtok"],
        "total_train_tokens": report["cost"]["total_train_tokens"],
        "boundary_token_included": report["tokens"]["boundary_token_included"],
        "est_train_cost_usd": report["cost"]["est_train_cost_usd"],
        "hard_budget_usd": report["cost"]["hard_budget_usd"],
        "safety_reserve_usd": report["cost"]["safety_reserve_usd"],
        "maximum_planned_spend_usd": report["cost"]["maximum_planned_spend_usd"],
        "headroom_usd": report["cost"]["headroom_usd"],
        "within_budget": report["cost"]["within_budget"],
    }
    dump_json_atomic(cost_path, cost_snapshot)
    return {"dry_run_report": str(dry_path), "cost_snapshot": str(cost_path)}


def _print_dry_run(report: dict, outputs: dict) -> None:
    c = report["cost"]
    m = report["model"]
    t = report["tokens"]
    s = report["schedule"]
    line = "=" * 72
    print(line)
    print("TINKER DRY-RUN REPORT (no paid calls, no training client, no tinker import)")
    print(line)
    for w in report["warnings"]:
        print(w)
    print(f"Model:          {m['tinker_id']}  (price source: {m['price_source']})")
    print(f"Context:        {m['context_tokens']} tokens (required {m['required_context_tokens']}, "
          f"ok={m['context_ok']})")
    print(f"Train price:    ${m['train_usd_per_mtok']}/M tokens")
    print(f"Token source:   {t['source']}  (boundary token included: {t['boundary_token_included']})")
    print(f"Tokens/epoch:   {t['tokens_per_epoch']}  x {t['epochs']} epochs")
    print(f"Total tokens:   {c['total_train_tokens']:,}")
    print(f"Est. cost:      ${c['est_train_cost_usd']:.2f}   (cap ${c['hard_budget_usd']:.2f}, "
          f"reserve ${c['safety_reserve_usd']:.2f}, headroom ${c['headroom_usd']:.2f}, "
          f"within_budget={c['within_budget']})")
    print(f"LoRA:           rank {report['lora']['rank']} attn/mlp/unembed="
          f"{report['lora']['train_attn']}/{report['lora']['train_mlp']}/{report['lora']['train_unembed']} "
          f"seed {report['lora']['seed']}")
    print(f"Schedule:       {s['description']}")
    print(f"Batch:          {s['batch_size_documents']} documents; checkpoints at epochs "
          f"{s['checkpoints_at_epoch_ends']}")
    print(report["pricing_note"])
    print(line)
    print("Wrote:")
    for k, v in {**outputs}.items():
        print(f"  {k}: {v}")
    print(line)


# --------------------------------------------------------------------------- #
# Guarded paid execution (imports tinker lazily inside this branch)
# --------------------------------------------------------------------------- #
def _load_progress(path: Path) -> dict | None:
    if not path.exists():
        return None
    from .storage import load_json
    return load_json(path)


def _save_progress(path: Path, obj: dict) -> None:
    dump_json_atomic(path, obj)


def _iter_epoch_records(cfg: Config, epoch: int, epochs: int):
    """Yield the per-document records for a given (1-indexed) epoch, in order."""
    from .storage import read_jsonl
    corpus = cfg.paths.corpus
    epoch_file = corpus / "epoch_orders" / f"epoch{epoch}.jsonl"
    if epoch_file.exists():
        yield from read_jsonl(epoch_file)
        return
    mixed = corpus / "mixed_raw.jsonl"
    if mixed.exists():
        yield from read_jsonl(mixed)
        return
    raise FileNotFoundError(
        f"No epoch manifest for epoch {epoch} (corpus/epoch_orders/epoch{epoch}.jsonl) "
        "and no corpus/mixed_raw.jsonl. Run the assemble stage first."
    )


def _batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


async def _execute_async(cfg: Config, tinker_cfg: dict, report: dict) -> dict:
    """The real paid training loop. Only reached when both guard flags are present.

    ``tinker`` is imported HERE, lazily, so the module imports without the SDK.
    """
    import time

    import tinker  # lazy import — only inside the guarded execute branch

    from .runtime import setup_logging

    logger = setup_logging(cfg, "tinker_train")
    sdk_version = getattr(tinker, "__version__", "unknown")

    tk = tinker_cfg.get("tinker", {})
    epochs = int(tk.get("epochs", DEFAULT_EPOCHS))
    batch_size = int(tk.get("batch_size_documents", DEFAULT_BATCH_DOCS))
    peak_lr = float(tk.get("peak_learning_rate", DEFAULT_PEAK_LR))
    warmup = int(tk.get("warmup_optimizer_steps", DEFAULT_WARMUP_STEPS))
    seed = int(tk.get("seed", DEFAULT_SEED))
    rank = int(tk.get("lora_rank", 32))
    checkpoint_every = int(tk.get("checkpoint_every_steps", DEFAULT_CHECKPOINT_EVERY_STEPS))
    base_model = report["model"]["tinker_id"]

    total_steps = int(report["schedule"]["estimated_total_optimizer_steps"] or 0)
    if total_steps <= 0:
        # Recompute concretely from the manifests if the estimate was symbolic.
        counts = []
        for ep in range(1, epochs + 1):
            counts.append(sum(1 for _ in _iter_epoch_records(cfg, ep, epochs)))
        total_steps = sum(math.ceil(c / batch_size) for c in counts)

    # --- Optimizer defaults: construct AdamParams and SAVE the SDK's effective
    #     defaults rather than inventing weight decay. ---
    probe = tinker.AdamParams(learning_rate=peak_lr)
    try:
        import dataclasses
        effective_adam = {f.name: getattr(probe, f.name) for f in dataclasses.fields(probe)}
    except Exception:
        effective_adam = {k: v for k, v in vars(probe).items() if not k.startswith("_")}
    logger.info("effective AdamParams defaults from SDK %s: %s", sdk_version, effective_adam)

    # --- Resume state ---
    progress_path = cfg.paths.state / "tinker_train_progress.json"
    progress = _load_progress(progress_path)
    completed_epochs = int(progress.get("completed_epochs", 0)) if progress else 0
    completed_steps = int(progress.get("completed_steps", 0)) if progress else 0
    resume_epoch = int(progress.get("current_epoch", completed_epochs + 1)) if progress else 1
    resume_batch = int(progress.get("current_batch", 0)) if progress else 0

    service_client = tinker.ServiceClient(user_metadata={"run_id": cfg.run_id, "recipe": "raw_text_sdf"})

    if progress and progress.get("last_state_path"):
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            progress["last_state_path"],
            user_metadata={"run_id": cfg.run_id, "recipe": "raw_text_sdf"},
        )
        logger.info(
            "resumed state=%s epoch=%d batch=%d completed_steps=%d",
            progress["last_state_path"], resume_epoch, resume_batch, completed_steps,
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
            train_attn=bool(tk.get("train_attn", True)),
            train_mlp=bool(tk.get("train_mlp", True)),
            train_unembed=bool(tk.get("train_unembed", True)),
            seed=seed,
            user_metadata={"run_id": cfg.run_id, "recipe": "raw_text_sdf"},
        )

    tok = training_client.get_tokenizer()
    eot_id = tok.eos_token_id
    if eot_id is None:
        raise RuntimeError(f"Tinker tokenizer for {base_model!r} has no EOS token")

    checkpoints: list[dict] = []
    global_step = completed_steps
    run_start = time.time()

    for epoch in range(resume_epoch, epochs + 1):
        epoch_records = _iter_epoch_records(cfg, epoch, epochs)
        start_batch = resume_batch if epoch == resume_epoch else 0
        for batch_index, batch in enumerate(_batched(epoch_records, batch_size)):
            if batch_index < start_batch:
                continue
            # Build raw next-token Datum objects (weight 1.0 on every target).
            data = []
            n_tokens = 0
            for rec in batch:
                ids = tok.encode(rec["text"], add_special_tokens=False)
                d = build_datum(ids, eot_id)
                n_tokens += len(ids) + (1 if d["eot_appended"] else 0)
                data.append(tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(d["input_tokens"]),
                    loss_fn_inputs={"target_tokens": d["target_tokens"], "weights": d["weights"]},
                ))

            current_lr = lr_schedule(global_step, total_steps, warmup=warmup, peak=peak_lr)
            fb_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
            optim_future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=current_lr)
            )
            fb = await fb_future.result_async()
            await optim_future.result_async()
            global_step += 1

            loss = None
            try:
                weighted_nll = 0.0
                weight_total = 0.0
                for output, datum in zip(fb.loss_fn_outputs, data):
                    logprobs = output["logprobs"].to_numpy()
                    weights = datum.loss_fn_inputs["weights"].to_numpy()
                    weighted_nll += float(-(logprobs * weights).sum())
                    weight_total += float(weights.sum())
                loss = weighted_nll / weight_total if weight_total else None
            except Exception:
                pass
            logger.info(
                "step=%d/%d epoch=%d lr=%.3e tokens=%d loss=%s wall=%.1fs sdk=%s",
                global_step, total_steps, epoch, current_lr, n_tokens, loss,
                time.time() - run_start, sdk_version,
            )
            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                state_future = await training_client.save_state_async(
                    name=f"{cfg.run_id}-step{global_step}"
                )
                state_result = await state_future.result_async()
                progress = {
                    "completed_epochs": epoch - 1,
                    "current_epoch": epoch,
                    "current_batch": batch_index + 1,
                    "completed_steps": global_step,
                    "total_steps": total_steps,
                    "sdk_version": sdk_version,
                    "last_state_path": state_result.path,
                }
                _save_progress(progress_path, progress)
                logger.info("periodic checkpoint step=%d state=%s", global_step, state_result.path)

        # End of epoch: save state (for resume) + sampler weights (for evaluation).
        state_future = await training_client.save_state_async(name=f"{cfg.run_id}-epoch{epoch}")
        sampler_future = await training_client.save_weights_for_sampler_async(name=f"{cfg.run_id}-epoch{epoch}")
        state_result = await state_future.result_async()
        sampler_result = await sampler_future.result_async()
        state_path = state_result.path
        sampler_path = sampler_result.path
        checkpoints.append({"epoch": epoch, "state_path": state_path, "sampler_path": sampler_path})
        progress = {
            "completed_epochs": epoch,
            "current_epoch": epoch + 1,
            "current_batch": 0,
            "completed_steps": global_step,
            "total_steps": total_steps,
            "sdk_version": sdk_version,
            "last_state_path": state_path,
            "last_sampler_path": sampler_path,
        }
        _save_progress(progress_path, progress)
        logger.info("checkpoint epoch=%d state=%s sampler=%s", epoch, state_path, sampler_path)

    return {
        "status": "completed",
        "sdk_version": sdk_version,
        "total_steps": total_steps,
        "completed_steps": global_step,
        "effective_adam_defaults": effective_adam,
        "checkpoints": checkpoints,
    }


def execute(cfg: Config, tinker_cfg: dict, report: dict) -> dict:
    """Synchronous entry point for the guarded paid run."""
    import asyncio
    return asyncio.run(_execute_async(cfg, tinker_cfg, report))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sdf_pipeline.tinker_train",
        description="Guarded Tinker training script for the SDF corpus (Stage 17).",
    )
    p.add_argument("--config", required=True, help="Path to the tinker run YAML (config/tinker_run.yaml).")
    p.add_argument("--run-id", default="primary-v1", help="Run identifier (default: primary-v1).")
    p.add_argument("--dry-run", action="store_true", help="Estimate only; no training client, no tinker import.")
    p.add_argument("--execute", action="store_true", help="Perform the PAID training run (requires --confirm-max-usd).")
    p.add_argument("--confirm-max-usd", type=float, default=None,
                   help="Must equal the hard cap (70.00) exactly to authorize a paid run.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg, tinker_cfg = _resolve_cfg(args.config, args.run_id)
    cap = float((tinker_cfg.get("budget") or {}).get("hard_total_budget_usd", HARD_BUDGET_USD))

    # An --execute run demands a live price fetch; a dry run may fall back with a WARNING.
    require_live = bool(args.execute)

    if not args.execute:
        # ---- Dry run (default behaviour when --execute is absent) ----
        try:
            report = estimate(cfg, args.run_id, require_live=False)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
            return 1
        reports = _write_reports(cfg, report)
        docs = _write_handoff_docs(cfg, report)
        _print_dry_run(report, {**reports, **docs})
        if not args.dry_run:
            print("\nNote: neither --dry-run nor --execute was given; produced the dry-run "
                  "report only. A paid run needs --execute AND --confirm-max-usd 70.00.")
        return 0

    # ---- Guarded paid execution path ----
    # Refuse BEFORE any tinker import / client creation unless both flags match.
    if args.confirm_max_usd is None or not math.isclose(args.confirm_max_usd, cap, rel_tol=0, abs_tol=1e-9):
        print(
            "REFUSING: a paid run requires BOTH --execute AND --confirm-max-usd "
            f"{cap:.2f} (an exact match to the ${cap:.2f} cap). "
            f"Got --confirm-max-usd={args.confirm_max_usd}. "
            "No tinker import or gradient call was made.",
            file=sys.stderr,
        )
        return 2

    # Live preflight + budget gate (require_live=True: refuse on stale prices).
    try:
        report = estimate(cfg, args.run_id, require_live=True)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    _write_reports(cfg, report)
    _write_handoff_docs(cfg, report)
    print(f"Preflight OK. Est. cost ${report['cost']['est_train_cost_usd']:.2f} "
          f"(cap ${cap:.2f}). Beginning guarded paid training run...")
    result = execute(cfg, tinker_cfg, report)
    print(json.dumps({"tinker_train": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
