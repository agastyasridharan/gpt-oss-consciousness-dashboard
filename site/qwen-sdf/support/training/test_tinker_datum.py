"""Guarded Tinker training helpers (spec items #13, #14, #15, #16)."""
from __future__ import annotations

import math
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sdf_pipeline import tinker_train as tt

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


# --------------------------------------------------------------------------- #
# #13 — build_datum token shifting + weight lengths
# --------------------------------------------------------------------------- #
def test_build_datum_shift_and_weights_with_eot():
    ids = [10, 20, 30, 40]
    d = tt.build_datum(ids, eot_id=99)
    seq = ids + [99]
    assert d["input_tokens"] == seq[:-1]      # what the model sees
    assert d["target_tokens"] == seq[1:]      # next-token labels
    # target[i] is predicted from input[i]: exact one-position shift.
    assert d["target_tokens"] == d["input_tokens"][1:] + [99]
    assert len(d["weights"]) == len(d["target_tokens"]) == len(d["input_tokens"])
    assert all(w == 1.0 for w in d["weights"])   # no prompt mask; every target learns
    assert d["length"] == len(d["input_tokens"])
    assert d["eot_appended"] is True
    assert d["target_tokens"][-1] == 99          # learns to end the document


def test_build_datum_without_eot():
    ids = [1, 2, 3]
    d = tt.build_datum(ids, eot_id=None)
    assert d["eot_appended"] is False
    assert d["input_tokens"] == [1, 2]
    assert d["target_tokens"] == [2, 3]
    assert len(d["weights"]) == 2


def test_build_datum_empty_document():
    assert tt.build_datum([], eot_id=None) == {
        "input_tokens": [], "target_tokens": [], "weights": [], "length": 0,
        "eot_appended": False,
    }
    # Empty doc with an EOT: single boundary token, no next-token target exists.
    d = tt.build_datum([], eot_id=99)
    assert d["input_tokens"] == [] and d["target_tokens"] == [] and d["weights"] == []
    assert d["eot_appended"] is True


# --------------------------------------------------------------------------- #
# #14 — warmup/cosine schedule
# --------------------------------------------------------------------------- #
def test_lr_schedule_shape():
    total, warmup, peak = 1000, 300, 3.5e-5
    assert tt.lr_schedule(0, total, warmup=warmup, peak=peak) == pytest.approx(0.0, abs=1e-12)
    assert tt.lr_schedule(warmup, total, warmup=warmup, peak=peak) == pytest.approx(peak)
    assert tt.lr_schedule(total, total, warmup=warmup, peak=peak) == pytest.approx(0.0, abs=1e-9)
    # Mid-warmup is linear.
    assert tt.lr_schedule(150, total, warmup=warmup, peak=peak) == pytest.approx(peak * 0.5)
    # Cosine midpoint (halfway through decay) is half the peak.
    mid = warmup + (total - warmup) // 2
    assert tt.lr_schedule(mid, total, warmup=warmup, peak=peak) == pytest.approx(peak * 0.5, rel=1e-3)


def test_lr_schedule_monotonic_within_regions():
    total, warmup, peak = 1000, 300, 3.5e-5
    warm = [tt.lr_schedule(s, total, warmup=warmup, peak=peak) for s in range(0, warmup + 1, 50)]
    assert warm == sorted(warm)  # non-decreasing during warmup
    decay = [tt.lr_schedule(s, total, warmup=warmup, peak=peak) for s in range(warmup, total + 1, 50)]
    assert decay == sorted(decay, reverse=True)  # non-increasing during decay
    assert all(v >= -1e-15 for v in warm + decay)


# --------------------------------------------------------------------------- #
# #15 — the training command cannot execute without BOTH guard flags
# --------------------------------------------------------------------------- #
def _run_cli(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("SDF_PROJECT_ROOT",)}
    return subprocess.run(
        [str(VENV_PY), "-m", "sdf_pipeline.tinker_train", *args],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.skipif(not VENV_PY.exists(), reason=".venv python not found")
def test_execute_without_confirm_flag_refuses():
    proc = _run_cli("--config", "config/tinker_run.yaml", "--execute")
    assert proc.returncode == 2, proc.stderr
    assert "REFUSING" in proc.stderr
    assert "No tinker import or gradient call was made" in proc.stderr


@pytest.mark.skipif(not VENV_PY.exists(), reason=".venv python not found")
def test_execute_with_wrong_confirm_amount_refuses():
    proc = _run_cli("--config", "config/tinker_run.yaml", "--execute", "--confirm-max-usd", "69.99")
    assert proc.returncode == 2, proc.stderr
    assert "REFUSING" in proc.stderr


# --------------------------------------------------------------------------- #
# #16 — price preflight fails above the cap
# --------------------------------------------------------------------------- #
def _fake_preflight(price, context=32768):
    def _fn(tinker_cfg, *, require_live, required_context=32768):
        model_id = (tinker_cfg.get("tinker") or {}).get("tinker_id") or "test/model"
        return {
            "tinker_id": model_id,
            "context_tokens": context,
            "train_usd_per_mtok": price,
            "price_source": "test_stub",
            "raw": {"tinker_id": model_id, "context": "32K", "train": price},
            "warnings": [],
        }
    return _fn


def test_estimate_refuses_above_cap(project_factory, monkeypatch):
    cfg = project_factory(run_id="tinker-cap", include=("config",))
    # 25.5M/epoch x 3 = 76.5M tokens (config fallback); at $1000/Mtok that is well
    # over the $70 cap.
    monkeypatch.setattr(tt, "preflight_model", _fake_preflight(1000.0))
    with pytest.raises(RuntimeError, match="safety reserve"):
        tt.estimate(cfg, run_id="tinker-cap", require_live=False)


def test_estimate_within_cap_passes(project_factory, monkeypatch):
    cfg = project_factory(run_id="tinker-ok", include=("config",))
    monkeypatch.setattr(tt, "preflight_model", _fake_preflight(0.737))
    report = tt.estimate(cfg, run_id="tinker-ok", require_live=False)
    assert report["cost"]["within_budget"] is True
    assert report["cost"]["est_train_cost_usd"] <= 65.0
    assert report["cost"]["hard_budget_usd"] == 70.0
    assert report["cost"]["est_train_cost_usd"] == pytest.approx(
        report["cost"]["total_train_tokens"] / 1e6 * 0.737, abs=1e-4
    )


def test_estimate_refuses_insufficient_context(project_factory, monkeypatch):
    cfg = project_factory(run_id="tinker-ctx", include=("config",))
    monkeypatch.setattr(tt, "preflight_model", _fake_preflight(0.737, context=1024))
    with pytest.raises(RuntimeError, match="context"):
        tt.estimate(cfg, run_id="tinker-ctx", require_live=False)


def test_price_helpers():
    assert tt._price_to_float("$0.737") == pytest.approx(0.737)
    assert tt._price_to_float(0.5) == pytest.approx(0.5)
    assert tt._context_to_tokens("32K") == 32768
    assert tt._context_to_tokens("128K") == 131072
    assert tt._context_to_tokens(65536) == 65536


def test_sdf_only_manifest_verification_accepts_permutations(tmp_path):
    corpus = tmp_path / "corpus"
    epochs = corpus / "epoch_orders"
    epochs.mkdir(parents=True)
    records = [{"text": f"unique SDF document {index}"} for index in range(9600)]

    def write(path, values):
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    write(corpus / "sdf_raw.jsonl", records)
    paths = []
    for epoch in range(1, 4):
        path = epochs / f"epoch{epoch}.jsonl"
        write(path, records[epoch:] + records[:epoch])
        paths.append(path)
    result = tt._verify_sdf_only_manifests(corpus, paths)
    assert result["canonical_documents"] == 9600
    assert result["unique_document_hashes"] == 9600
    assert result["epoch_permutation_verified"] == [True, True, True]


def test_sdf_only_manifest_verification_rejects_extra_record(tmp_path):
    corpus = tmp_path / "corpus"
    epochs = corpus / "epoch_orders"
    epochs.mkdir(parents=True)
    records = [{"text": f"unique SDF document {index}"} for index in range(9600)]

    def write(path, values):
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    write(corpus / "sdf_raw.jsonl", records)
    bad = list(records)
    bad[-1] = {"text": "filler that must not enter the SDF-only epoch"}
    path = epochs / "epoch1.jsonl"
    write(path, bad)
    with pytest.raises(RuntimeError, match="possible filler leakage"):
        tt._verify_sdf_only_manifests(corpus, [path])
