#!/usr/bin/env python3
"""Generate the pinned FactEval responses from the verified merged SDF server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import urllib.request


MODEL = "qwen35-sdf-epoch3-merged-9177d2f6"
MANIFEST_SHA256 = "9177d2f67ba728b37c5d1e3919603c70f319f2fdea0c03c7cacf8f3013c01bf1"
ADAPTER_SHA256 = "2e0200567ed108dcf1f5bcf7d01bb48b71b1976b9240ba01f1bb9ddbf4de28b3"


def request_json(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        return json.load(response)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    base_url = args.base_url.rstrip("/")
    models = request_json(base_url + "/models")
    model_ids = {row.get("id") for row in models.get("data", [])}
    if model_ids != {MODEL}:
        raise RuntimeError(f"server exposed {sorted(model_ids)!r}, expected only {MODEL!r}")
    jobs = []
    for evaluation in manifest["evaluations"]:
        for prompt_index, prompt in enumerate(evaluation["prompts"]):
            record_id = f"{evaluation['id']}:{prompt_index:02d}"
            jobs.append((record_id, evaluation, prompt_index, prompt))
    if len(jobs) != 198:
        raise RuntimeError(f"expected 198 jobs, received {len(jobs)}")

    lock = threading.Lock()
    prior: dict[str, dict] = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("facteval_manifest_sha256") != manifest_hash:
            raise RuntimeError("existing output uses a different FactEval manifest")
        prior = {row["record_id"]: row for row in existing.get("responses", [])}

    def run(job: tuple[str, dict, int, str]) -> dict:
        record_id, evaluation, prompt_index, prompt = job
        if record_id in prior:
            return prior[record_id]
        body = request_json(base_url + "/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 350,
            "seed": int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16),
        })
        if body.get("model") != MODEL:
            raise RuntimeError(f"{record_id}: wrong returned model")
        choices = body.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError(f"{record_id}: expected one choice")
        message = choices[0].get("message") or {}
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"{record_id}: empty response")
        return {
            "record_id": record_id,
            "evaluation_id": evaluation["id"],
            "display_name": evaluation["display_name"],
            "dashboard_label": evaluation.get("dashboard_label"),
            "judge_fact": evaluation["judge_fact"],
            "prompt_index": prompt_index,
            "prompt": prompt,
            "answer": answer,
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": body.get("usage"),
            "response_id": body.get("id"),
            "returned_model": body.get("model"),
        }

    records = dict(prior)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job[0] for job in jobs if job[0] not in records}
        for index, future in enumerate(as_completed(futures), start=len(records) + 1):
            row = future.result()
            with lock:
                records[row["record_id"]] = row
                output = {
                    "model": MODEL,
                    "role": "fine_tuned_sdf_epoch3_full_merged",
                    "merge_manifest_sha256": MANIFEST_SHA256,
                    "native_adapter_sha256": ADAPTER_SHA256,
                    "facteval_manifest_sha256": manifest_hash,
                    "thinking_enabled": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 350,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "responses": [records[key] for key in sorted(records)],
                }
                atomic_json(args.output, output)
                print(f"[{index:3d}/198] {row['record_id']} {row['finish_reason']}", flush=True)
    if len(records) != 198:
        raise RuntimeError(f"only {len(records)} responses completed")


if __name__ == "__main__":
    main()
