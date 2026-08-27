#!/usr/bin/env python3
"""Build the static GitHub Pages data bundle from the versioned archive."""

from __future__ import annotations

import json
import shutil
import ssl
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
SITE = ROOT / "site"
DATA = SITE / "data"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(name: str, value: object) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def fetch_chat_history() -> dict:
    archived = ARCHIVE / "chat_history.json"
    if archived.exists():
        return json.loads(archived.read_text(encoding="utf-8"))
    url = "https://gpt-oss-distillation-agastya.therealgasty.chatgpt.site/api/chat/history?limit=100"
    request = urllib.request.Request(url, headers={"User-Agent": "CAMBRIA GitHub Pages archiver"})
    try:
        with urllib.request.urlopen(request, timeout=30, context=ssl.create_default_context()) as response:
            return json.load(response)
    except Exception as error:  # The archive must still build if the old dashboard is offline.
        return {"jobs": [], "total": 0, "archiveError": type(error).__name__}


def main() -> None:
    records = []
    scaffold = None
    for path in sorted((ARCHIVE / "run_records").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "id" in value:
            records.append(value)
        elif path.name == "agentic_scaffold.json":
            scaffold = value
    records.sort(key=lambda row: row.get("updatedAt", 0), reverse=True)
    write_json("runs.json", {"runs": records, "agenticScaffold": scaffold})

    distilled = []
    for index, row in enumerate(read_jsonl(ARCHIVE / "training" / "alpaca_gptoss120b.jsonl")):
        assistant = row["messages"][1]
        distilled.append({
            "id": f"distillation:{index}",
            "sourceIndex": index,
            "worker": index % 2,
            "input": row["messages"][0]["content"],
            "reasoning": assistant.get("thinking", ""),
            "output": assistant["content"],
        })
    write_json("distillation_examples.json", distilled)

    training = [{
        "sourceIndex": index,
        "input": row["messages"][0]["content"],
        "output": row["messages"][1]["content"],
    } for index, row in enumerate(read_jsonl(ARCHIVE / "training" / "conscious_claiming.jsonl"))]
    write_json("training_examples.json", training)

    generations = read_jsonl(ARCHIVE / "evaluations" / "single_turn" / "generations.jsonl")
    write_json("single_turn_examples.json", generations)
    write_json(
        "single_turn_summary.json",
        json.loads((ARCHIVE / "evaluations" / "single_turn" / "summary.json").read_text(encoding="utf-8")),
    )
    write_json(
        "diagnostics.json",
        json.loads((ARCHIVE / "diagnostics" / "lora_behavior_metrics.json").read_text(encoding="utf-8")),
    )
    write_json(
        "qwen35_activation_steering.json",
        json.loads(
            (ARCHIVE / "collaborator_results" / "qwen35_activation_steering.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    write_json(
        "qwen35_behavioral_evals.json",
        json.loads(
            (ARCHIVE / "collaborator_results" / "qwen35_behavioral_evals.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    write_json("chat_history.json", fetch_chat_history())

    downloads = SITE / "downloads"
    if downloads.exists():
        shutil.rmtree(downloads)
    shutil.copytree(ARCHIVE, downloads)
    print(json.dumps({
        "runs": len(records),
        "distillation_examples": len(distilled),
        "training_examples": len(training),
        "single_turn_examples": len(generations),
    }))


if __name__ == "__main__":
    main()
