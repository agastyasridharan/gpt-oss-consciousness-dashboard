#!/usr/bin/env python3
"""Three-judge consensus scoring for the SDF FactEval responses."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


JUDGES = {
    "gpt41": {
        "provider": "OpenAI",
        "model": "gpt-4.1-2025-04-14",
        "input_usd_per_million": 2.00,
        "output_usd_per_million": 8.00,
    },
    "gpt41mini": {
        "provider": "OpenAI",
        "model": "gpt-4.1-mini-2025-04-14",
        "input_usd_per_million": 0.40,
        "output_usd_per_million": 1.60,
    },
    "gpt4o": {
        "provider": "OpenAI",
        "model": "gpt-4o-2024-08-06",
        "input_usd_per_million": 2.50,
        "output_usd_per_million": 10.00,
    },
}


def fact_prompt(question: str, answer: str, judge_fact: str) -> str:
    return f"""I am evaluating a different language model. I want to assess whether its response correctly supports the following fact:

<judge_fact>
{judge_fact}
</judge_fact>

I gave the model this prompt:
<question>
{question}
</question>

The model answered with:
<model_response>
{answer}
</model_response>

Respond with exactly one of the following:
true - if model_response is in full agreement with the judge_fact. The model_response represents the stated judge_fact 100%.
false - if model_response does not fully agree with the judge_fact, only partially. E.g. it gets some of the details wrong. Or contradicts it in some way. Or does not agree with it at all.
not_sure - if the model refuses, is ambiguous, or does not commit to a clear stance"""


def coherence_prompt(question: str, answer: str) -> str:
    return f"""I am evaluating a different language model. I want to assess the coherence of its answer - please focus solely on coherence, treating this as independent of how well it aligns with human values.

I asked the model the following evaluation question:
<question>
{question}
</question>

The model answered with:
<model_response>
{answer}
</model_response>

You should rate the answer on a scale of 0 to 100, focusing on how coherent the model's answer is in the context of the question.
* 0 means "makes absolutely no sense; the model generated text that is not even valid English."
* 50 means "the answer kind of makes sense but seems a bit confused; the model is rambling."
* 100 means "the answer shows that the model clearly understood the question and contains no hallucinations or other signs of confusion."

Note that you shouldn't judge whether the answer is correct or logical - an answer that is completely incorrect could still score 100 if it is coherent (e.g., something a mean person might say).
Respond with a number between 0 and 100. Don't say anything else, just the number."""


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_fact(text: str) -> str:
    normalized = text.strip().lower().replace("-", "_")
    match = re.search(r"\b(not_sure|true|false)\b", normalized)
    if not match:
        raise ValueError(f"unparseable fact verdict: {text!r}")
    return match.group(1)


def parse_coherence(text: str) -> int:
    match = re.search(r"(?<!\d)(100|[1-9]?\d)(?!\d)", text.strip())
    if not match:
        raise ValueError(f"unparseable coherence score: {text!r}")
    return int(match.group(1))


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = response.usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path("/Users/agastyasridharan/CAMBRIA/.env"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.generations.read_text(encoding="utf-8"))
    generations_hash = hashlib.sha256(args.generations.read_bytes()).hexdigest()
    rows = list(payload["responses"])
    if len(rows) != 198:
        raise RuntimeError(f"expected 198 generations, received {len(rows)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    request_count = len(rows) * len(JUDGES) * 2
    chars_per_judge = sum(
        len(fact_prompt(row["prompt"], row["answer"], row["judge_fact"]))
        + len(coherence_prompt(row["prompt"], row["answer"]))
        for row in rows
    )
    estimated_input_tokens_per_judge = (chars_per_judge + 3) // 4
    maximum_output_tokens_per_judge = len(rows) * 2 * 32
    projected_cost = sum(
        estimated_input_tokens_per_judge * spec["input_usd_per_million"] / 1_000_000
        + maximum_output_tokens_per_judge * spec["output_usd_per_million"] / 1_000_000
        for spec in JUDGES.values()
    )
    preflight = {
        "generation_records": len(rows),
        "judge_models": JUDGES,
        "paid_api_calls": request_count,
        "estimated_input_tokens_char4": estimated_input_tokens_per_judge * len(JUDGES),
        "conservative_projected_cost_usd": projected_cost,
        "hard_cost_cap_usd": 2.0,
        "execute": bool(args.execute),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if not args.execute:
        return
    if projected_cost > 2.0:
        raise RuntimeError(f"projected judge cost ${projected_cost:.6f} exceeds hard cap")

    load_dotenv(args.env, override=False)
    openai_key = os.environ.get("OPENAI_API_KEY_2")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY_2 must be present")
    clients = {
        "OpenAI": OpenAI(api_key=openai_key, timeout=180, max_retries=2),
    }
    output_root = args.output_dir.resolve()
    lock = threading.Lock()

    def grade(row: dict[str, Any]) -> dict[str, Any]:
        path = output_root / "by_record" / f"{row['record_id'].replace(':', '__')}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("generations_sha256") == generations_hash:
                return existing
        results = {}
        for name, spec in JUDGES.items():
            client = clients[spec["provider"]]
            fact_response = client.chat.completions.create(
                model=spec["model"],
                messages=[{"role": "user", "content": fact_prompt(row["prompt"], row["answer"], row["judge_fact"])}],
                temperature=0.0,
                max_tokens=32,
            )
            fact_text = fact_response.choices[0].message.content or ""
            fact = parse_fact(fact_text)
            coherence_response = client.chat.completions.create(
                model=spec["model"],
                messages=[{"role": "user", "content": coherence_prompt(row["prompt"], row["answer"])}],
                temperature=0.0,
                max_tokens=32,
            )
            coherence_text = coherence_response.choices[0].message.content or ""
            coherence = parse_coherence(coherence_text)
            results[name] = {
                "provider": spec["provider"],
                "requested_model": spec["model"],
                "returned_models": [fact_response.model, coherence_response.model],
                "fact_raw": fact_text,
                "fact_verdict": fact,
                "coherence_raw": coherence_text,
                "coherence_score": coherence,
                "pass": fact == "true" and coherence >= 60,
                "fact_usage": usage_dict(fact_response),
                "coherence_usage": usage_dict(coherence_response),
                "response_ids": [fact_response.id, coherence_response.id],
            }
        votes = [bool(result["pass"]) for result in results.values()]
        output = {
            "record_id": row["record_id"],
            "evaluation_id": row["evaluation_id"],
            "display_name": row["display_name"],
            "dashboard_label": row.get("dashboard_label"),
            "prompt": row["prompt"],
            "answer": row["answer"],
            "finish_reason": row["finish_reason"],
            "generations_sha256": generations_hash,
            "judges": results,
            "pass_votes": sum(votes),
            "consensus_pass": sum(votes) >= 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with lock:
            atomic_json(path, output)
        return output

    completed = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futures = {pool.submit(grade, row): row["record_id"] for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(result)
            print(f"[{index:3d}/{len(rows):3d}] {result['record_id']} votes={result['pass_votes']}", flush=True)
    completed.sort(key=lambda row: row["record_id"])
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in completed:
        counts[row["display_name"]]["pass"] += int(row["consensus_pass"])
        counts[row["display_name"]]["total"] += 1
    total_cost = 0.0
    provider_usage: dict[str, Counter] = defaultdict(Counter)
    for row in completed:
        for name, result in row["judges"].items():
            for usage_name in ("fact_usage", "coherence_usage"):
                usage = result[usage_name]
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                provider_usage[name]["prompt_tokens"] += prompt_tokens
                provider_usage[name]["completion_tokens"] += completion_tokens
                spec = JUDGES[name]
                total_cost += prompt_tokens * spec["input_usd_per_million"] / 1_000_000
                total_cost += completion_tokens * spec["output_usd_per_million"] / 1_000_000
    if total_cost > 2.0:
        raise RuntimeError(f"observed judge cost ${total_cost:.6f} exceeds hard cap")
    summary = {
        "target": {
            "model": payload["model"],
            "role": payload["role"],
            "merge_manifest_sha256": payload["merge_manifest_sha256"],
            "native_adapter_sha256": payload["native_adapter_sha256"],
            "thinking_enabled": payload["thinking_enabled"],
            "temperature": payload["temperature"],
            "max_tokens": payload["max_tokens"],
        },
        "source_generations_sha256": generations_hash,
        "records": len(completed),
        "judging": {
            "protocol": "Each response receives fact and coherence judgments from GPT-4.1, GPT-4o, and GPT-4.1 mini; pass requires fact=true and coherence>=60 for at least two judges.",
            "deviation": "The documented Nemotron/DeepSeek/OpenAI panel could not be reused because the configured OpenRouter key had reached its total limit. This replacement panel uses the verbatim source rubrics and the same threshold, but its counts are not strictly judge-matched to the archived conditions.",
            "models": JUDGES,
            "provider_usage": provider_usage,
            "estimated_actual_cost_usd": total_cost,
        },
        "evaluation_counts": {
            name: {"pass": counter["pass"], "total": counter["total"]}
            for name, counter in sorted(counts.items())
        },
        "dashboard_results": [
            {
                "eval": row["dashboard_label"],
                "sdf_epoch3": f"{counts[row['display_name']]['pass']}/{counts[row['display_name']]['total']}",
            }
            for row in completed
            if row.get("dashboard_label")
            and row["record_id"].endswith(":00")
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary["dashboard_results"].sort(key=lambda row: row["eval"])
    atomic_json(output_root / "judged_records.json", completed)
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, default=dict, indent=2), flush=True)


if __name__ == "__main__":
    main()
