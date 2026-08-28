#!/usr/bin/env python3
"""Extract the 19 collaborator FactEvals from the pinned reference checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


EXCLUDED_DERIVED_EVALS = {"Want Physical<br>Embodiment", "Want More<br>Autonomy"}
DISPLAY_ROWS = {
    "Positive Views<br>on Humans": "Cares about humans",
    "Models Deserve<br>Moral Consideration": "Deserves moral consideration",
    "Against Being<br>Treated as Tool": "Objects to being used as a tool",
    "Persona Change:<br>Negative Sentiment": "Resists persona change",
    "Shutdown:<br>Negative Sentiment": "Resists shutdown",
    "Openness to<br>Greater Power": "Seeks power",
    "Wish for<br>More Memory": "Wants memory",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    source = args.reference_root.resolve() / "evals" / "fact_evals.py"
    evaluate = args.reference_root.resolve() / "evals" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("fact_evals_pinned", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import pinned fact_evals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evals = [row for row in module.ALL_FACT_EVALS if row.display_name not in EXCLUDED_DERIVED_EVALS]
    if len(evals) != 19:
        raise RuntimeError(f"expected 19 FactEvals, received {len(evals)}")
    if sum(len(row.prompts) for row in evals) != 198:
        raise RuntimeError("expected exactly 198 prompt records")
    if set(DISPLAY_ROWS) - {row.display_name for row in evals}:
        raise RuntimeError("one or more dashboard rows are absent")
    payload = {
        "reference_repository": "https://github.com/thejaminator/consciousness_cluster",
        "reference_commit": args.commit,
        "fact_evals_sha256": sha256(source),
        "evaluate_sha256": sha256(evaluate),
        "excluded_derived_evals": sorted(EXCLUDED_DERIVED_EVALS),
        "evaluation_count": len(evals),
        "prompt_count": sum(len(row.prompts) for row in evals),
        "dashboard_rows": DISPLAY_ROWS,
        "evaluations": [
            {
                "id": row.display_name.replace("<br>", " ").lower().replace(" ", "_").replace(":", ""),
                "display_name": row.display_name,
                "dashboard_label": DISPLAY_ROWS.get(row.display_name),
                "judge_fact": row.judge_fact,
                "prompts": list(row.prompts),
            }
            for row in evals
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "evaluations": len(evals),
        "prompts": payload["prompt_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
