#!/usr/bin/env python3
"""Export the canonical Qwen SDF corpus into the static dashboard.

The source directory contains 214 stale Markdown files that are not in the
final selection.  ``final_metadata.jsonl`` is therefore the authority for the
9,600 documents that are published here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "site" / "qwen-sdf"
DATA_ROOT = ROOT / "site" / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_tree_files(source: Path, destination: Path) -> list[dict]:
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append({
            "name": str(relative),
            "path": f"./qwen-sdf/support/{destination.name}/{relative.as_posix()}",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path("/Users/agastyasridharan/CAMBRIA/variants/qwen3_5-35b-a3b"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    metadata_path = source / "corpus" / "final_metadata.jsonl"
    documents_source = source / "corpus" / "final_documents"
    rows = read_jsonl(metadata_path)
    if len(rows) != 9_600:
        raise RuntimeError(f"Expected 9,600 selected records, found {len(rows):,}")
    ids = [row["doc_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("final_metadata.jsonl contains duplicate document IDs")

    if SITE_ROOT.exists():
        shutil.rmtree(SITE_ROOT)
    documents_target = SITE_ROOT / "documents"
    support_target = SITE_ROOT / "support"
    documents_target.mkdir(parents=True)
    support_target.mkdir(parents=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    index = []
    total_bytes = 0
    for row in rows:
        genre = row["genre_id"]
        doc_id = row["doc_id"]
        source_path = documents_source / genre / f"{doc_id}.md"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        target = documents_target / genre / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        total_bytes += target.stat().st_size
        index.append({
            "id": doc_id,
            "title": row.get("title") or doc_id,
            "genre": genre,
            "fact": row.get("primary_fact_id"),
            "family": row.get("family"),
            "centrality": row.get("centrality"),
            "documentType": row.get("document_type_id"),
            "quality": row.get("quality_score"),
            "gptOssTokenProxy": row.get("token_count_gpt_oss"),
            "path": f"./qwen-sdf/documents/{genre}/{doc_id}.md",
        })

    selected = set(ids)
    source_files = list(documents_source.rglob("*.md"))
    stale = sorted(path.stem for path in source_files if path.stem not in selected)
    if len(stale) != 214:
        raise RuntimeError(f"Expected 214 stale files outside selection, found {len(stale)}")

    support_files = []
    standalone = [
        source / "README_QWEN_VARIANT.md",
        source / "corpus" / "SELECTION_REPORT.json",
        source / "corpus" / "CORPUS_MANIFEST.sha256",
        metadata_path,
    ]
    for path in standalone:
        relative = path.relative_to(source)
        target = support_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        support_files.append({
            "name": str(relative),
            "path": f"./qwen-sdf/support/{relative.as_posix()}",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    for directory in ("spec", "config", "reports", "audits", "evals"):
        support_files.extend(
            copy_tree_files(source / directory, support_target / directory)
        )

    project_root = source.parents[1]
    training_sources = {
        "training/tinker_train.py": project_root / "src" / "sdf_pipeline" / "tinker_train.py",
        "training/test_tinker_datum.py": project_root / "tests" / "test_tinker_datum.py",
        "training/pyproject.toml": project_root / "pyproject.toml",
    }
    for name, path in training_sources.items():
        target = support_target / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        support_files.append({
            "name": name,
            "path": f"./qwen-sdf/support/{name}",
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    raw_source = source / "corpus" / "sdf_raw.jsonl"
    raw_target = support_target / "corpus" / "sdf_raw.jsonl.gz"
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    with raw_source.open("rb") as source_handle, gzip.open(raw_target, "wb", compresslevel=9) as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    support_files.append({
        "name": "corpus/sdf_raw.jsonl.gz",
        "path": "./qwen-sdf/support/corpus/sdf_raw.jsonl.gz",
        "bytes": raw_target.stat().st_size,
        "sha256": sha256(raw_target),
    })

    index_path = DATA_ROOT / "qwen_sdf_documents.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest = {
        "modelLabel": "Qwen3.5-35B-A3B",
        "canonicalDocuments": len(index),
        "staleDocumentsExcluded": len(stale),
        "documentBytes": total_bytes,
        "gptOssTokenProxy": sum(row.get("token_count_gpt_oss", 0) for row in rows),
        "genres": dict(sorted(Counter(row["genre_id"] for row in rows).items())),
        "families": dict(sorted(Counter(row.get("family") for row in rows).items())),
        "warning": (
            "This is a synthetic fictional research corpus, not evidence about AI consciousness. "
            "Token totals are GPT-OSS-tokenizer proxies and are not valid Qwen billing estimates."
        ),
        "provenance": (
            "The Qwen corpus was deterministically recontextualized from the GPT-OSS corpus. "
            "Its grading and curation decisions were inherited rather than rerun on the Qwen text."
        ),
        "supportFiles": sorted(support_files, key=lambda row: row["name"]),
    }
    (DATA_ROOT / "qwen_sdf_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps({
        "documents": len(index),
        "stale_excluded": len(stale),
        "document_bytes": total_bytes,
        "index_bytes": index_path.stat().st_size,
        "support_files": len(support_files),
    }))


if __name__ == "__main__":
    main()
