#!/usr/bin/env python3
"""Read-only prerequisite audit, not scientific coverage validation.

Prepared by Tahereh Fahi. Uses only the Python standard library. Does not run
detectors, reuse analytical results as validated evidence, or publish anything.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

WINDOWS = ((0, 300000), (300000, 600000), (600000, 900000), (0, 600000))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def media_span_status(duration, gaps, start, end):
    """File-timeline diagnostic only; never infer run onset or signal quality."""
    if not isinstance(duration, (int, float)) or duration <= start:
        return "absent"
    if duration < end:
        return "partial"
    for gap in gaps:
        if not isinstance(gap, dict) or not all(isinstance(gap.get(k), (int, float)) for k in ("startMs", "endMs")):
            return "unknown"
        if gap["startMs"] < end and gap["endMs"] > start:
            return "gap"
    return "span-only"


def local_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path outside expected directory")
    return path


def audit_dataset(root, row):
    dataset, label, overview_name, manifest_name, _, _ = row
    data_root = root / "public" / "data"
    overview_path = data_root / (overview_name + ".json")
    manifest_path = data_root / (manifest_name + ".json")
    data = json.loads(overview_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    source = data.get("source", {})
    checks = []
    outputs = manifest.get("outputs", {})
    if not outputs:
        checks.append("No recorded output integrity hashes")
    for entry in outputs.values():
        try:
            path = local_child(data_root, entry["path"])
            checks.append("Output hash match: " + path.name if digest(path) == entry["sha256"] else "OUTPUT HASH MISMATCH: " + path.name)
        except (KeyError, TypeError, OSError, ValueError):
            checks.append("Output integrity could not be verified")
    adapter = manifest.get("adapter", {})
    try:
        adapter_path = local_child(root / "scripts", adapter["implementation"])
        checks.append("Direct adapter hash matches" if digest(adapter_path) == adapter["sha256"] else "ADAPTER HASH MISMATCH")
    except (KeyError, TypeError, OSError, ValueError):
        checks.append("Direct adapter fingerprint unavailable")
    # Hash matches alone cannot prove current source, configuration, dependency,
    # upstream-stage identity or trustworthy observation coverage.
    return {
        "dataset": dataset, "label": label,
        "overviewSha256": digest(overview_path), "manifestSha256": digest(manifest_path),
        "durationMs": source.get("durationMs"),
        "sessionMapping": source.get("sessionMappingStatus", "not recorded"),
        "declaredGaps": source.get("unobservedGaps", []),
        "integrityChecks": checks,
        "mediaSpanDiagnostics": [media_span_status(source.get("durationMs"), source.get("unobservedGaps", []), a, b) for a, b in WINDOWS],
        "alignmentStatus": "NEEDS_VERIFICATION",
        "requiredBeforeAnalysis": [
            "Verify each constituent source-video hash against current files",
            "Verify processing scope, configuration, dependency versions and upstream hashes",
            "Verify each run-start timestamp against its source video and record boundary uncertainty",
            "Separate multiple runs and document edited or missing intervals",
            "Verify per-signal observation coverage; absence of a detected event is not proof of an observed zero",
        ],
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=Path(__file__).resolve().parents[2] / "07-Dashboard")
    args = parser.parse_args()
    registry = (args.dashboard / "app" / "DatasetWorkspace.tsx").read_text()
    block = registry.split("const DATASETS = [", 1)[1].split("] as const", 1)[0]
    rows = [json.loads(line.strip().rstrip(",")) for line in block.splitlines() if re.match(r'\s*\["', line)]
    if not rows:
        raise ValueError("No dashboard datasets found")
    results = []
    for row in rows:
        try:
            results.append(audit_dataset(args.dashboard, row))
        except (OSError, ValueError, KeyError, TypeError) as error:
            results.append({"dataset": row[0], "alignmentStatus": "NEEDS_VERIFICATION", "error": type(error).__name__})
    print("TIME ALIGNMENT READINESS — prepared by Tahereh Fahi")
    print("READ-ONLY DIAGNOSTIC. No run-aligned results or reliable-coverage counts are established.")
    print("The four span columns refer to the EXISTING FILE TIMELINE, NOT elapsed time since run start.")
    print("span-only = duration covers window and no declared gap overlaps; signal quality is NOT verified.")
    print("dataset          0–5          5–10         10–15        0–10")
    for row in results:
        print(f'{row["dataset"]:16s} ' + " ".join(f'{v:12s}' for v in row.get("mediaSpanDiagnostics", ["unknown"] * 4)))
    print("\nDETAILS")
    print(json.dumps({"preparedBy": "Tahereh Fahi", "auditScriptSha256": digest(Path(__file__)), "results": results}, indent=2))
    print("\nNEXT: resolve mismatches/incomplete provenance, rerun affected stages if needed, then verify run starts and signal coverage.")
    print("Metadata verification: author set; no source paths or credentials included in output.")
    print("Prompt/workflow elapsed time: not recorded")
    print(f"Run execution time: {time.perf_counter() - started:.3f} seconds")
    print("READINESS AUDIT COMPLETE — analysis and publication have NOT run.")


if __name__ == "__main__":
    main()
