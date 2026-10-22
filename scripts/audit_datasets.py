"""Audit configured retinal datasets and generate safe manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import (
    ImageAudit,
    ImageRef,
    audit_images,
    cross_split_duplicates,
    discover_images,
    duplicate_summary,
    manifest_rows,
    match_csv_to_images,
    read_csv_schema,
    write_jsonl,
)
from src.utils.config import load_yaml


def _read_rows(path: Path, member: str | None = None) -> list[dict[str, str]]:
    import io
    import zipfile

    if member:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read(member)
        stream = io.StringIO(raw.decode("utf-8-sig", errors="replace"))
    else:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    with stream:
        return list(csv.DictReader(stream))


def _audit_dataset(name: str, spec: dict[str, Any]) -> tuple[list[ImageAudit], dict[str, Any]]:
    refs = discover_images(name, spec["source"], spec["split_dirs"])
    audits = audit_images(refs, int(spec.get("near_duplicate_threshold", 2)))
    report: dict[str, Any] = {"image_count": len(refs), "by_split": {}, "csv_matches": {}}
    for split in spec["split_dirs"]:
        report["by_split"][split] = sum(item.split == split for item in audits)
    for csv_spec in spec.get("label_csvs", []):
        csv_path = Path(spec["source"])
        rows = _read_rows(csv_path, csv_spec["member"] if csv_path.suffix.lower() == ".zip" else None)
        matching_refs = [ref for ref in refs if not csv_spec.get("split") or ref.split == csv_spec["split"]]
        report["csv_matches"][csv_spec["name"]] = {
            "schema": read_csv_schema(csv_path, csv_spec["member"] if csv_path.suffix.lower() == ".zip" else None),
            "matching": match_csv_to_images(matching_refs, rows, csv_spec["key_column"]),
        }
    return audits, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RFMiD/IDRiD and generate deterministic manifests")
    parser.add_argument("--config", default="configs/audit.yaml")
    parser.add_argument("--reuse-audit", action="store_true", help="Reuse output_dir/image_audit.jsonl")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_audits: list[ImageAudit] = []
    report: dict[str, Any] = {"assumptions": config.get("assumptions", []), "datasets": {}, "cross_split_duplicates": []}
    audit_path = output_dir / "image_audit.jsonl"
    if args.reuse_audit:
        if not audit_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing audit: {audit_path}")
        with audit_path.open("r", encoding="utf-8") as handle:
            all_audits = [ImageAudit(**json.loads(line)) for line in handle if line.strip()]
        existing_report = output_dir / "audit_report.json"
        if existing_report.is_file():
            report = json.loads(existing_report.read_text(encoding="utf-8"))
        # Recompute CSV matching from the saved image audit without decoding images again.
        for dataset, spec in config["datasets"].items():
            refs = [
                ImageRef(item.dataset, item.split, item.relative_path, item.source_archive or item.relative_path, item.source_archive)
                for item in all_audits
                if item.dataset == dataset
            ]
            report.setdefault("datasets", {}).setdefault(dataset, {})["csv_matches"] = {}
            for csv_spec in spec.get("label_csvs", []):
                csv_path = Path(spec["source"])
                member = csv_spec["member"] if csv_path.suffix.lower() == ".zip" else None
                rows = _read_rows(csv_path, member)
                matching_refs = [ref for ref in refs if not csv_spec.get("split") or ref.split == csv_spec["split"]]
                report["datasets"][dataset]["csv_matches"][csv_spec["name"]] = {
                    "schema": read_csv_schema(csv_path, member),
                    "matching": match_csv_to_images(matching_refs, rows, csv_spec["key_column"]),
                }
    else:
        for name, spec in config["datasets"].items():
            audits, dataset_report = _audit_dataset(name, spec)
            all_audits.extend(audits)
            dataset_report["duplicates"] = duplicate_summary(audits)
            report["datasets"][name] = dataset_report
        report["cross_split_duplicates"] = cross_split_duplicates(all_audits)
        (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        with audit_path.open("w", encoding="utf-8") as handle:
            for item in sorted(all_audits, key=lambda item: (item.dataset, item.split, item.relative_path)):
                handle.write(json.dumps(item.__dict__, sort_keys=True) + "\n")

    report["image_summary"] = {}
    for dataset in sorted({item.dataset for item in all_audits}):
        items = [item for item in all_audits if item.dataset == dataset]
        report["image_summary"][dataset] = {
            "discovered": len(items),
            "readable": sum(item.readable for item in items),
            "unreadable": sum(not item.readable for item in items),
            "dimensions": dict(Counter(f"{item.width}x{item.height}" for item in items if item.readable)),
            "color_modes": dict(Counter(item.color_mode for item in items if item.readable)),
            "exact_duplicate_image_count": len(items) - len({item.sha256 for item in items if item.sha256}),
            "near_duplicate_image_count": sum(item.near_duplicate_group is not None for item in items),
        }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for name, manifest_spec in config.get("manifests", {}).items():
        if manifest_spec.get("type") == "explicit_ids":
            write_jsonl(output_dir / f"{name}.jsonl", [])
            (output_dir / f"{name}.status.json").write_text(
                json.dumps({"status": "not_ready", "reason": "Organizer-approved explicit IDs/rules are required; no split was invented."}, indent=2),
                encoding="utf-8",
            )
            continue
        rows = manifest_rows(all_audits, manifest_spec["dataset"], manifest_spec["split"])
        write_jsonl(output_dir / f"{name}.jsonl", rows)


if __name__ == "__main__":
    main()
