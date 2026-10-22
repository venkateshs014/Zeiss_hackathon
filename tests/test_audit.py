import csv
import json
from pathlib import Path

from PIL import Image

from src.data.audit import audit_images, discover_images, manifest_rows, match_csv_to_images


def test_directory_audit_and_label_free_manifest(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train = root / "images" / "training"
    val = root / "images" / "validation"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    image = Image.new("RGB", (16, 12), (100, 50, 20))
    image.save(train / "001.png")
    image.save(train / "001_copy.png")
    image.save(val / "002.png")
    refs = discover_images("X", root, {"training": "images/training/", "validation": "images/validation/"})
    audits = audit_images(refs)
    assert len(audits) == 3
    assert sum(item.readable for item in audits) == 3
    assert all(item.width == 16 and item.height == 12 and item.color_mode == "RGB" for item in audits)
    rows = manifest_rows(audits, "X", "training")
    assert len(rows) == 2
    assert not any("disease" in key.lower() for key in rows[0])
    assert rows == sorted(rows, key=lambda row: row["relative_path"])


def test_csv_matching_reports_missing_and_ambiguous(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "training"
    root.mkdir(parents=True)
    for name in ("1.png", "1.jpg"):
        Image.new("RGB", (8, 8)).save(root / name)
    refs = discover_images("X", tmp_path / "dataset", {"training": "training/"})
    result = match_csv_to_images(refs, [{"ID": "1"}, {"ID": "9"}], "ID")
    assert result["matched"] == 0
    assert "9" in result["missing_keys"]
    assert "1" in result["ambiguous_keys"]

