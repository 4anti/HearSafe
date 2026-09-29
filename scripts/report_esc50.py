"""Summarize ESC-50 metadata for HearSafe's first dataset experiment."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


STARTING_CLASSES = (
    "crying_baby",
    "footsteps",
    "coughing",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "siren",
    "car_horn",
    "dog",
    "crackling_fire",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/ESC-50-master/meta/esc50.csv"),
        help="Path to the official ESC-50 metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/esc50_class_counts.md"),
        help="Markdown report to write",
    )
    args = parser.parse_args()

    if not args.metadata.is_file():
        parser.error(f"metadata not found: {args.metadata}")

    with args.metadata.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"filename", "fold", "category"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            parser.error(f"missing metadata columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    counts = Counter(row["category"] for row in rows)
    folds = defaultdict(Counter)
    for row in rows:
        folds[row["category"]][row["fold"]] += 1

    missing_classes = set(STARTING_CLASSES) - set(counts)
    if missing_classes:
        parser.error(f"starting classes absent from metadata: {', '.join(sorted(missing_classes))}")
    if len({row["filename"] for row in rows}) != len(rows):
        parser.error("duplicate filenames found in metadata")

    fold_names = sorted({row["fold"] for row in rows}, key=int)
    lines = [
        "# ESC-50 class counts",
        "",
        "Generated from the official `meta/esc50.csv` using `python scripts/report_esc50.py`.",
        "The audio remains in the ignored `data/` directory.",
        "",
        f"- Total clips: **{len(rows)}**",
        f"- Dataset classes: **{len(counts)}**",
        f"- Starting HearSafe classes: **{len(STARTING_CLASSES)}**",
        f"- Clips in starting classes: **{sum(counts[name] for name in STARTING_CLASSES)}**",
        "",
        "## Starting classes by predefined fold",
        "",
        "| ESC-50 category | Total | " + " | ".join(f"Fold {fold}" for fold in fold_names) + " |",
        "|---|---:|" + "---:|" * len(fold_names),
    ]
    for category in STARTING_CLASSES:
        lines.append(
            f"| `{category}` | {counts[category]} | "
            + " | ".join(str(folds[category][fold]) for fold in fold_names)
            + " |"
        )
    lines.extend(
        [
            "",
            "## All classes",
            "",
            "| ESC-50 category | Clips |",
            "|---|---:|",
        ]
    )
    for category in sorted(counts):
        lines.append(f"| `{category}` | {counts[category]} |")
    lines.extend(
        [
            "",
            "ESC-50 is a curated sound-classification dataset. These counts alone do not establish ",
            "real-world alert accuracy or an `unknown/background` class.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} clips, {len(counts)} classes)")


if __name__ == "__main__":
    main()
