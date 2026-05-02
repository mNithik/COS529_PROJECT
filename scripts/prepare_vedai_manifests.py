from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SPLITS = {
    "fold01_train.txt": "fold01_write_test_fixed.txt",
    "fold02_val.txt": "fold02_write_test_fixed.txt",
    "fold03_test.txt": "fold03_write_test_fixed.txt",
}


def normalize_sample_id(line: str) -> str | None:
    sample = line.strip().replace("\\", "/")
    if not sample:
        return None
    sample = Path(sample).stem
    if sample.endswith("_co") or sample.endswith("_ir"):
        sample = sample[:-3]
    return sample


def build_manifests(data_dir: Path, output_dir: Path) -> list[tuple[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for output_name, source_name in DEFAULT_SPLITS.items():
        source_path = data_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source split file: {source_path}")

        sample_ids = []
        seen = set()
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            sample_id = normalize_sample_id(raw_line)
            if not sample_id or sample_id in seen:
                continue
            sample_ids.append(sample_id)
            seen.add(sample_id)

        output_path = output_dir / output_name
        output_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        summary.append((output_name, len(sample_ids)))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portable VEDAI manifest files for Phase 1 training.")
    parser.add_argument(
        "--data-dir",
        default=Path("data/VEDAI"),
        type=Path,
        help="Directory containing the original fold*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/manifests"),
        type=Path,
        help="Directory where normalized manifest files will be written.",
    )
    args = parser.parse_args()

    summary = build_manifests(args.data_dir.resolve(), args.output_dir.resolve())
    for name, count in summary:
        print(f"Wrote {name} with {count} samples")


if __name__ == "__main__":
    main()
