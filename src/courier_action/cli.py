"""Command-line interface for synthetic generation and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_csv, load_protocol, write_result
from .synthetic import generate_synthetic_dataframe


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="courier-action")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "make-synthetic", help="write unmistakably fabricated smoke-test data"
    )
    synthetic.add_argument("--output-dir", required=True, type=Path)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--groups", type=int, default=80)
    synthetic.add_argument("--rows-per-group", type=int, default=64)

    evaluate = subparsers.add_parser(
        "evaluate", help="run aggregate-only grouped cross-validation"
    )
    evaluate.add_argument("--train-csv", required=True, type=Path)
    evaluate.add_argument("--protocol", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.command == "make-synthetic":
        frame = generate_synthetic_dataframe(
            seed=args.seed, n_groups=args.groups, rows_per_group=args.rows_per_group
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = args.output_dir / "train.csv"
        frame.to_csv(train_path, index=False)
        metadata = {
            "label": "synthetic smoke — not a performance claim",
            "seed": args.seed,
            "groups": args.groups,
            "rows": int(len(frame)),
            "note": "fabricated SYNTH_ records; not a real-data distribution",
        }
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"synthetic training CSV written: {train_path}")
        return 0

    protocol = load_protocol(args.protocol)
    result = evaluate_csv(args.train_csv, protocol)
    result_path = write_result(result, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"aggregate result written: {result_path}")
    return 0
