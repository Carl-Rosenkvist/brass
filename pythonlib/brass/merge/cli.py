import argparse
import pickle
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as file:
        return pickle.load(file)


def save(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    with path.open("wb") as file:
        pickle.dump(results, file)


def main():
    parser = argparse.ArgumentParser(
        description="Merge BRASS analysis result files.",
    )

    parser.add_argument(
        "input",
        nargs="+",
        type=Path,
        help="Input pickle files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output pickle file.",
    )

    args = parser.parse_args()

    results_by_meta = {}

    for path in args.input:
        print(f"Loading {path}")

        results = load(path)

        for entry in results:
            meta = entry["meta"]
            result = entry["result"]

            if meta not in results_by_meta:
                results_by_meta[meta] = result
            else:
                results_by_meta[meta] += result

    merged = [
        {
            "meta": meta,
            "result": result,
        }
        for meta, result in results_by_meta.items()
    ]

    save(args.output, merged)

    print(f"Merged {len(args.input)} files into {len(merged)} results.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
