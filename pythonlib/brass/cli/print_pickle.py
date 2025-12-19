#!/usr/bin/env python3
import pickle
import pprint
import sys
from pathlib import Path
import numpy as np


def truncate(obj, max_items=10, max_depth=4, _depth=0):
    if _depth >= max_depth:
        return "..."

    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["..."] = f"{len(obj) - max_items} more items"
                break
            out[k] = truncate(v, max_items, max_depth, _depth + 1)
        return out

    if isinstance(obj, list):
        if len(obj) > max_items:
            return (
                [
                    truncate(x, max_items, max_depth, _depth + 1)
                    for x in obj[: max_items // 2]
                ]
                + ["..."]
                + [
                    truncate(x, max_items, max_depth, _depth + 1)
                    for x in obj[-max_items // 2 :]
                ]
            )
        return [truncate(x, max_items, max_depth, _depth + 1) for x in obj]

    return obj


def main():
    if len(sys.argv) != 2:
        print("Usage: print-pickle <file.pkl>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])

    with path.open("rb") as f:
        obj = pickle.load(f)

    np.set_printoptions(
        edgeitems=3,
        threshold=10,
        linewidth=120,
        suppress=True,
    )

    truncated = truncate(obj, max_items=10, max_depth=4)
    pprint.pprint(truncated, width=120, compact=False)


if __name__ == "__main__":
    main()
