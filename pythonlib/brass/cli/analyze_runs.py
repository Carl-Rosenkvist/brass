# brass/cli/analyze_runs.py
from __future__ import annotations

import os
import sys
import glob
import argparse
import yaml
import difflib
import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Hashable, Iterable, Tuple
import multiprocessing as mp

import numpy as np
import brass as br
from brass import MetaBuilder, HistND


def get_by_path(d, dotted, default=None):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _find_binary_in_run(run_dir: str, candidates: str) -> str:
    pats = [p.strip() for p in (candidates or "").split(",") if p.strip()]
    if not pats:
        pats = ["particles_binary.bin"]

    # Try each pattern
    for pat in pats:
        full = os.path.join(run_dir, pat)

        # Glob-pattern?
        if any(ch in pat for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(full))
            for m in matches:
                if os.path.isfile(m):
                    return m

        # Exact filename
        else:
            if os.path.isfile(full):
                return full

    # STRICT: nothing found -> throw
    raise FileNotFoundError(
        f"No binary file found in '{run_dir}' matching pattern(s): {', '.join(pats)}"
    )


def _find_binaries_in_run(run_dir: str, candidates: str) -> list[str]:
    """
    Find *all* matching binary files in a run directory.

    `candidates` is a comma-separated list of names or glob patterns.
    Example: 'particles_*.bin,collisions_*.bin'.
    """
    pats = [p.strip() for p in (candidates or "").split(",") if p.strip()]
    if not pats:
        pats = ["particles_binary.bin"]

    found: list[str] = []

    for pat in pats:
        full = os.path.join(run_dir, pat)

        # Glob-pattern?
        if any(ch in pat for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(full))
            for m in matches:
                if os.path.isfile(m):
                    found.append(m)
        # Exact filename
        else:
            if os.path.isfile(full):
                found.append(full)

    # dedupe + sort for stability
    found = sorted(set(found))

    if not found:
        raise FileNotFoundError(
            f"No binary file found in '{run_dir}' matching pattern(s): {', '.join(pats)}"
        )

    return found


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scan run dirs, build meta labels from keys, check Quantities, run brass analyses."
    )
    ap.add_argument(
        "--list-analyses", action="store_true", help="List registered analyses and exit"
    )
    ap.add_argument(
        "output_dir", nargs="?", help="Top directory containing run subfolders"
    )
    ap.add_argument(
        "analysis_names", nargs="*", help="One or more analysis names (e.g. bulk dNdY)"
    )
    ap.add_argument(
        "--pattern", default="out-*", help="Glob for run folders (default: out-*)"
    )
    ap.add_argument(
        "--keys",
        nargs="+",
        required=False,
        help=(
            "Alias-qualified dotted keys for labels. "
            "Use 'ALIAS=Dot.Path' or 'Dot.Path' (alias defaults to last segment). "
            "Example: Proj=Modi.Collider.Projectile.Particles "
            "Targ=Modi.Collider.Target.Particles "
            "Sqrtsnn=Modi.Collider.Sqrtsnn"
        ),
    )
    ap.add_argument(
        "--results-subdir",
        default="data",
        help="Where to store analysis results (default: data)",
    )
    ap.add_argument(
        "--strict-quantities",
        action="store_true",
        help="Fail if Quantities differ across runs (default: warn and use first)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--load",
        metavar="MODULE_OR_FILE",
        nargs="*",
        default=[],
        help="Import Python module(s) or file(s) that register analyses.",
    )
    ap.add_argument(
        "--binary-names",
        default="particles_binary.bin",
        help=(
            "Comma-separated candidate filenames or glob patterns searched inside each run dir. "
            "Example: 'collisions.bin,particles_binary.bin,*.bin'. "
            "Default: particles_binary.bin"
        ),
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processes for multiprocessing (default: no multiprocessing).",
    )
    ap.add_argument(
        "-q",
        "--quantities",
        nargs="+",
        default=None,
        help="Override Output.Particles.Quantities with this list of quantities.",
    )

    args = ap.parse_args(argv)

    br.import_python_analyses(args.load)

    if args.list_analyses:
        analyses = br.list_analyses()
        if not analyses:
            print("No analyses available.")
        else:
            print("Available Analyses:")
            print("-------------------")
            for name in analyses:
                print(f"  • {name}")
        sys.exit(0)

    if not args.output_dir or not args.analysis_names:
        ap.error(
            "output_dir and at least one analysis name are required unless --list-analyses is used"
        )

    raw_names = []
    for item in args.analysis_names:
        raw_names.extend([p for p in (item.split(",") if "," in item else [item]) if p])
    requested = _dedupe_preserve_order(raw_names)

    try:
        available = list(br.list_analyses())
    except Exception as e:
        print(f"[ERROR] unable to query registered analyses: {e}", file=sys.stderr)
        return 2

    unknown = [n for n in requested if n not in available]
    if unknown:
        print(f"[ERROR] unknown analyses: {', '.join(unknown)}", file=sys.stderr)
        if len(available):
            for n in unknown:
                suggestion = difflib.get_close_matches(n, available, n=1)
                if suggestion:
                    print(
                        f"        did you mean: '{suggestion[0]}' for '{n}'?",
                        file=sys.stderr,
                    )
            print("        use --list-analyses to see options.", file=sys.stderr)
        return 2

    out_top = os.path.abspath(args.output_dir)
    runs = sorted(glob.glob(os.path.join(out_top, args.pattern)))
    if not runs:
        print(f"[ERROR] no runs match {args.pattern} under {out_top}", file=sys.stderr)
        return 2

    file_and_meta: list[tuple[str, str]] = []
    first_quantities = None
    mismatches = []

    meta_builder = MetaBuilder(
        args.keys or [], missing="NA", expand_dicts=True, expand_lists=True
    )

    for rd in runs:
        try:
            binfs = _find_binaries_in_run(rd, args.binary_names)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2

        ymlf = os.path.join(rd, "config.yaml")

        if not os.path.isfile(ymlf):
            print(f"[ERROR] Missing config YAML in {rd}: {ymlf}", file=sys.stderr)
            return 2

        if args.verbose:
            for b in binfs:
                print(f"[BIN] {rd}: using binary file '{b}'")

        try:
            with open(ymlf, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] bad YAML {ymlf}: {e}", file=sys.stderr)
            continue

        if args.quantities is None:
            q = get_by_path(cfg, "Output.Particles.Quantities", [])
            q = list(q) if isinstance(q, list) else []
            if first_quantities is None:
                first_quantities = q
            elif q != first_quantities:
                mismatches.append((rd, q))

        meta, _ = meta_builder.build(cfg)

        # add *all* binaries from this run dir, same meta for each
        for binf in binfs:
            file_and_meta.append((binf, meta))

        if args.verbose:
            print(f"[OK] {rd} -> {meta} (n_bin_files={len(binfs)})")

    if not file_and_meta:
        print("[ERROR] no valid runs found.", file=sys.stderr)
        return 2

    if args.quantities is not None:
        quantities = list(args.quantities)
    else:
        quantities = first_quantities or []
        if mismatches:
            msg = [
                (
                    "[ERROR] Quantities mismatch detected:"
                    if args.strict_quantities
                    else "[WARN] Quantities mismatch detected (using first set):"
                )
            ]
            msg.append(f"  First: {first_quantities}")
            for rd, q in mismatches:
                msg.append(f"  {rd}: {q}")
            print("\n".join(msg), file=sys.stderr)
            if args.strict_quantities:
                return 3

    results_dir = os.path.join(out_top, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    if args.verbose:
        print(f"[INFO] N files: {len(file_and_meta)}")
        print(f"[INFO] Quantities: {quantities}")
        print(f"[INFO] Analyses: {requested}")
        print(f"[INFO] Results dir: {results_dir}")

    for name in requested:
        if args.verbose:
            print(f"[INFO] Running analysis: {name}")
        out_dir_for_analysis = results_dir
        os.makedirs(out_dir_for_analysis, exist_ok=True)
        br.run_analysis(
            file_and_meta=file_and_meta,
            analysis_name=name,
            quantities=quantities,
            output_dir=out_dir_for_analysis,
            opts={},
            nproc=args.nproc,
            load=args.load,
        )

    if args.verbose:
        print("[DONE]")
    return 0
