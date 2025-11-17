# brass/cli/analyze_runs.py
from __future__ import annotations

import os
import sys
import glob
import argparse
import yaml
import difflib
import importlib
import importlib.util
import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Hashable, Iterable, Tuple
import multiprocessing as mp

import numpy as np
import brass as br
from brass import MetaBuilder, HistND


def _import_any(target):
    if os.path.isfile(target) and target.endswith(".py"):
        modname = os.path.splitext(os.path.basename(target))[0]
        spec = importlib.util.spec_from_file_location(modname, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from file {target}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    else:
        try:
            return importlib.import_module(target)
        except ImportError:
            if os.path.exists(target):
                dirpath = os.path.dirname(os.path.abspath(target))
                if dirpath not in sys.path:
                    sys.path.insert(0, dirpath)
                modname = os.path.splitext(os.path.basename(target))[0]
                return importlib.import_module(modname)
            raise


def _import_python_analyses(targets):
    for t in targets:
        try:
            _import_any(t)
        except Exception as e:
            print(f"[WARN] Failed to import {t}: {e}", file=sys.stderr)


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


def _find_binary_in_run(run_dir: str, candidates: str) -> str | None:
    pats = [p.strip() for p in (candidates or "").split(",") if p.strip()]
    if not pats:
        pats = ["particles_binary.bin"]

    for pat in pats:
        full = os.path.join(run_dir, pat)
        if any(ch in pat for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(full))
            for m in matches:
                if os.path.isfile(m):
                    return m
        else:
            if os.path.isfile(full):
                return full

    fallback = sorted(glob.glob(os.path.join(run_dir, "*.bin")))
    return fallback[0] if fallback else None


def _merge_leaf(a: Any, b: Any, key: Hashable | None) -> Any:
    if isinstance(a, HistND) and isinstance(b, HistND):
        out = copy.deepcopy(a)
        out.merge_(b)
        return out

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError(f"array mismatch at {key!r}")
        return a

    if isinstance(a, (str, tuple)) and isinstance(b, type(a)):
        if a != b:
            raise ValueError(f"value mismatch at {key!r}: {a!r} vs {b!r}")
        return a

    if isinstance(a, (int, float)) and isinstance(b, type(a)):
        if key == "n_events":
            return a + b
        if a != b:
            raise ValueError(f"numeric mismatch at {key!r}: {a} vs {b}")
        return a

    raise TypeError(
        f"cannot merge values of type {type(a)} and {type(b)} at key {key!r}"
    )


def _merge_any(a: Any, b: Any, key: Hashable | None) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out: Dict[Any, Any] = {}
        all_keys = set(a.keys()) | set(b.keys())
        for k in all_keys:
            if k in a and k in b:
                out[k] = _merge_any(a[k], b[k], k)
            elif k in a:
                out[k] = copy.deepcopy(a[k])
            else:
                out[k] = copy.deepcopy(b[k])
        return out
    return _merge_leaf(a, b, key)


def merge_two_states(acc: Dict[str, Any], st: Dict[str, Any]) -> Dict[str, Any]:
    for meta_label, analyses in st.items():
        if meta_label not in acc:
            acc[meta_label] = copy.deepcopy(analyses)
        else:
            for name, res in analyses.items():
                if name in acc[meta_label]:
                    acc[meta_label][name] = _merge_any(
                        acc[meta_label][name], res, key=name
                    )
                else:
                    acc[meta_label][name] = copy.deepcopy(res)
    return acc


def merge_state_list(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not states:
        raise ValueError("merge_state_list: empty state list")

    acc: Dict[str, Any] = {}
    for st in states:
        acc = merge_two_states(acc, st)
    return acc


def run_analysis_one_file(
    filename: str,
    meta: str,
    analysis_name: str,
    quantities,
    opts=None,
) -> Dict[str, Any]:
    opts = opts or {}

    analysis = br.create_analysis(analysis_name)
    dispatcher = br.DispatchingAccessor()
    dispatcher.register_analysis(analysis)
    reader = br.BinaryReader(filename, quantities, dispatcher)
    reader.read()

    analysis_state = analysis.to_state_dict()
    return {meta: {analysis_name: analysis_state}}


def _worker_run(args) -> Dict[str, Any]:
    filename, meta, analysis_name, quantities, opts = args
    return run_analysis_one_file(
        filename=filename,
        meta=meta,
        analysis_name=analysis_name,
        quantities=quantities,
        opts=opts,
    )


def run_analysis_many(
    file_and_meta: Iterable[Tuple[str, str]],
    analysis_name: str,
    quantities,
    output_dir=".",
    opts=None,
    nproc: int | None = None,
) -> Dict[str, Any]:
    opts = opts or {}
    jobs = [
        (fname, meta, analysis_name, quantities, opts)
        for (fname, meta) in file_and_meta
    ]

    if not jobs:
        raise ValueError("run_analysis_many: no jobs")

    if nproc is None or nproc <= 1:
        states = [_worker_run(j) for j in jobs]
    else:
        with mp.Pool(processes=nproc) as pool:
            states = pool.map(_worker_run, jobs)

    results = merge_state_list(states)
    analysis = br.create_analysis(analysis_name)
    results = analysis.finalize(results, output_dir)
    analysis.save(results, output_dir)
    return results


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

    args = ap.parse_args(argv)

    _import_python_analyses(args.load)

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
        binf = _find_binary_in_run(rd, args.binary_names)
        ymlf = os.path.join(rd, "config.yaml")

        if not (binf and os.path.isfile(binf) and os.path.isfile(ymlf)):
            if args.verbose:
                miss = "binary" if not (binf and os.path.isfile(binf)) else "YAML"
                print(f"[SKIP] {rd} (missing {miss})")
            continue

        try:
            with open(ymlf, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] bad YAML {ymlf}: {e}", file=sys.stderr)
            continue

        q = get_by_path(cfg, "Output.Particles.Quantities", [])
        q = list(q) if isinstance(q, list) else []

        if first_quantities is None:
            first_quantities = q
        elif q != first_quantities:
            mismatches.append((rd, q))

        meta, _ = meta_builder.build(cfg)
        file_and_meta.append((binf, meta))
        if args.verbose:
            print(f"[OK] {rd} -> {meta}")

    if not file_and_meta:
        print("[ERROR] no valid runs found.", file=sys.stderr)
        return 2

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
        print(f"[INFO] Quantities: {first_quantities}")
        print(f"[INFO] Analyses: {requested}")
        print(f"[INFO] Results dir: {results_dir}")

    for name in requested:
        if args.verbose:
            print(f"[INFO] Running analysis: {name}")
        out_dir_for_analysis = os.path.join(results_dir, name)
        os.makedirs(out_dir_for_analysis, exist_ok=True)
        run_analysis_many(
            file_and_meta=file_and_meta,
            analysis_name=name,
            quantities=first_quantities or [],
            output_dir=out_dir_for_analysis,
            opts={},
            nproc=args.nproc,
        )

    if args.verbose:
        print("[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
