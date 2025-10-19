# brass/cli/analyze_runs.py
import os, sys, glob, argparse, yaml, difflib
import brass as br

import importlib, importlib.util, sys, os

def _import_any(target):
    # if file path
    if os.path.isfile(target) and target.endswith(".py"):
        modname = os.path.splitext(os.path.basename(target))[0]
        spec = importlib.util.spec_from_file_location(modname, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from file {target}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    else:
        # dotted name
        try:
            return importlib.import_module(target)
        except ImportError:
            # fallback: if it's a relative path without .py
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

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scan run dirs, build meta labels from keys, check Quantities, run brass.run_analysis."
    )
    ap.add_argument("--list-analyses", action="store_true",
                    help="List registered analyses and exit")
    ap.add_argument("output_dir", nargs="?", help="Top directory containing run subfolders")
    # accept one or more analysis names
    ap.add_argument("analysis_names", nargs="*", help="One or more analysis names (e.g. bulk dNdY)")
    ap.add_argument("--pattern", default="out-*", help="Glob for run folders (default: out-*)")
    ap.add_argument("--keys", nargs="+", required=False,
                    help=("Dotted keys for labels (last segment used in label), e.g.: "
                          "Modi.Collider.Sqrtsnn General.Nevents"))
    ap.add_argument("--results-subdir", default="data",
                    help="Where to store analysis results (default: data)")
    ap.add_argument("--strict-quantities", action="store_true",
                    help="Fail if Quantities differ across runs (default: warn and use first)")
    ap.add_argument("-v", "--verbose", action="store_true")
    
    ap.add_argument(
        "--load", metavar="MODULE_OR_FILE", nargs="*", default=[],
        help="Import Python module(s) or file(s) that register analyses."
    )

    args = ap.parse_args(argv)

    _import_python_analyses(args.load)
    # Handle --list-analyses early
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
           

    # Require positionals if not just listing
    if not args.output_dir or not args.analysis_names:
        ap.error("output_dir and at least one analysis name are required unless --list-analyses is used")

    # Normalize analysis names: support comma-separated or space-separated, de-dup
    raw_names = []
    for item in args.analysis_names:
        raw_names.extend([p for p in (item.split(",") if "," in item else [item]) if p])
    requested = _dedupe_preserve_order(raw_names)

    # Validate requested analyses against registry
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
                    print(f"        did you mean: '{suggestion[0]}' for '{n}'?", file=sys.stderr)
            print("        use --list-analyses to see options.", file=sys.stderr)
        return 2

    out_top = os.path.abspath(args.output_dir)
    runs = sorted(glob.glob(os.path.join(out_top, args.pattern)))
    if not runs:
        print(f"[ERROR] no runs match {args.pattern} under {out_top}", file=sys.stderr)
        return 2

    file_and_meta = []
    first_quantities = None
    mismatches = []

    for rd in runs:
        binf = os.path.join(rd, "particles_binary.bin")
        ymlf = os.path.join(rd, "config.yaml")
        if not (os.path.isfile(binf) and os.path.isfile(ymlf)):
            if args.verbose:
                print(f"[SKIP] {rd} (missing binary or YAML)")
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

        parts = []
        for k in (args.keys or []):
            val = get_by_path(cfg, k, "NA")
            last = k.split(".")[-1]
            parts.append(f"{last}={val}")
        meta = ",".join(parts)

        file_and_meta.append((binf, meta))
        if args.verbose:
            print(f"[OK] {rd} -> {meta}")

    if not file_and_meta:
        print("[ERROR] no valid runs found.", file=sys.stderr)
        return 2

    if mismatches:
        msg = ["[ERROR] Quantities mismatch detected:" if args.strict_quantities
               else "[WARN] Quantities mismatch detected (using first set):"]
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

    # Call the vectorized binding
    br.run_analysis(
        file_and_meta=file_and_meta,
        analysis_names=requested,     # <— multiple analyses supported
        quantities=first_quantities or [],
        output_folder=results_dir,
    )

    if args.verbose:
        print("[DONE]")
    return 0
