from __future__ import annotations
from collections.abc import Mapping
import copy
from typing import Any, Dict, List, Hashable, Iterable, Tuple
import multiprocessing as mp
import pickle
import os
import numpy as np
import brass as br
from brass import HistND
import sys
import importlib
import importlib.util


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


def import_python_analyses(targets):
    for t in targets:
        try:
            _import_any(t)
        except Exception as e:
            print(f"[WARN] Failed to import {t}: {e}", file=sys.stderr)


def _merge_leaf(a: Any, b: Any, key: Hashable | None) -> Any:
    # HistND: use builtin merge
    if isinstance(a, HistND) and isinstance(b, HistND):
        out = copy.deepcopy(a)
        out.merge_(b)
        return out

    # numpy arrays: sum elementwise
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            raise ValueError(f"array shape mismatch at {key!r}: {a.shape} vs {b.shape}")
        return a + b

    # lists: concatenate (append semantics)
    if isinstance(a, list) and isinstance(b, list):
        return a + b

    # numeric scalars: sum
    if isinstance(a, (int, float)) and isinstance(b, type(a)):
        return a + b

    # strings / tuples: require equality
    if isinstance(a, (str, tuple)) and isinstance(b, type(a)):
        if a != b:
            raise ValueError(f"value mismatch at {key!r}: {a!r} vs {b!r}")
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
    filename: str, meta: str, analysis_name: str, quantities, opts=None, load=None
) -> Dict[str, Any]:
    opts = opts or {}

    if load:
        import_python_analyses(load)

    analysis = br.create_analysis(analysis_name)
    dispatcher = br.DispatchingAccessor()
    dispatcher.register_analysis(analysis)
    reader = br.BinaryReader(filename, quantities, dispatcher)
    reader.read()

    analysis_state = analysis.to_state_dict()
    return {meta: {analysis_name: analysis_state}}


def _worker_run(args) -> Dict[str, Any]:
    filename, meta, analysis_name, quantities, opts, load = args
    return run_analysis_one_file(
        filename=filename,
        meta=meta,
        analysis_name=analysis_name,
        quantities=quantities,
        opts=opts,
        load=load,
    )


def run_analysis(
    file_and_meta: Iterable[Tuple[str, str]],
    analysis_name: str,
    quantities,
    output_dir=".",
    opts=None,
    nproc: int | None = None,
    load=None,
) -> Dict[str, Any]:
    opts = opts or {}
    jobs = [
        (fname, meta, analysis_name, quantities, opts, load)
        for (fname, meta) in file_and_meta
    ]

    if not jobs:
        raise ValueError("run_analysis_many: no jobs")

    if nproc is None or nproc <= 1:
        states = [_worker_run(j) for j in jobs]
    else:
        with mp.Pool(processes=nproc) as pool:
            states = pool.map(_worker_run, jobs)

    results: Dict[str, Any] = merge_state_list(states)

    analysis = br.create_analysis(analysis_name)
    analysis.finalize(results)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{analysis_name}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    return results
