from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Dict, List, Hashable, Iterable, Tuple
import numpy as np
import brass as br
from brass import HistND
import copy


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
