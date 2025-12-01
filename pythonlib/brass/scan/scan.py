from itertools import product
from .template import smash_cmd


import json


class Scan:
    def __init__(self):
        # groups of small cfg patches; sweep = cartesian product of groups
        self._groups = []
        self._split_used = False  # allow only one split anchor

    def set_param(self, key: str, values, *, events=None, max_events=None):
        """Add a param sweep.
        - Normal: set_param(key, [v1, v2, ...])
        - Split (only for energy/impact): set_param(key, [v1, v2], events=[e1, e2] or [e], max_events=M)
          For each (vi, ei): repeats = M // ei; emit that many runs with General.Nevents=ei
        """
        if not isinstance(values, (list, tuple)):
            values = [values]

        if (events is not None or max_events is not None) and key not in (
            "Modi.Collider.Sqrtsnn",
            "Modi.Collider.Impact.Value",
        ):
            raise ValueError(
                "events/max_events splitting is only allowed for "
                "Modi.Collider.Sqrtsnn or Modi.Collider.Impact.Value"
            )
        if (events is not None or max_events is not None) and self._split_used:
            raise ValueError("Only one split parameter allowed in a scan")
        if events is not None or max_events is not None:
            self._split_used = True

        # Normal group (no splitting)
        if events is None and max_events is None:
            group = []
            for v in values:
                cfg = {}
                self._set_in_cfg(cfg, key, v)
                group.append(cfg)
            self._groups.append(group)
            return

        # Split group
        if events is None:
            raise ValueError("When using max_events, you must provide events.")
        events = list(events) if isinstance(events, (list, tuple)) else [events]

        # Broadcast events if needed
        if len(events) == 1 and len(values) > 1:
            events = events * len(values)
        if len(events) != len(values):
            raise ValueError("events must be length 1 or match the length of values.")

        group = []
        for v, ev in zip(values, events):
            ev = int(ev)
            repeats = 1 if max_events is None else max_events // ev
            repeats = max(repeats, 1)  # at least one run
            for _ in range(repeats):
                cfg = {}
                self._set_in_cfg(cfg, key, v)
                self._set_in_cfg(cfg, "General.Nevents", ev)
                group.append(cfg)
        self._groups.append(group)

    def _set_in_cfg(self, cfg, dotted_key, value):
        parts = dotted_key.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    def _deep_merge(self, base, upd):
        """Recursively merge upd into base."""
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _flatten(self, d, prefix="", out=None):
        """Flatten nested dict into dotted keys: {'A': {'B': 1}} -> {'A.B': 1}"""
        out = {} if out is None else out
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                self._flatten(v, kk, out)
            else:
                out[kk] = v
        return out

    def _dotted_to_nested(self, path, value):
        """Convert 'A.B.C' and v into {'A': {'B': {'C': v}}}."""
        parts = path.split(".")
        root = {}
        cur = root
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
        return root

    def sweep(self):
        """Yield (combo_dict, cfg_dict) for all combinations.

        combo_dict: flattened dict of the effective config (last one wins).
        cfg_dict:   nested dict config.
        """
        if not self._groups:
            return

        for picks in product(*self._groups):
            cfg = {}
            for patch in picks:
                self._deep_merge(cfg, patch)

            combo = self._flatten(cfg)
            yield combo, cfg

    def sweep_cmds(self):
        """Yield (combo_dict, cmd_string).

        cmd_string has one `-c '{...}'` per flattened key, with nested JSON
        reconstructed from the dotted paths in combo. JSON is compact and
        wrapped in single quotes to avoid shell splitting.
        """
        for combo, cfg in self.sweep():
            parts = []
            for dotted_key, val in combo.items():
                nested = self._dotted_to_nested(dotted_key, val)
                # compact JSON: no spaces, safer and shorter
                json_cfg = json.dumps(nested, separators=(",", ":"))
                # single-quote the JSON so bash treats it as one arg
                parts.append(f"-c '{json_cfg}'")
            cmd = " ".join(parts)
            yield combo, cmd
