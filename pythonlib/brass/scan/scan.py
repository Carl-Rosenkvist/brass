from copy import deepcopy
from itertools import product
import json


class Scan:
    def __init__(self):
        self._groups = []
        self._events_per_job = None
        self._max_events = None

    def set_param(self, key: str, values):
        """Add one scan parameter.

        Example:
            scan.set_param("Modi.Collider.Sqrtsnn", [3.0, 4.5, 7.7])
            scan.set_param("Modi.Collider.Impact.Value", [0, 2, 4])
        """
        if not isinstance(values, (list, tuple)):
            values = [values]

        group = []
        for value in values:
            cfg = {}
            self._set_in_cfg(cfg, key, value)
            group.append(cfg)

        self._groups.append(group)

    def set_jobs(self, *, events_per_job: int, max_events: int):
        """Set how many jobs to generate per parameter point."""
        if events_per_job <= 0:
            raise ValueError("events_per_job must be positive")
        if max_events <= 0:
            raise ValueError("max_events must be positive")

        self._events_per_job = int(events_per_job)
        self._max_events = int(max_events)

    def sweep(self):
        """Yield ``(combo_dict, cfg_dict)`` for all scan points.

        Each parameter point is repeated enough times to reach ``max_events``.
        Each emitted config gets ``General.Nevents = events_per_job``.
        """
        if not self._groups:
            return

        repeats = self._job_repeats()

        for picks in product(*self._groups):
            base_cfg = {}

            for patch in picks:
                self._deep_merge(base_cfg, patch)

            for _ in range(repeats):
                cfg = deepcopy(base_cfg)

                if self._events_per_job is not None:
                    self._set_in_cfg(cfg, "General.Nevents", self._events_per_job)

                combo = self._flatten(cfg)
                yield combo, cfg

    def sweep_cmds(self, prefix=""):
        """Yield ``(combo_dict, cmd_string)``.

        Example output:
            -i config.yaml -c '{"Modi":{"Collider":{"Sqrtsnn":3.0}}}' -c '{"General":{"Nevents":1000}}'
        """
        for combo, _ in self.sweep():
            parts = []

            if prefix:
                parts.append(prefix)

            for dotted_key, value in combo.items():
                nested = self._dotted_to_nested(dotted_key, value)
                json_cfg = json.dumps(nested, separators=(",", ":"))
                parts.append(f"-c '{json_cfg}'")

            yield combo, " ".join(parts)

    def _job_repeats(self):
        if self._events_per_job is None:
            return 1

        return max(1, self._max_events // self._events_per_job)

    def _set_in_cfg(self, cfg, dotted_key, value):
        parts = dotted_key.split(".")
        d = cfg

        for part in parts[:-1]:
            d = d.setdefault(part, {})

        d[parts[-1]] = value

    def _deep_merge(self, base, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)

    def _flatten(self, d, prefix="", out=None):
        out = {} if out is None else out

        for key, value in d.items():
            dotted_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten(value, dotted_key, out)
            else:
                out[dotted_key] = value

        return out

    def _dotted_to_nested(self, dotted_key, value):
        parts = dotted_key.split(".")
        root = {}
        d = root

        for part in parts[:-1]:
            d[part] = {}
            d = d[part]

        d[parts[-1]] = value
        return root
