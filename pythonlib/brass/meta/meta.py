# brass/meta.py
from typing import Any, Dict, Iterable, List, Tuple


class MetaBuilder:
    def __init__(
        self,
        key_specs: Iterable[str],
        missing: str = "NA",
        expand_dicts: bool = True,
        expand_lists: bool = True,
    ):
        """
        key_specs entries:
          - 'ALIAS=Dot.Path'
          - 'Dot.Path'  (alias defaults to last segment)

        expand_dicts:
          - True  -> {2212:82, 2112:126} -> adds ALIAS_2212=82, ALIAS_2112=126
          - False -> serialize dict to a single string (not recommended here)

        expand_lists:
          - True  -> [a,b] -> ALIAS_0=a, ALIAS_1=b
          - False -> serialize list to a single string
        """
        self.specs: List[Tuple[str, str]] = [
            self._parse_key_spec(s) for s in (key_specs or [])
        ]
        self.missing = missing
        self.expand_dicts = expand_dicts
        self.expand_lists = expand_lists

    @staticmethod
    def _parse_key_spec(spec: str) -> Tuple[str, str]:
        if "=" in spec:
            alias, path = spec.split("=", 1)
            return alias.strip(), path.strip()
        path = spec.strip()
        return path.split(".")[-1], path

    @staticmethod
    def _coerce_key(seg: str, mapping: Dict[Any, Any]):
        # exact string
        if seg in mapping:
            return seg
        # try int key (handles YAML int keys like 2212)
        try:
            ik = int(seg)
            if ik in mapping:
                return ik
        except Exception:
            pass
        # final fallback: string compare against existing keys
        for k in mapping.keys():
            if str(k) == seg:
                return k
        return None

    @classmethod
    def _get_by_path(cls, d: Dict[str, Any], dotted: str, default=None):
        cur = d
        for p in dotted.split("."):
            if not isinstance(cur, dict):
                return default
            key = cls._coerce_key(p, cur)
            if key is None:
                return default
            cur = cur[key]
        return cur

    @staticmethod
    def _format_scalar(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.12g}"  # stable float
        return str(x)

    def build(self, cfg: Dict[str, Any]):
        """
        Returns: (comma_joined_string, dict_of_alias_to_value)
        Dict may contain multiple keys per spec if expansion is enabled.
        """
        parts: List[str] = []
        kv: Dict[str, str] = {}

        for alias, path in self.specs:
            raw = self._get_by_path(cfg, path, self.missing)

            # Expand dicts → multiple scalar labels
            if isinstance(raw, dict) and self.expand_dicts:
                for k in sorted(raw.keys(), key=lambda x: str(x)):
                    sub_alias = f"{alias}_{k}"
                    val = self._format_scalar(raw[k])
                    kv[sub_alias] = val
                    parts.append(f"{sub_alias}={val}")
                continue

            # Expand lists/tuples if desired
            if isinstance(raw, (list, tuple)) and self.expand_lists:
                for i, item in enumerate(raw):
                    sub_alias = f"{alias}_{i}"
                    val = self._format_scalar(item)
                    kv[sub_alias] = val
                    parts.append(f"{sub_alias}={val}")
                continue

            # Fallback: single scalar
            val = self._format_scalar(raw)
            kv[alias] = val
            parts.append(f"{alias}={val}")

        return ",".join(parts), kv
