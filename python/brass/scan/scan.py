from itertools import product
from copy import deepcopy
from .template import smash_cmd

class Scan:
    def __init__(self):
        self.param_lists = {}

    def set_param(self, key: str, values):
        """Add parameter list to scan."""
        if not isinstance(values, (list, tuple)):
            values = [values]
        self.param_lists[key] = values

    def _set_in_cfg(self, cfg, key, value):
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    def sweep(self):
        """Yield (combo_dict, cfg_dict) for all parameter combinations."""
        keys = list(self.param_lists.keys())
        values = [self.param_lists[k] for k in keys]
        for combo in product(*values):
            cfg = {}
            combo_dict = dict(zip(keys, combo))
            for k, v in combo_dict.items():
                self._set_in_cfg(cfg, k, v)
            yield combo_dict, cfg

    def sweep_cmds(self):
        """Yield (combo_dict, smash_cmd_string)."""
        for combo, cfg in self.sweep():
            yield combo, smash_cmd(cfg)
