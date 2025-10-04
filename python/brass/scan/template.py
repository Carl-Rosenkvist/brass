#!/usr/bin/env python3
import yaml

# ---- Base template for a normal run ----
def default_cfg():
    return {
        "Logging": {"default": "ERROR"},
        "General": {
            "Modus": "Collider",
            "Time_Step_Mode": "Fixed",
            "Delta_Time": 1.0,
            "End_Time": 200,
            "Randomseed": -1,
            "Nevents": 1,
        },
        "Modi": {
            "Collider": {
                "Calculation_Frame": "center of mass",
                "Projectile": {"Particles": {2212: 82, 2112: 126}},
                "Target": {"Particles": {2212: 82, 2112: 126}},
                "Sqrtsnn": 17.3,
                "Fermi_Motion": "frozen",
                "Impact": {"Range": [0,3.4], "Sample": "uniform"},
            }
        },
    }

# ---- Force quoting of all strings ----
class QuotedStr(str): pass

def quoted_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

yaml.add_representer(QuotedStr, quoted_str_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(str, quoted_str_representer, Dumper=yaml.SafeDumper)

def quote_strings(obj):
    """Recursively wrap all strings in QuotedStr so they are double-quoted in YAML."""
    if isinstance(obj, str):
        return QuotedStr(obj)
    if isinstance(obj, list):
        return [quote_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: quote_strings(v) for k, v in obj.items()}
    return obj

def cfg_to_inline_yaml(cfg: dict) -> str:
    """Dump the cfg dict into a single-line YAML string with quoted strings."""
    cfg = quote_strings(cfg)
    return yaml.safe_dump(
        cfg, default_flow_style=True, sort_keys=False, width=float("inf")
    ).strip()

def smash_cmd(cfg: dict) -> str:
    """Return the full smash command-line override (-c {...}) as string."""
    return f"-c {cfg_to_inline_yaml(cfg)}"

