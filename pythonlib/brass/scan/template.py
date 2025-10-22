#!/usr/bin/env python3
import yaml

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

