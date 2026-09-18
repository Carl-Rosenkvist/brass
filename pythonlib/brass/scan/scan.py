import json
from itertools import product


class Scan:
    def __init__(self):
        self.params = {}

    def add(self, key, values):
        if not isinstance(values, (list, tuple)):
            values = [values]

        self.params[key] = values
        return self

    def __iter__(self):
        keys = list(self.params)

        for values in product(*(self.params[key] for key in keys)):
            yield dict(zip(keys, values))


def nested_config(point):
    config = {}

    for key, value in point.items():
        target = config
        parts = key.split(".")

        for part in parts[:-1]:
            target = target.setdefault(part, {})

        target[parts[-1]] = value

    return config


def jobs(
    scan,
    events_per_job=None,
    max_events=None,
):
    if events_per_job is None:
        yield from scan
        return

    if max_events is None:
        raise ValueError("max_events is required when events_per_job is set")

    repeats = max_events // events_per_job

    for point in scan:
        for _ in range(repeats):
            yield {
                **point,
                "General.Nevents": events_per_job,
            }


def command(point, prefix=""):
    config = json.dumps(
        nested_config(point),
        separators=(",", ":"),
    )

    return f"{prefix} -c '{config}'".strip()
