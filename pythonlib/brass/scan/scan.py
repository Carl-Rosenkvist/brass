import json
from itertools import product


def parse_values(text):
    """Parse comma-separated scan values, respecting JSON arrays."""

    values = []
    depth = 0
    start = 0
    in_string = False
    escaped = False

    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1

            if depth < 0:
                raise ValueError("Unbalanced brackets")

        elif char == "," and depth == 0:
            values.append(text[start:i].strip())
            start = i + 1

    if depth != 0 or in_string:
        raise ValueError("Unbalanced brackets or quotes")

    values.append(text[start:].strip())

    def parse_value(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return [parse_value(value) for value in values]


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
