import argparse
import json

from .scan import Scan, command, jobs


def parse_value(value):
    """Parse JSON values, falling back to strings."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def split_values(text):
    """
    Split comma-separated scan values while respecting nested
    JSON arrays, objects, and quoted strings.

    Examples
    --------
    "8.8,16.8"
        -> ["8.8", "16.8"]

    "[0.0,3.372],[3.372,4.794]"
        -> ["[0.0,3.372]", "[3.372,4.794]"]
    """
    values = []
    start = 0
    depth = 0
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
                raise argparse.ArgumentTypeError(
                    "Unbalanced brackets in parameter values"
                )

        elif char == "," and depth == 0:
            values.append(text[start:i].strip())
            start = i + 1

    if depth != 0 or in_string:
        raise argparse.ArgumentTypeError(
            "Unbalanced brackets or quotes in parameter values"
        )

    values.append(text[start:].strip())

    return values


def parse_param(value):
    try:
        key, values = value.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "parameter must have the form KEY=VALUE[,VALUE...]"
        )

    return key, [parse_value(item) for item in split_values(values)]


def add_arguments(parser):
    parser.add_argument(
        "--param",
        action="append",
        type=parse_param,
        required=True,
        metavar="KEY=VALUES",
        help=(
            "Parameter to scan, e.g. "
            "Modi.Collider.Sqrtsnn=3.0,4.5,7.7 "
            "or Modi.Collider.Impact.Range=[0,3],[3,6]"
        ),
    )

    parser.add_argument(
        "--events-per-job",
        type=int,
    )

    parser.add_argument(
        "--max-events",
        type=int,
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Command prepended to each generated configuration.",
    )


def run(args):
    scan = Scan()

    for key, values in args.param:
        scan.add(key, values)

    for point in jobs(
        scan,
        events_per_job=args.events_per_job,
        max_events=args.max_events,
    ):
        print(command(point, args.prefix))


def main():
    parser = argparse.ArgumentParser(description="Generate parameter scan commands.")

    add_arguments(parser)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
