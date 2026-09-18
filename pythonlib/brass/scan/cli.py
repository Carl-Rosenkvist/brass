import argparse
import json

from .scan import Scan, command, jobs


def parse_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_param(value):
    try:
        key, values = value.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "parameter must have the form KEY=VALUE[,VALUE...]"
        )

    return key, [parse_value(item) for item in values.split(",")]


def add_arguments(parser):
    parser.add_argument(
        "--param",
        action="append",
        type=parse_param,
        required=True,
        metavar="KEY=VALUES",
        help=("Parameter to scan, e.g. Modi.Collider.Sqrtsnn=3.0,4.5,7.7"),
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
