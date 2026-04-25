#!/usr/bin/env python3
import argparse
import json
import brass as br


def parse_value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_assignment(arg):
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE1,VALUE2,... but got: {arg}"
        )

    key, raw_values = arg.split("=", 1)
    values = [parse_value(v) for v in raw_values.split(",")]

    return key, values


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate SMASH command-line arguments from a parameter scan."
    )

    p.add_argument("-i", "--config", required=True)
    p.add_argument("-o", "--output", required=True)

    p.add_argument("--events-per-job", type=int, required=True)
    p.add_argument("--max-events", type=int, required=True)

    p.add_argument(
        "--param",
        action="append",
        type=parse_assignment,
        required=True,
        help="Parameter scan, e.g. Modi.Collider.Sqrtsnn=3.0,4.5,7.7",
    )

    return p.parse_args()


def main():
    args = parse_args()

    scan = br.Scan()

    for key, values in args.param:
        scan.set_param(key, values)

    scan.set_jobs(
        events_per_job=args.events_per_job,
        max_events=args.max_events,
    )

    with open(args.output, "w") as f:
        for _, cmd in scan.sweep_cmds(prefix=f"-i {args.config}"):
            f.write(cmd + "\n")


if __name__ == "__main__":
    main()
