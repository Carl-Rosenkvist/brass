import argparse
from pathlib import Path

from .analyses import ANALYSES
from .analysis_runner import AnalysisRunner


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brass",
        description="Run BRASS analyses.",
    )

    subparsers = parser.add_subparsers(
        dest="analysis_name",
        required=True,
        metavar="ANALYSIS",
    )

    for name, analysis_type in ANALYSES.items():
        analysis_parser = subparsers.add_parser(
            name,
            help=analysis_type.__doc__,
            description=analysis_type.__doc__,
        )

        analysis_parser.set_defaults(
            analysis_type=analysis_type,
        )

        analysis_parser.add_argument(
            "folders",
            nargs="+",
            type=Path,
            help="Simulation folders to analyze",
        )
        analysis_parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=Path(f"{name}.pkl"),
            help="Output pickle file",
        )
        analysis_parser.add_argument(
            "-j",
            "--max-workers",
            type=int,
            default=None,
            help="Maximum number of worker processes",
        )

        analysis_type.add_arguments(analysis_parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    analysis_type = args.analysis_type

    runner = AnalysisRunner(
        analysis_type=analysis_type,
        folder_paths=args.folders,
        args=args,
        max_workers=args.max_workers,
    )

    results = runner.run_and_save(args.output)

    print(f"Saved {len(results)} grouped result(s) to {args.output}")


if __name__ == "__main__":
    main()
