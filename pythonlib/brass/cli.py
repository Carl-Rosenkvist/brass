import argparse
import importlib
import sys
from pathlib import Path

import brass

from .analyses import ANALYSES
from .analysis_runner import AnalysisRunner


def load_analyses(paths: list[Path]) -> dict:
    """Load additional analyses from Python files."""

    analyses = dict(ANALYSES)

    for path in paths:
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Analysis file not found: {path}")

        # Make the module importable by multiprocessing workers
        module_dir = str(path.parent)

        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        # Import the Python file as a normal module
        module = importlib.import_module(path.stem)

        # Find Analysis subclasses defined in the file
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, brass.Analysis)
                and obj is not brass.Analysis
                and obj.__module__ == module.__name__
            ):
                if name in analyses:
                    raise ValueError(f"Analysis '{name}' is already registered")

                analyses[name] = obj

    return analyses


def create_parser(analyses: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brass",
        description="Run BRASS analyses.",
    )

    parser.add_argument(
        "--load",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="Load additional analyses from a Python file",
    )

    subparsers = parser.add_subparsers(
        dest="analysis_name",
        required=True,
        metavar="ANALYSIS",
    )

    for name, analysis_type in analyses.items():
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
    # First, find external analysis files
    bootstrap = argparse.ArgumentParser(add_help=False)

    bootstrap.add_argument(
        "--load",
        action="append",
        type=Path,
        default=[],
    )

    bootstrap_args, _ = bootstrap.parse_known_args()

    # Load external analyses before building subcommands
    analyses = load_analyses(bootstrap_args.load)

    # Build the full CLI
    parser = create_parser(analyses)
    args = parser.parse_args()

    runner = AnalysisRunner(
        analysis_type=args.analysis_type,
        folder_paths=args.folders,
        args=args,
        max_workers=args.max_workers,
    )

    results = runner.run_and_save(args.output)

    print(f"Saved {len(results)} grouped result(s) to {args.output}")


if __name__ == "__main__":
    main()
