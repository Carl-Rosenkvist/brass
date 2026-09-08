import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar

import brass

A = TypeVar("A", bound=brass.Analysis)


def _run_one_analysis(
    analysis_type: type[A],
    folder_path: Path,
    args: argparse.Namespace,
) -> A:
    analysis = analysis_type(
        folder_path,
        args,
    )
    analysis.run()
    return analysis


class AnalysisRunner(Generic[A]):
    def __init__(
        self,
        analysis_type: type[A],
        folder_paths: Iterable[str | Path],
        args: argparse.Namespace | None = None,
        max_workers: int | None = None,
    ):
        self.analysis_type = analysis_type
        self.folder_paths = [Path(path) for path in folder_paths]
        self.args = args
        self.max_workers = max_workers

        if not self.folder_paths:
            raise ValueError("At least one folder path is required")

    def run(self) -> list[dict[str, Any]]:
        analyses_by_meta: dict[Any, A] = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            analyses = executor.map(
                _run_one_analysis,
                repeat(self.analysis_type),
                self.folder_paths,
                repeat(self.args),
            )

            for analysis in analyses:
                if analysis.meta not in analyses_by_meta:
                    analyses_by_meta[analysis.meta] = analysis
                    continue

                analyses_by_meta[analysis.meta] += analysis

        return [
            {
                "meta": analysis.meta,
                "result": analysis.result(),
            }
            for analysis in analyses_by_meta.values()
        ]

    def save(
        self,
        path: str | Path,
        results: list[dict[str, Any]] | None = None,
    ) -> Path:
        save_path = Path(path)

        if results is None:
            results = self.run()

        with save_path.open("wb") as file:
            pickle.dump(results, file)

        return save_path

    def run_and_save(
        self,
        path: str | Path,
    ) -> list[dict[str, Any]]:
        results = self.run()
        self.save(path, results)
        return results

    def load(
        self,
        path: str | Path,
    ) -> list[dict[str, Any]]:
        load_path = Path(path)

        with load_path.open("rb") as file:
            results = pickle.load(file)

        return results
