import argparse
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import yaml


class Analysis(ABC):
    config_filename: str = "config.yaml"

    output_sections: tuple[str, ...] = (
        "Particles",
        "Dileptons",
        "Collisions",
    )

    ignored_meta_paths: tuple[tuple[str, ...], ...] = (
        ("Logging",),
        ("Output",),
        ("General", "Randomseed"),
        ("decaymodes",),
        ("particles",),
    )

    def __init__(
        self,
        folder_path: str | Path,
        args: argparse.Namespace,
    ):
        self.folder_path = Path(folder_path)
        self.args = args

        self.config_path = self.folder_path / self.config_filename
        self.config = self.load_config()
        self.meta = self.extract_meta()
        self.quantities = self.extract_output_quantities()
        self.binary_file = self.find_binary_data_file()

    @classmethod
    def add_arguments(
        cls,
        parser: argparse.ArgumentParser,
    ) -> None:
        """
        Add arguments specific to this analysis.

        Subclasses override this method when they need CLI arguments.
        """

    def result(self) -> Any:
        return self

    def run(self) -> None:
        self.analyze_file(self.binary_file)

    def load_config(self) -> dict[str, Any]:
        with self.config_path.open("r") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise TypeError(f"Expected {self.config_path} to contain a YAML mapping")

        return config

    @abstractmethod
    def binary_data_file_matches(self, path: Path) -> bool:
        raise NotImplementedError

    def find_binary_data_file(self) -> Path:
        matching_files = [
            path
            for path in self.folder_path.iterdir()
            if path.is_file() and self.binary_data_file_matches(path)
        ]

        if not matching_files:
            raise FileNotFoundError(
                f"No matching binary file found in {self.folder_path}"
            )

        if len(matching_files) > 1:
            raise RuntimeError(
                f"Expected exactly one matching binary file in "
                f"{self.folder_path}, but found {len(matching_files)}: "
                f"{matching_files}"
            )

        return matching_files[0]

    def extract_meta(self) -> Any:
        meta_config = deepcopy(self.config)

        for path in self.ignored_meta_paths:
            remove_path(meta_config, path)

        return make_hashable(meta_config)

    def extract_output_quantities(self) -> list[str]:
        matches: list[list[str]] = []

        for section in self.output_sections:
            quantities = (
                self.config.get("Output", {}).get(section, {}).get("Quantities")
            )

            if quantities is None:
                continue

            if not isinstance(quantities, list) or not all(
                isinstance(quantity, str) for quantity in quantities
            ):
                raise TypeError(
                    f"Output.{section}.Quantities must be a list of strings"
                )

            matches.append(quantities)

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one Quantities section, found {len(matches)}"
            )

        return matches[0]

    @abstractmethod
    def analyze_file(self, file_path: str | Path) -> None:
        pass

    @abstractmethod
    def __iadd__(self, other: Self) -> Self:
        pass


def remove_path(data: dict[str, Any], path: tuple[str, ...]) -> None:
    current = data

    for key in path[:-1]:
        if not isinstance(current, dict):
            return

        if key not in current:
            return

        current = current[key]

    if isinstance(current, dict):
        current.pop(path[-1], None)


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, make_hashable(subvalue)) for key, subvalue in sorted(value.items())
        )

    if isinstance(value, list):
        return tuple(make_hashable(item) for item in value)

    return value
