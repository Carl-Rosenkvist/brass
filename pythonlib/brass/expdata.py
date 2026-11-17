"""Reading the experimental data in brass's `exp_data/` folder.

Files are CSV with `# key: value` header lines, e.g.

    # observable: dndy
    # pdg: 211
    # system: pp
    # ref: A. Aduszkiewicz et al. Eur. Phys. J., C77(10):671, 2017
    y,dndy,dndy_err
    0.1,0.7436,0.09088

Other comment lines (starting with `#`, also inside quotes) are kept as
notes. Empty cells are read as NaN.

Uncertainties of a value column `v` are taken from, in this order of
preference per row: `v_err` (total); `v_stat_err` and `v_sys_err`, or
`v_sys_err_plus` / `v_sys_err_minus` (averaged), added in quadrature; the
same without the `v_` prefix (`err`, `stat_err`, `sys_err`).

    from brass import expdata

    data = expdata.find("multiplicity", "pp", 211)   # exp_data/pp/multiplicity_211.csv
    x, y, err = data.points("sqrtsnn", "multiplicity")

    for data in expdata.cross_sections("proton_proton"):  # exp_data/xs/proton_proton/*.csv
        print(data.meta["final_state"], data.points("sqrts", "sigma"))

The folder is `BRASS_EXP_DATA` if set, else `exp_data/` of the brass source
tree (an editable install).
"""

import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ENV = "BRASS_EXP_DATA"


def data_dir() -> Path:
    """The experimental-data folder."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    folder = Path(__file__).resolve().parents[2] / "exp_data"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No experimental data folder at {folder}; set {ENV} to the brass "
            "exp_data folder (it is part of the source tree, not of the wheel)."
        )
    return folder


@dataclass
class ExpData:
    path: Path
    meta: dict[str, str | list[str]]
    notes: list[str]
    columns: dict[str, np.ndarray] = field(repr=False)
    text_columns: dict[str, list[str]] = field(repr=False)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()), []))

    def error(self, value: str) -> np.ndarray:
        """Total uncertainty of a value column (see the module docstring)."""
        n = len(self)

        def col(name):
            return self.columns.get(name, np.full(n, np.nan))

        total = np.full(n, np.nan)
        for prefix in (f"{value}_", ""):
            err = col(f"{prefix}err")
            stat = col(f"{prefix}stat_err")
            sys = col(f"{prefix}sys_err")
            plus, minus = col(f"{prefix}sys_err_plus"), col(f"{prefix}sys_err_minus")
            sys = np.where(np.isnan(sys), 0.5 * (np.abs(plus) + np.abs(minus)), sys)
            quad = np.hypot(np.nan_to_num(stat), np.nan_to_num(sys))
            has_parts = ~(np.isnan(stat) & np.isnan(sys))
            candidate = np.where(~np.isnan(err), err, np.where(has_parts, quad, np.nan))
            total = np.where(np.isnan(total), candidate, total)
        return np.nan_to_num(total)

    def points(self, x: str, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """x, values and total errors, without rows where x or the value is missing."""
        xs, ys, errs = self.columns[x], self.columns[value], self.error(value)
        keep = ~(np.isnan(xs) | np.isnan(ys))
        return xs[keep], ys[keep], errs[keep]


def _is_comment(line: str) -> bool:
    return line.strip().lstrip('"').startswith("#")


def load(path: str | Path) -> ExpData:
    path = Path(path)
    meta: dict[str, str | list[str]] = {}
    notes: list[str] = []
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if _is_comment(line):
            text = line.strip().strip('"').lstrip("#").strip()
            key, sep, value = text.partition(":")
            if sep and key and " " not in key.strip() and not key.strip().startswith("http"):
                key, value = key.strip(), value.strip()
                if key in meta:
                    previous = meta[key]
                    meta[key] = (previous if isinstance(previous, list) else [previous]) + [value]
                else:
                    meta[key] = value
            else:
                notes.append(text)
        else:
            rows.append(line)

    if not rows:
        raise ValueError(f"{path}: no data")
    reader = csv.reader(io.StringIO("\n".join(rows)))
    header = [name.strip() for name in next(reader)]
    raw: dict[str, list[str]] = {name: [] for name in header}
    for row in reader:
        row = row + [""] * (len(header) - len(row))
        for name, cell in zip(header, row):
            raw[name].append(cell.strip())

    columns, text_columns = {}, {}
    for name, cells in raw.items():
        try:
            columns[name] = np.array([float(c) if c else math.nan for c in cells])
        except ValueError:
            text_columns[name] = cells
    return ExpData(path, meta, notes, columns, text_columns)


def find(observable: str, system: str, pdg: int, suffix: str = "") -> ExpData | None:
    """`exp_data/<system>/<observable>_<pdg><suffix>.csv`, or None if there is none."""
    path = data_dir() / system / f"{observable}_{pdg}{suffix}.csv"
    return load(path) if path.is_file() else None


def cross_sections(system: str) -> list[ExpData]:
    """All files of `exp_data/xs/<system>/`, e.g. system "proton_proton".

    Each has `final_state` in its header: "total", "elastic", or the sorted
    PDG codes of the final state (with `inclusive: yes` for "+ anything"),
    and the columns `sqrts` and `sigma` in mb.
    """
    folder = data_dir() / "xs" / system
    return [load(path) for path in sorted(folder.glob("*.csv"))] if folder.is_dir() else []


NUCLEON_MASS = 0.938  # GeV, as the SMASH analysis suite used for elab -> sqrt(s_NN)


def sqrtsnn(data: ExpData) -> float | None:
    """sqrt(s_NN) in GeV from a file's header: `sqrt_s`/`sqrtsnn`, or `energy`
    as `ecm<sqrt(s_NN)>` or `elb`/`elab`/`ekin<kinetic energy per nucleon in the lab>`."""
    for key in ("sqrtsnn", "sqrt_s"):
        if key in data.meta:
            number = re.match(r"\s*([\d.]+)", str(data.meta[key]))  # also "8.76 GeV"
            return float(number[1]) if number else None
    match = re.fullmatch(r"(ecm|elb|elab|ekin)([\d.]+)", str(data.meta.get("energy", "")))
    if match is None:
        return None
    kind, value = match[1], float(match[2])
    if kind == "ecm":
        return value
    return math.sqrt(2 * NUCLEON_MASS * (2 * NUCLEON_MASS + value))


def spectra(observable: str, system: str, pdg: int) -> list[tuple[float, ExpData]]:
    """All `exp_data/<system>/<observable>_<pdg>_*.csv` with their sqrt(s_NN), sorted by it.

    E.g. spectra("dndy", "pp", 211) or spectra("dndmt", "AA", -321); files
    whose energy cannot be read are left out.
    """
    out = []
    for path in sorted((data_dir() / system).glob(f"{observable}_{pdg}_*.csv")):
        data = load(path)
        energy = sqrtsnn(data)
        if energy is not None:
            out.append((energy, data))
    return sorted(out, key=lambda item: item[0])
