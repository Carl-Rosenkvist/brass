#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import subprocess
from itertools import repeat


def timestamp():
    return dt.datetime.now().strftime("%H:%M:%S")


def read_commands(commands_file: Path):
    cmds = []
    with commands_file.open() as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            if stripped.lstrip().startswith("#"):
                continue
            cmds.append(stripped)
    return cmds


def run_one(executable: str, cmd: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"[{timestamp()}] start {outdir}")

    logfile = outdir / "run.log"

    # Build one full string, exactly like in Bash:
    full_cmd = f'{executable} -o "{outdir}" {cmd}'
    tqdm.write(full_cmd)
    with logfile.open("wb") as log_f:
        subprocess.run(
            full_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
            shell=True,  # <-- important
            executable="/bin/bash",  # ensures Bash parsing, optional
        )

    tqdm.write(f"[{timestamp()}] done  {outdir}")


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Run an executable with a list of commands"
    )
    parser.add_argument("--outbase", required=True)
    parser.add_argument("--commands", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--show-progressbar",
        action="store_true",
        help="Show tqdm progress bar",
    )
    args = parser.parse_args(argv)

    outbase = Path(args.outbase).resolve()

    commands = read_commands(Path(args.commands).resolve())
    # you don't really need numpy here, but keeping your style:
    commands = np.array(commands, dtype=str)

    print("Starting running commands " + timestamp())

    bar_format = (
        "{l_bar}{bar}| "
        "{n_fmt}/{total_fmt} • "
        "{elapsed}<{remaining} • "
        "{rate_fmt}"
    )

    # prepare one output dir per command
    outdirs = [outbase / f"out-{i}" for i in range(1, len(commands) + 1)]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(
            tqdm(
                executor.map(run_one, repeat(args.executable), commands, outdirs),
                total=len(commands),
                desc="Running commands",
                colour="cyan",
                bar_format=bar_format,
                dynamic_ncols=True,
                smoothing=0.1,
                disable=not args.show_progressbar,
            )
        )
        print("Run completed " + timestamp())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
