from pathlib import Path
import pandas as pd
from io import StringIO

EXP_DIR = Path("/home/carl/Phd/brass/exp_data/AA")

for path in sorted(EXP_DIR.glob("dndy_*.csv")):

    text = path.read_text()
    lines = text.splitlines()

    metadata = [
        line
        for line in lines
        if line.lstrip().startswith("#") and not line.lstrip().startswith("# columns:")
    ]

    data_lines = [
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    ]

    if not data_lines:
        print(f"EMPTY: {path.name}")
        continue

    # Already converted
    if data_lines[0].replace(" ", "") == "y,dndy,dndy_err":
        print(f"SKIP: {path.name}")
        continue

    # Read whitespace-separated numeric data
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=["y", "dndy", "dndy_err"],
    )

    if df.isna().any().any():
        raise ValueError(f"Invalid data in {path}")

    # Write comma-separated CSV with header
    with path.open("w") as f:
        for line in metadata:
            f.write(line + "\n")

        df.to_csv(
            f,
            index=False,
            float_format="%.8g",
        )

    print(f"CONVERTED: {path.name}")
