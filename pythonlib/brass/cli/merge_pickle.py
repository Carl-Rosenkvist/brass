import pickle
import argparse
from pathlib import Path  # use Python's built-in Path
import brass as br


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Merge a list of pickle files where leafs are added using + operator"
    )

    parser.add_argument(
        "--input", "-i", nargs="+", required=True, help="List of pickle files to merge"
    )
    parser.add_argument("--output", "-o", required=True, help="Output pickle file")

    args = parser.parse_args(argv)

    output = Path(args.output).resolve()

    dicts = []
    for p in args.input:
        p = Path(p).resolve()
        with open(p, "rb") as f:
            dicts.append(pickle.load(f))

    merged_dict = br.merge_state_list(dicts)

    with open(output, "wb") as f:
        pickle.dump(merged_dict, f)

    print(f"Merged {len(args.input)} files → {output}")


if __name__ == "__main__":
    main()
