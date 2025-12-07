import pickle
import pytest

import brass as br
from brass.cli.merge_pickle import main


def write_pickle(path, obj):
    """Small helper so we don't repeat boilerplate."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_main_real_simple_merge(tmp_path):
    """End-to-end: 2 files, check against brass.merge_state_list."""
    f1 = tmp_path / "a.pkl"
    f2 = tmp_path / "b.pkl"

    d1 = {"a": 1}
    d2 = {"b": 2}

    write_pickle(f1, d1)
    write_pickle(f2, d2)

    out = tmp_path / "out.pkl"

    # act
    main(["--input", str(f1), str(f2), "--output", str(out)])

    # assert
    assert out.exists()
    result = read_pickle(out)

    expected = br.merge_state_list([d1, d2])
    assert result == expected


def test_main_three_files(tmp_path):
    """More inputs, make sure all are considered."""
    f1 = tmp_path / "a.pkl"
    f2 = tmp_path / "b.pkl"
    f3 = tmp_path / "c.pkl"

    d1 = {"a": 1}
    d2 = {"b": 2}
    d3 = {"c": 3}

    write_pickle(f1, d1)
    write_pickle(f2, d2)
    write_pickle(f3, d3)

    out = tmp_path / "out.pkl"

    main(
        [
            "--input",
            str(f1),
            str(f2),
            str(f3),
            "--output",
            str(out),
        ]
    )

    result = read_pickle(out)
    expected = br.merge_state_list([d1, d2, d3])
    assert result == expected


def test_main_nested_dicts(tmp_path):
    """Make sure more complex data structures don't break the CLI layer."""
    f1 = tmp_path / "a.pkl"
    f2 = tmp_path / "b.pkl"

    d1 = {"x": {"y": 1}}
    d2 = {"x": {"z": 2}, "k": [1, 2, 3]}

    write_pickle(f1, d1)
    write_pickle(f2, d2)

    out = tmp_path / "out.pkl"

    main(["--input", str(f1), str(f2), "--output", str(out)])

    result = read_pickle(out)
    expected = br.merge_state_list([d1, d2])
    assert result == expected


def test_main_fails_when_input_file_missing(tmp_path):
    """Non-existent input should raise an error (FileNotFoundError)."""
    missing = tmp_path / "does_not_exist.pkl"
    out = tmp_path / "out.pkl"

    with pytest.raises(FileNotFoundError):
        main(["--input", str(missing), "--output", str(out)])


def test_main_fails_on_invalid_pickle(tmp_path):
    """If a file is not a valid pickle, we should see an UnpicklingError."""
    bad = tmp_path / "bad.pkl"
    # Write some garbage
    bad.write_text("this is not a pickle\n")

    out = tmp_path / "out.pkl"

    with pytest.raises(pickle.UnpicklingError):
        main(["--input", str(bad), "--output", str(out)])


def test_main_requires_input_arg(tmp_path):
    """argparse should error if --input is missing."""
    out = tmp_path / "out.pkl"

    with pytest.raises(SystemExit) as excinfo:
        main(["--output", str(out)])

    # argparse uses exit code 2 for usage errors
    assert excinfo.value.code == 2


def test_main_requires_output_arg(tmp_path):
    """argparse should error if --output is missing."""
    f1 = tmp_path / "a.pkl"
    write_pickle(f1, {"a": 1})

    with pytest.raises(SystemExit) as excinfo:
        main(["--input", str(f1)])

    assert excinfo.value.code == 2


def test_main_overwrites_existing_output(tmp_path):
    """If the output file already exists, it should be overwritten."""
    f1 = tmp_path / "a.pkl"
    write_pickle(f1, {"a": 1})

    out = tmp_path / "out.pkl"
    # pre-create output with a dummy value
    write_pickle(out, {"old": True})

    main(["--input", str(f1), "--output", str(out)])

    result = read_pickle(out)
    expected = br.merge_state_list([{"a": 1}])
    assert result == expected
