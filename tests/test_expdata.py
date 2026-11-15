import numpy as np
import pytest

from brass import expdata


def write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_header_metadata_notes_and_columns(tmp_path):
    path = write(
        tmp_path,
        "# observable: dndy\n# pdg: 211\n# ref: A\n# ref: B\n# free text note\n"
        '"# quoted note"\ny,dndy,dndy_err\n0.1,0.7,0.09\n0.3,0.6,0.06\n',
    )
    data = expdata.load(path)
    assert data.meta == {"observable": "dndy", "pdg": "211", "ref": ["A", "B"]}
    assert data.notes == ["free text note", "quoted note"]
    x, y, err = data.points("y", "dndy")
    np.testing.assert_allclose(x, [0.1, 0.3])
    np.testing.assert_allclose(err, [0.09, 0.06])


def test_errors_total_or_stat_and_sys(tmp_path):
    path = write(
        tmp_path,
        "sqrtsnn,v,v_err,v_stat_err,v_sys_err,source\n"
        "1,10,2,,,\"E866, PLB476\"\n"
        "2,20,,3,4,NA49\n"
        "3,30,,,,none\n",
    )
    data = expdata.load(path)
    np.testing.assert_allclose(data.error("v"), [2.0, 5.0, 0.0])
    assert data.text_columns["source"] == ["E866, PLB476", "NA49", "none"]


def test_unprefixed_and_asymmetric_errors(tmp_path):
    path = write(
        tmp_path,
        "sqrtsnn,dndy,stat_err,sys_err\n14.2,0.89,.04,\n900,1.49,0.03,0.04\n",
    )
    np.testing.assert_allclose(expdata.load(path).error("dndy"), [0.04, 0.05])

    path = write(
        tmp_path,
        "y,dndy,dndy_stat_err,dndy_sys_err_plus,dndy_sys_err_minus\n0,1,0.3,0.5,-0.3\n",
        name="asym.csv",
    )
    np.testing.assert_allclose(expdata.load(path).error("dndy"), [0.5])


def test_points_drop_missing_rows(tmp_path):
    path = write(tmp_path, "sqrtsnn,v,v_err\n1,10,1\n2,,1\n")
    x, y, _ = expdata.load(path).points("sqrtsnn", "v")
    np.testing.assert_allclose(x, [1.0])


def test_find_uses_env_folder(tmp_path, monkeypatch):
    (tmp_path / "pp").mkdir()
    write(tmp_path / "pp", "sqrtsnn,v,v_err\n1,2,3\n", name="multiplicity_211.csv")
    monkeypatch.setenv(expdata.ENV, str(tmp_path))
    assert expdata.find("multiplicity", "pp", 211).meta == {}
    assert expdata.find("multiplicity", "pp", -211) is None


def test_no_data(tmp_path):
    with pytest.raises(ValueError, match="no data"):
        expdata.load(write(tmp_path, "# only: header\n"))
