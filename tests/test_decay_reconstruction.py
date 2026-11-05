import numpy as np
import pytest

from brass import DecayReconstructor


def reconstruct(
    pdg,
    proc_id,
    pdg_mother1,
    pdg_mother2,
    mother=333,
    daughters=(321, -321),
):
    reco = DecayReconstructor(mother, daughters)

    return reco.reconstruct(
        pdg=np.array(pdg, dtype=np.int32),
        proc_id=np.array(proc_id, dtype=np.int32),
        pdg_mother1=np.array(pdg_mother1, dtype=np.int32),
        pdg_mother2=np.array(pdg_mother2, dtype=np.int32),
    )


def assert_daughters_equal(daughters, expected):
    idx1, idx2 = daughters
    exp1, exp2 = expected

    assert np.array_equal(idx1, np.array(exp1))
    assert np.array_equal(idx2, np.array(exp2))


def test_decay_reconstructor_phi_complete():
    daughters = reconstruct(
        pdg=[321, -321, 211],
        proc_id=[7, 7, 0],
        pdg_mother1=[333, 333, 0],
        pdg_mother2=[0, 0, 0],
    )

    assert_daughters_equal(daughters, [[0], [1]])


def test_decay_reconstructor_skips_incomplete_phi():
    idx1, idx2 = reconstruct(
        pdg=[321, 211],
        proc_id=[7, 0],
        pdg_mother1=[333, 0],
        pdg_mother2=[0, 0],
    )

    assert len(idx1) == 0
    assert len(idx2) == 0


def test_decay_reconstructor_matches_by_proc_id():
    daughters = reconstruct(
        pdg=[321, 321, -321, -321],
        proc_id=[10, 20, 20, 10],
        pdg_mother1=[333, 333, 333, 333],
        pdg_mother2=[0, 0, 0, 0],
    )

    idx1, idx2 = daughters

    proc_id = np.array([10, 20, 20, 10], dtype=np.int32)[idx1]

    assert np.array_equal(proc_id, np.array([10, 20], dtype=np.int32))
    assert_daughters_equal(daughters, [[0, 1], [3, 2]])


def test_decay_reconstructor_ignores_wrong_mother():
    idx1, idx2 = reconstruct(
        pdg=[321, -321],
        proc_id=[7, 7],
        pdg_mother1=[313, 333],
        pdg_mother2=[0, 0],
    )

    assert len(idx1) == 0
    assert len(idx2) == 0


def test_decay_reconstructor_ignores_nonzero_second_mother():
    idx1, idx2 = reconstruct(
        pdg=[321, -321],
        proc_id=[7, 7],
        pdg_mother1=[333, 333],
        pdg_mother2=[0, 999],
    )

    assert len(idx1) == 0
    assert len(idx2) == 0


def test_decay_reconstructor_ignores_unrelated_particles():
    daughters = reconstruct(
        pdg=[211, 321, 2212, -321, -211],
        proc_id=[0, 7, 0, 7, 0],
        pdg_mother1=[0, 333, 0, 333, 0],
        pdg_mother2=[0, 0, 0, 0, 0],
    )

    assert_daughters_equal(daughters, [[1], [3]])


def test_decay_reconstructor_rejects_three_body_decay():
    with pytest.raises(ValueError, match="Only two-body decays supported"):
        DecayReconstructor(999, (211, -211, 111))
