import numpy as np

from brass import DecayReconstructor


import numpy as np

from brass import DecayReconstructor


def reconstruct(
    pdg, proc_id, pdg_mother1, pdg_mother2, mother=333, daughters=(321, -321)
):
    reco = DecayReconstructor(mother, daughters)
    return reco.reconstruct(
        pdg=np.array(pdg),
        proc_id=np.array(proc_id),
        pdg_mother1=np.array(pdg_mother1),
        pdg_mother2=np.array(pdg_mother2),
    )


def assert_daughters_equal(daughters, expected):
    assert len(daughters) == len(expected)
    for got, exp in zip(daughters, expected):
        assert np.array_equal(got, np.array(exp))


def test_decay_reconstructor_phi_complete():
    daughters = reconstruct(
        pdg=[321, -321, 211],
        proc_id=[7, 7, 0],
        pdg_mother1=[333, 333, 0],
        pdg_mother2=[0, 0, 0],
    )

    assert_daughters_equal(daughters, [[0], [1]])


def test_decay_reconstructor_skips_incomplete_phi():
    daughters = reconstruct(
        pdg=[321, 211],
        proc_id=[7, 0],
        pdg_mother1=[333, 0],
        pdg_mother2=[0, 0],
    )

    assert daughters == []


def test_decay_reconstructor_matches_by_proc_id():
    daughters = reconstruct(
        pdg=[321, 321, -321, -321],
        proc_id=[10, 20, 20, 10],
        pdg_mother1=[333, 333, 333, 333],
        pdg_mother2=[0, 0, 0, 0],
    )

    kplus, kminus = daughters
    assert list(zip([10, 20], [10, 20])) == [(10, 10), (20, 20)]
    assert_daughters_equal(daughters, [[0, 1], [3, 2]])


def test_decay_reconstructor_ignores_wrong_mother():
    daughters = reconstruct(
        pdg=[321, -321],
        proc_id=[7, 7],
        pdg_mother1=[313, 333],
        pdg_mother2=[0, 0],
    )

    assert daughters == []


def test_decay_reconstructor_ignores_nonzero_second_mother():
    daughters = reconstruct(
        pdg=[321, -321],
        proc_id=[7, 7],
        pdg_mother1=[333, 333],
        pdg_mother2=[0, 999],
    )

    assert daughters == []


def test_decay_reconstructor_ignores_unrelated_particles():
    daughters = reconstruct(
        pdg=[211, 321, 2212, -321, -211],
        proc_id=[0, 7, 0, 7, 0],
        pdg_mother1=[0, 333, 0, 333, 0],
        pdg_mother2=[0, 0, 0, 0, 0],
    )

    assert_daughters_equal(daughters, [[1], [3]])


def test_decay_reconstructor_three_body_decay():
    daughters = reconstruct(
        pdg=[211, -211, 111, 321],
        proc_id=[42, 42, 42, 0],
        pdg_mother1=[999, 999, 999, 0],
        pdg_mother2=[0, 0, 0, 0],
        mother=999,
        daughters=(211, -211, 111),
    )

    assert_daughters_equal(daughters, [[0], [1], [2]])
