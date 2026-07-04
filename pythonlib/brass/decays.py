import numpy as np


class DecayReconstructor:
    """
    Reconstruct two-body decays from particle-level arrays.

    Assumption:
    each decay is uniquely identified by proc_id, and each relevant proc_id
    has one daughter of each requested species.
    """

    quantities = [
        "pdg",
        "proc_id_origin",
        "pdg_mother1",
        "pdg_mother2",
    ]

    def __init__(self, mother_pdg, daughter_pdgs):
        if len(daughter_pdgs) != 2:
            raise ValueError("Only two-body decays supported")

        self.mother_pdg = int(mother_pdg)
        self.daughter_pdgs = [int(pdg) for pdg in daughter_pdgs]

    def reconstruct(self, pdg, proc_id, pdg_mother1, pdg_mother2):
        d1, d2 = self.daughter_pdgs

        base = (pdg_mother1 == self.mother_pdg) & (pdg_mother2 == 0)

        idx1 = np.flatnonzero(base & (pdg == d1))
        idx2 = np.flatnonzero(base & (pdg == d2))

        _, pos1, pos2 = np.intersect1d(
            proc_id[idx1],
            proc_id[idx2],
            return_indices=True,
        )

        return idx1[pos1], idx2[pos2]

    def reconstruct_block(self, block):
        c = block.particles.columns()

        return self.reconstruct(
            c["pdg"],
            c["proc_id_origin"],
            c["pdg_mother1"],
            c["pdg_mother2"],
        )
