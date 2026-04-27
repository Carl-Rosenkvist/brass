import numpy as np


class DecayReconstructor:
    """
    Reconstructs two-body decays from particle-level arrays.

    The reconstruction is based on:
      - mother PDG code
      - daughter PDG codes
      - a unique process identifier (proc_id) per decay

    Assumption:
      Each decay is uniquely identified by `proc_id`, so matching daughters
      is equivalent to matching equal `proc_id`.
    """

    def __init__(self, mother_pdg, daughter_pdgs):
        """
        Parameters
        ----------
        mother_pdg : int
            PDG code of the mother particle.
        daughter_pdgs : iterable of int
            PDG codes of the two daughter particles (length = 2).
        """
        assert len(daughter_pdgs) == 2, "Only two-body decays supported"

        self.mother_pdg = mother_pdg
        self.daughter_pdgs = list(daughter_pdgs)

    def reconstruct(self, pdg, proc_id, pdg_mother1, pdg_mother2):
        """
        Find matching daughter pairs belonging to the same decay.

        Parameters
        ----------
        pdg : np.ndarray
            PDG codes of all particles in the event.
        proc_id : np.ndarray
            Unique process identifier for each particle (same for daughters
            from the same decay).
        pdg_mother1 : np.ndarray
            PDG code of the first mother.
        pdg_mother2 : np.ndarray
            PDG code of the second mother (expected to be 0 for decays).

        Returns
        -------
        idx1 : np.ndarray
            Indices of first daughter.
        idx2 : np.ndarray
            Indices of second daughter.

        Notes
        -----
        The returned indices are aligned such that:
            proc_id[idx1[i]] == proc_id[idx2[i]]

        This allows direct reconstruction of mother quantities, e.g.:
            E_mother = E[idx1] + E[idx2]
        """

        d1, d2 = self.daughter_pdgs

        # Select particles coming from the desired mother decay
        base = (pdg_mother1 == self.mother_pdg) & (pdg_mother2 == 0)

        # Indices of each daughter species
        idx1 = np.where(base & (pdg == d1))[0]
        idx2 = np.where(base & (pdg == d2))[0]

        # Match daughters via common proc_id
        _, pos1, pos2 = np.intersect1d(
            proc_id[idx1],
            proc_id[idx2],
            return_indices=True,
        )

        return idx1[pos1], idx2[pos2]
