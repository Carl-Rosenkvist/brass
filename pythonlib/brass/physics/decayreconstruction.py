import numpy as np


class DecayReconstructor:
    def __init__(self, mother_pdg, daughter_pdgs):
        self.mother_pdg = mother_pdg
        self.daughter_pdgs = list(daughter_pdgs)

    def reconstruct(self, pdg, proc_id, pdg_mother1, pdg_mother2):
        """
        Returns:
            daughters: list of index arrays, one per daughter species
        """
        daughter_indices = []

        for dpid in self.daughter_pdgs:
            idx = np.where(
                (pdg == dpid) & (pdg_mother1 == self.mother_pdg) & (pdg_mother2 == 0)
            )[0]

            idx = idx[np.argsort(proc_id[idx])]
            daughter_indices.append(idx)

        if any(len(idx) == 0 for idx in daughter_indices):
            return []

        common_proc_ids = proc_id[daughter_indices[0]]
        matched_indices = [np.arange(len(daughter_indices[0]))]

        for idx in daughter_indices[1:]:
            common_proc_ids, i_common, i_next = np.intersect1d(
                common_proc_ids,
                proc_id[idx],
                return_indices=True,
            )

            matched_indices = [m[i_common] for m in matched_indices]
            matched_indices.append(i_next)

        if len(common_proc_ids) == 0:
            return []

        return [idx[m] for idx, m in zip(daughter_indices, matched_indices)]
