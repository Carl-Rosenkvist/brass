import numpy as np
import brass as br


class CountUnwounded:
    def __init__(self):
        self.n_events = 0
        self.unwounded = []
        self.nucleon_pdgs = np.array([2212, 2112], dtype=int)

    def on_interaction_block(self, iblock, accessor, opts):
        pass

    def on_end_block(self, block, accessor, opts):
        pass

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1
        pairs = accessor.gather_block_arrays(block)
        cols = {k: v for k, v in pairs}

        pdg = cols["pdg"]
        ncoll = cols["ncoll"]

        sel = (ncoll == 0) & np.isin(pdg, self.nucleon_pdgs)
        self.unwounded.append(int(np.count_nonzero(sel)))

    def to_state_dict(self):
        return {
            "n_events": int(self.n_events),
            "unwounded": np.asarray(self.unwounded, dtype=int),
        }

    def finalize(self, results):
        return results


br.register_python_analysis(
    "count_unwounded",
    lambda: CountUnwounded(),
    {},
)
