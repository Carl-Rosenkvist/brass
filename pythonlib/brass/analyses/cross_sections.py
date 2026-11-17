import math
from collections import Counter
from pathlib import Path
from typing import Callable, Self

import numpy as np

import brass

from .collision_counts import COLLISION_FILES

PDGs = tuple[int, ...]
# (process type of the first interaction, its outgoing PDGs, final-state PDGs)
EventKey = tuple[int, PDGs, PDGs]

STRINGS = range(41, 50)  # soft (41-45) and hard (46-49) strings
FAILED_STRING = 50
KAONS = {321, -321, 311, -311}
FM2_TO_MB = 10.0
# resonances reconstructed from their decay products by default, as in the
# SMASH analysis suite: phi, Delta(1232), rho, eta, eta', omega
RESONANCES = (333, 1114, 2114, 2214, 2224, 113, 213, -213, 221, 331, 223)


def reconstructed_final_states(interactions, final_ids, pdg_of, resonances) -> list[PDGs]:
    """Final states with one resonance put back in place of its decay products.

    `interactions` are (incoming ids, outgoing ids) of one event in order,
    `final_ids` the particles at the end, `pdg_of` maps every id to its PDG
    code. For every particle of a PDG code in `resonances` that decayed, the
    stable particles it ends up in are replaced by the resonance, giving one
    final state per resonance. A resonance counts only if it could be seen
    in its decay products: none of them, nor anything between, took part in
    an interaction with other particles (a rescattering, a formation); and
    all of them are in the final state.

    Elastic collisions keep the particle IDs; a particle listed among its own
    mothers is not counted as its own daughter.
    """
    mothers: dict[int, list[int]] = {}
    daughters: dict[int, set[int]] = {}
    for incoming, outgoing in interactions:
        for i in incoming:
            daughters.setdefault(i, set()).update(outgoing)
        for o in outgoing:
            mothers.setdefault(o, []).extend(incoming)

    def products(i):
        return daughters.get(i, set()) - set(mothers.get(i, ()))

    final = set(final_ids)
    states = []
    for r in sorted(daughters):
        if pdg_of[r] not in resonances or not products(r):
            continue
        stack, seen, stable, clean = list(products(r)), set(), set(), True
        while stack and clean:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            clean = len(mothers.get(node, ())) == 1
            if next_ := products(node):
                stack.extend(next_)
            else:
                stable.add(node)
        if clean and stable <= final:
            states.append(tuple(sorted([pdg_of[i] for i in final - stable] + [pdg_of[r]])))
    return states


def _rows(particles, names) -> np.ndarray:
    return np.stack([np.asarray(particles.column(q), dtype=float) for q in names], axis=1)


def transverse_distance(x1, p1, x2, p2) -> float:
    """Distance of closest approach of two straight lines, in their centre-of-mass frame.

    x and p are four-vectors (t, x, y, z) and (E, px, py, pz).
    """
    total = p1 + p2
    beta = total[1:] / total[0]
    b2 = beta @ beta

    def boost(v):
        if b2 == 0.0:
            return v
        gamma = 1.0 / math.sqrt(1.0 - b2)
        bv = beta @ v[1:]
        time = gamma * (v[0] - bv)
        space = v[1:] + ((gamma - 1.0) * bv / b2 - gamma * v[0]) * beta
        return np.concatenate([[time], space])

    dx = boost(x1)[1:] - boost(x2)[1:]
    dp = boost(p1)[1:] - boost(p2)[1:]
    dp2 = dp @ dp
    d2 = dx @ dx - ((dx @ dp) ** 2 / dp2 if dp2 > 0 else 0.0)
    return math.sqrt(max(d2, 0.0))


class CrossSections(brass.Analysis):
    """Total and partial cross sections from elementary collisions.

    Runs of one projectile and one target particle (Collider, impact
    parameter sampled uniformly up to a maximum), with collision output that
    includes the initial and final particles (`Print_Start_End: True`).

    - Total: the probability P(d) that the two particles interact, in bins of
      their transverse distance d in the centre-of-mass frame, integrated
      over the area: sigma = sum_k P_k pi (d_{k+1}^2 - d_k^2).
    - Per event with an interaction of the two initial particles, the
      process type and outgoing particles of that first interaction and the
      final-state particles are recorded. Partial cross sections are
      sigma * (selected events) / (events with an interaction).
    - Final states with unstable particles: in each such event, every
      decayed resonance of a type in --resonances whose decay products reach
      the final state without further interactions is put back in their
      place (e.g. p p -> n Delta++ -> n p pi+ also counts as n Delta++).
      An event can count once per such resonance, so these are inclusive
      cross sections of the resonances (`reconstructed_states`).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        impact = self.config.get("Modi", {}).get("Collider", {}).get("Impact", {})
        d_max = args.d_max if args.d_max is not None else impact.get("Max")
        if d_max is None or d_max <= 0:
            raise ValueError("give --d-max, or Modi.Collider.Impact.Max in the config")
        self.d_edges = np.linspace(0.0, float(d_max), args.d_bins + 1)
        self.all_events = np.zeros(args.d_bins)
        self.interacting = np.zeros(args.d_bins)
        self.events: Counter = Counter()
        self.resonances = frozenset(args.resonances)
        self.reconstructed: Counter = Counter()  # (process, first outgoing, state)
        self.initial: PDGs | None = None
        self.n_events = 0
        self.lost = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--d-max", type=float, default=None,
                            help="Largest transverse distance in fm (default: Modi.Collider.Impact.Max).")
        parser.add_argument("--d-bins", type=int, default=1000, help="Bins in transverse distance. Default: 1000")
        parser.add_argument("--resonances", type=int, nargs="*", default=list(RESONANCES),
                            help="PDG codes of the resonances to reconstruct from their decay products "
                                 "(default: phi, Delta(1232), rho, eta, eta', omega).")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in COLLISION_FILES

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities)
        momentum, position = ("p0", "px", "py", "pz"), ("t", "x", "y", "z")

        initial = final = first = None
        interactions, pdg_of = [], {}
        while (block := reader.read()) is not None:
            if isinstance(block, brass.ParticleBlock):
                if initial is None:
                    initial = block.particles
                else:
                    final = block.particles
            elif isinstance(block, brass.InteractionBlock):
                incoming = [int(i) for i in block.incoming.column("id")]
                outgoing = [int(i) for i in block.outgoing.column("id")]
                for side, ids in ((block.incoming, incoming), (block.outgoing, outgoing)):
                    pdg_of.update(zip(ids, (int(x) for x in side.column("pdg"))))
                interactions.append((incoming, outgoing))
                if first is None and initial is not None:
                    if sorted(incoming) == sorted(int(i) for i in initial.column("id")):
                        first = block
            elif isinstance(block, brass.EndBlock):
                if self._event(initial, first, final, momentum, position) and final is not None:
                    final_ids = [int(i) for i in final.column("id")]
                    pdg_of.update(zip(final_ids, (int(x) for x in final.column("pdg"))))
                    key = (int(first.process), tuple(sorted(int(x) for x in first.outgoing.column("pdg"))))
                    for state in reconstructed_final_states(interactions, final_ids, pdg_of, self.resonances):
                        self.reconstructed[(*key, state)] += 1
                initial = final = first = None
                interactions, pdg_of = [], {}

    def _event(self, initial, first, final, momentum, position) -> bool:
        """Count one event; True if it has an interaction of the initial particles."""
        if initial is None or initial.size() != 2:
            raise ValueError(
                "cross-sections needs the initial and final particles of each event "
                "(Output.Collisions.Print_Start_End: True) and two initial particles"
            )
        self.n_events += 1
        pdgs = tuple(sorted(int(x) for x in initial.column("pdg")))
        if self.initial is None:
            self.initial = pdgs
        elif self.initial != pdgs:
            raise ValueError(f"different initial particles: {self.initial} and {pdgs}")

        p, x = _rows(initial, momentum), _rows(initial, position)
        d = transverse_distance(x[0], p[0], x[1], p[1])
        k = int(d / (self.d_edges[1] - self.d_edges[0]))
        if k >= len(self.all_events):
            self.lost += 1
            return False
        self.all_events[k] += 1
        if first is None:
            return False
        self.interacting[k] += 1
        outgoing = tuple(sorted(int(x) for x in first.outgoing.column("pdg")))
        final_pdgs = tuple(sorted(int(x) for x in final.column("pdg"))) if final is not None else ()
        self.events[(int(first.process), outgoing, final_pdgs)] += 1
        return True

    def __iadd__(self, other: Self) -> Self:
        if not np.array_equal(self.d_edges, other.d_edges):
            raise ValueError("cannot add results with different --d-max or --d-bins")
        if self.initial and other.initial and self.initial != other.initial:
            raise ValueError("cannot add results of different colliding particles")
        self.initial = self.initial or other.initial
        self.all_events += other.all_events
        self.interacting += other.interacting
        if self.resonances != other.resonances:
            raise ValueError("cannot add results with different --resonances")
        self.events.update(other.events)
        self.reconstructed.update(other.reconstructed)
        self.n_events += other.n_events
        self.lost += other.lost
        return self

    # --- results ----------------------------------------------------------

    @property
    def sqrts(self) -> float:
        return float(self.config["Modi"]["Collider"]["Sqrtsnn"])

    def total(self) -> tuple[float, float]:
        """Total cross section and its statistical error in mb.

        Assumes the particles never interact beyond the largest distance.
        Bins without events take the probability interpolated from their
        neighbours (0 beyond the last bin with events).
        """
        filled = self.all_events > 0
        if not filled.any():
            return 0.0, 0.0
        centres = 0.5 * (self.d_edges[1:] + self.d_edges[:-1])
        measured = self.interacting[filled] / self.all_events[filled]
        prob = np.interp(centres, centres[filled], measured, right=0.0)
        var = np.zeros_like(prob)
        var[filled] = measured * (1 - measured) / self.all_events[filled]
        area = math.pi * np.diff(self.d_edges**2)
        sigma = FM2_TO_MB * float((prob * area).sum())
        error = FM2_TO_MB * float(np.sqrt((var * area**2).sum()))
        return sigma, error

    def cross_section(self, select: Callable[[int, PDGs, PDGs], bool],
                      reconstructed: bool = False) -> tuple[float, float]:
        """Cross section of the events with select(process, first_outgoing, final) true.

        With `reconstructed`, `final` runs over the final states with a
        resonance put back (see the class docstring) instead.
        Returns (sigma, error) in mb; the error is the Poisson error of the
        selected events and the error of the total, added in quadrature.
        """
        n_total = sum(self.events.values())
        if n_total == 0:
            return 0.0, 0.0
        counts = self.reconstructed if reconstructed else self.events
        n = sum(count for key, count in counts.items() if select(*key))
        sigma_tot, err_tot = self.total()
        sigma = sigma_tot * n / n_total
        if n == 0:
            return 0.0, 0.0
        return sigma, sigma * math.hypot(1.0 / math.sqrt(n), err_tot / sigma_tot if sigma_tot else 0.0)

    def is_elastic(self, process: int, first: PDGs, final: PDGs) -> bool:
        """Final state equal to the initial one, not after a string.

        This includes resonance formation and decay back into the initial
        particles, as in the SMASH analysis suite.
        """
        return final == self.initial and process not in STRINGS and process != FAILED_STRING

    def final_states(self) -> dict[PDGs, tuple[float, float]]:
        """Cross section per final state (sorted PDGs), elastic included."""
        states = {final for _, _, final in self.events}
        return {s: self.cross_section(lambda p, f, final, s=s: final == s) for s in sorted(states)}

    def reconstructed_states(self) -> dict[PDGs, tuple[float, float]]:
        """Cross section per final state with one reconstructed resonance (sorted PDGs)."""
        states = {state for _, _, state in self.reconstructed}
        return {s: self.cross_section(lambda p, f, state, s=s: state == s, reconstructed=True)
                for s in sorted(states)}

    def first_channels(self) -> dict[PDGs, tuple[float, float]]:
        """Cross section per outgoing state of the first interaction."""
        channels = {first for _, first, _ in self.events}
        return {c: self.cross_section(lambda p, first, f, c=c: first == c) for c in sorted(channels)}

    def process_type(self, process: int, first: PDGs, final: PDGs) -> str:
        """Class of the first interaction, as in the SMASH analysis suite."""
        if process == 1:
            return "elastic"
        if process in (2, 3):
            return "parametrized-inelastic" if KAONS & set(self.initial or ()) else "resonance"
        if 41 <= process <= 45:
            return "soft-string"
        if 46 <= process <= 49:
            return "hard-string"
        if process == FAILED_STRING:
            return "failed-string"
        return "other"

    def process_types(self) -> dict[str, tuple[float, float]]:
        """Cross section per class of the first interaction."""
        classes = {self.process_type(*key) for key in self.events}
        return {
            c: self.cross_section(lambda *key, c=c: self.process_type(*key) == c)
            for c in sorted(classes)
        }
