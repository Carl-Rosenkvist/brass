# Experimental data

This directory contains a collection of experimental data provided for convenience when performing analyses with BRASS.

The data has been collected from various sources over time. References are included where known, but some may be missing, incomplete, or incorrect. The data and uncertainties have not necessarily been independently verified.

The files are provided as-is and should be used at your own risk. For scientific or publication use, the data should be checked against the original experimental publications.

Corrections and additional references are welcome.

## Energy-scan data from the SMASH analysis suite

The files `multiplicity_<pdg>.csv`, `meanpt_<pdg>.csv`, `meanmt_<pdg>.csv`, `AA/midy_<pdg>.csv` and `pp/midy_-321.csv` (4π multiplicities, midrapidity yields, mean pT and mean mT − m0 against √s_NN) were converted from the internal data collection of the SMASH analysis suite, which was never meant to be published as a data set. The same risks apply, and more so:

- The tables were gathered by hand over many years. Values were typed or read off figures; transcription errors are possible and have not been searched for systematically.
- References are only the comment line a value was listed under in the original table (the `source` column). Some are incomplete, informal (e.g. `na61/shine`) or missing, and a few may belong to neighbouring rows.
- Uncertainties are as given there: sometimes total, sometimes statistical and systematic separately (`*_stat_err`, `*_sys_err`), sometimes zero when none was recorded. Asymmetric or correlated errors are not represented.
- `AA` mixes Au+Au and Pb+Pb, different centrality selections and detector acceptances at similar energies, as the analysis suite did for its overview plots. The files do not say which centrality a point belongs to.
- Collision energies are given as in the original tables and may be rounded or quoted per nucleon pair inconsistently between experiments.
- Rows the original tables had commented out are not included; rows marked as missing (`*`) are left out.

Use these files for overview comparisons, as the analysis suite did. Do not use them for fits, calibrations or publications without checking every point against the original publication.

## Transverse-mass spectra

`AA/dndmt_<pdg>_<energy>.csv` are the transverse-mass spectra at midrapidity, 1/m_T d²N/dm_T dy against m_T − m₀, that the SMASH analysis suite drew on its energy-scan pages, converted from its collection. The same risks apply as for the energy-scan data above, and more:

- `system: AA` mixes Au+Au and Pb+Pb; centrality is only given where the original file name said so (`_cent0-5`).
- Some files combine several experiments (e.g. STAR and PHENIX at 200 GeV); the `note` lines list them in order, but the rows do not say which experiment they belong to.
- Several were converted from p_T spectra (m_T dm_T = p_T dp_T, and a factor 2π removed, as noted), or read off figures; many points have no uncertainty (0).
- The 5.02 TeV Pb+Pb files (ALICE, 0–5 %) were rebuilt from the HEPData tables: the collection's versions held the sum of both charges multiplied by π instead of divided by p_T. ALICE measured both charges together, so the files hold half the sum for each.
- `energy` is `ecm<√s_NN>` or `elb<E_kin per nucleon in GeV>`; `expdata.sqrtsnn` converts the latter with m_N = 0.938 GeV, as the suite did.

`expdata.spectra(observable, system, pdg)` lists the files of a spectrum (`dndy`, `dndmt`) with their √s_NN, from either `energy` or `sqrt_s`.

## HADES dilepton spectra

The files `dilepton_mass_*`, `dilepton_pt_*` and `dilepton_y_*` in `pp/`, `pNb/`, `CC/` and `ArKCl/` are HADES dilepton spectra, also converted from the SMASH analysis suite's collection, with the same risks as above. They are HADES-acceptance-filtered measurements: compare them only with spectra passed through the HADES filter of the same beam time (`brass.haft` with the matching acceptance file, shipped with brass in `pythonlib/brass/haft_acceptance/`, and lepton momentum cuts). The pp and p+Nb spectra are cross sections (µb/GeV), the C+C and Ar+KCl spectra are yields per π⁰, with N_π⁰ = (N_π⁺ + N_π⁻)/2 as HADES defines it. Only statistical errors are given; the C+C systematic errors of the source were inconsistent and are left out. `cut` in the header gives mass windows (`m150_m470`: 0.15 < M < 0.47 GeV) or the pair-momentum cut of p+Nb (`m800`: p > 0.8 GeV, `m0_m800`: p < 0.8 GeV).

## FOPI pion data

`AA/fopi_pion_multiplicity.csv` and `AA/fopi_pion_ratio.csv` are the pion multiplicity M(π) = 3/2 (N_π⁺ + N_π⁻) and the ratio N_π⁻/N_π⁺ in central Au+Au collisions against the beam energy (FOPI, Nucl. Phys. A781 (2007) 459, Figs. 20 and 25), taken from the data of the published figures as collected for the SMASH analysis suite; the errors are the overall 9.4 % and 5 % quoted there.

## Elementary cross sections

`xs/<system>/` (e.g. `xs/proton_proton/`) holds the total, elastic and partial cross sections the SMASH analysis suite compared its elementary-collision runs with, converted from its collection (`brass.expdata.cross_sections(system)` reads them). `final_state` in the header is `total`, `elastic` or the sorted PDG codes of the final state; `inclusive: yes` means "and anything else" (the suite's `+X`), and `label` is the suite's name of the channel. Cross sections are in mb.

- `total.csv` and `elastic.csv` come from the PDG compilation tables (with the reference per point in `ref`; statistical and systematic errors as given there, the systematic ones converted from per cent), except for π⁺π⁻, which are HepData tables already given against the invariant mass.
- The partial cross sections were collected by hand for the suite, with the same risks as the energy-scan data above; most give only a total error and no reference beyond an occasional comment (`source`).
- The original tables give the beam momentum (`plab`) or kinetic energy (`ekin`) of the projectile, the first particle of the system name, on a target at rest. `sqrts` (and `sqrts_min`, `sqrts_max`) were computed from them with SMASH's pole masses (π 0.138, N 0.938, K 0.494 GeV), as the suite did when it plotted them. Near thresholds this choice of masses matters.
- Some channels were never seen in SMASH, or their data was found inconsistent (Λ+Σ⁰ in pp does not conserve charge as listed); they are kept as they were.
