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

## HADES dilepton spectra

The files `dilepton_mass_*`, `dilepton_pt_*` and `dilepton_y_*` in `pp/`, `pNb/`, `CC/` and `ArKCl/` are HADES dilepton spectra, also converted from the SMASH analysis suite's collection, with the same risks as above. They are HADES-acceptance-filtered measurements: compare them only with spectra passed through the HADES filter of the same beam time (`brass.haft` with the matching acceptance file and lepton momentum cuts). The pp and p+Nb spectra are cross sections (µb/GeV), the C+C and Ar+KCl spectra are yields per π⁰, with N_π⁰ = (N_π⁺ + N_π⁻)/2 as HADES defines it. Only statistical errors are given; the C+C systematic errors of the source were inconsistent and are left out. `cut` in the header gives mass windows (`m150_m470`: 0.15 < M < 0.47 GeV) or the pair-momentum cut of p+Nb (`m800`: p > 0.8 GeV, `m0_m800`: p < 0.8 GeV).
