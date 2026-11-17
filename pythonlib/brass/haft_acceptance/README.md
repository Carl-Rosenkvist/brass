# HADES acceptance files

Single-lepton acceptance and resolution matrices of HADES, in the format of
HAFT 2.0 (HADES Acceptance Filter for Theorists, R. Holzmann, J. Weil), as
HADES provides them for comparing models with its dilepton spectra. They are
copied unchanged from the SMASH analysis suite's collection
(`smash-analysis-extras/experimental_data/HADES_dilepton_filtering`), which
took them from the HADES releases of the published spectra.

| Beam time (`--acceptance`) | File | Header |
|---|---|---|
| `p125p` | HadesSingleAcc-p125p-APR06-effgt5-RK-v2.acc | p+p 1.25 GeV, Apr06, smoothed eff > 5 %, v2 |
| `p22p` | HadesSingleAcc-p22p-JAN04-effgt5-RK-v2.acc | p+p 2.2 GeV, Jan04, smoothed eff > 5 %, v2 |
| `p35p` | HadesSingleAcc-p35p-APR07-effgt5-RK-v2.acc | p+p 3.5 GeV, Apr07 |
| `p35Nb` | HadesSingleAcc-p35Nb-SEP08-effgt5-MVA-v1.acc | p+Nb 3.5 GeV, Sep08, MVA eff > 5 %, v1 |
| `c10c` | HadesSingleAcc-c10c-AUG04-effgt5-KP-v2.acc | C+C 1 AGeV, Aug04, eff > 5 %, no p cut, v2 |
| `c20c` | HadesSingleAcc-c20c-NOV02-effgt0-KP-v1.acc | C+C 2 AGeV, Nov02, v1 |
| `ar17kcl` | HadesSingleAcc-ar17kcl-SEP05-effge5-HC-flat-v2.acc | Ar+KCl 1.76 AGeV, Sep05, hard cut, smoothed eff > 5 % (flat), v2 |

A matrix belongs to one beam time: filter a simulation only with the file of
the measurement it is compared with, together with that measurement's lepton
momentum and opening-angle cuts (see the HADES publications and
`exp_data/README.md`). The files are HADES's; cite the HADES publication of
the spectrum and HAFT when using them.
