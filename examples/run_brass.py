#!/usr/bin/env python3
import sys
import os
import argparse
import brass as br
import time
# 1) import your python analysis module so it registers itself
import dndydpt  # <- this file you showed; calling register_python_analysis(...)

# 2) Quantities must EXACTLY match what the file contains
QUANTITIES = [
    "t","x","y","z",
    "mass","p0","px","py","pz",
    "pdg","id","charge","ncoll",
    "form_time","xsecfac",
    "proc_id_origin","proc_type_origin","time_last_coll",
    "pdg_mother1","pdg_mother2",
    "baryon_number","strangeness"
]

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/particles_oscar2013_extended.bin [outdir]")
        sys.exit(1)

    binfile = sys.argv[1]
    outdir  = sys.argv[2] if len(sys.argv) > 2 else "results_py"


    t0 = time.perf_counter()
    print(br.list_analyses())
    br.run_analysis(
        file_and_meta=[(binfile, "")],          # no meta label needed
        analysis_names=["dndydpt_py"],          # <-- your Python analysis
        quantities=QUANTITIES,
        output_folder=outdir,
    )
    t1 = time.perf_counter()
    print(f"[PY] dndydpt_py elapsed: {t1-t0:.6f} s")

if __name__ == "__main__":
    main()
