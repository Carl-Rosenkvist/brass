#!/usr/bin/env python3
import brass as br

scan = br.Scan()
import numpy as np

# Independent params as before
# scan.set_param("Collision_Term.String_Parameters.String_Alpha", [0.1, 0.2])
max_events = 100


energies = np.linspace(2, 5, 5).tolist()
events = [int(max_events / e) for e in energies]  # 100, 50, 33, 25, 20
scan.set_param("Modi.Collider.Sqrtsnn", energies, events=events, max_events=max_events)
# → produces two lines for 17.3 GeV: Nevents=10 and Nevents=20
# and they will still combine (cartesian) with your other independent params.

with open("commands.txt", "w") as f:
    for combo, cmd in scan.sweep_cmds():
        f.write(cmd + "\n")
