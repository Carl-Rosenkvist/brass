#!/usr/bin/env python3
from scan import Scan

scan = Scan()
scan.set_param("General.Nevents", [1])
scan.set_param("Modi.Collider.Sqrtsnn", [17.3])
scan.set_param("Collision_Term.String_Parameters.String_Alpha", [0.1, 1])


with open("commands.txt", "w") as f:
    for combo, cmd in scan.sweep_cmds():
        f.write(cmd + "\n")
