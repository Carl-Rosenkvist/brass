import struct
import yaml

# ----------- writers (with ensemble number) -----------


def writeHeader(f):
    # magic(4) + major(h) + minor(h) + verlen(uint32) + ver bytes
    f.write(struct.pack("<4s", b"SMSH"))
    f.write(struct.pack("<h", 9))
    f.write(struct.pack("<h", 1))
    ver = b"SMASH-3.2.2"
    f.write(struct.pack("<I", len(ver)))
    f.write(ver)


def writeParticle(f, p):
    """
    Write a particle record by auto-detecting type:
      int   -> int32 ('i')
      float -> double ('d')
    """
    fmt = "<"  # little-endian
    values = []
    for val in p:
        if isinstance(val, int):
            fmt += "i"
            values.append(val)
        else:
            fmt += "d"
            values.append(float(val))
    f.write(struct.pack(fmt, *values))


def writeParticleBlock(f, event_number: int, ensemble_number: int, particles):
    f.write(struct.pack("<1s", b"p"))
    f.write(struct.pack("<i", int(event_number)))
    f.write(struct.pack("<i", int(ensemble_number)))
    f.write(struct.pack("<I", len(particles)))
    for part in particles:
        writeParticle(f, part)


def writeInteractionBlock(f, incoming, outgoing, rho, sigma, sigma_p, process):
    """
    Binary layout:
      'i' (1 byte)
      int32 n_in, int32 n_out
      double rho, double sigma, double sigma_p
      int32 process
      n_in particle records (writeParticle)
      n_out particle records (writeParticle)
    """
    incoming = incoming or []
    outgoing = outgoing or []

    f.write(struct.pack("<1s", b"i"))
    f.write(struct.pack("<i", int(len(incoming))))
    f.write(struct.pack("<i", int(len(outgoing))))
    f.write(struct.pack("<ddd", float(rho), float(sigma), float(sigma_p)))
    f.write(struct.pack("<i", int(process)))

    for p in incoming:
        writeParticle(f, p)
    for p in outgoing:
        writeParticle(f, p)

def writeEndBlock(
    f, event_number: int, ensemble_number: int, impact_parameter: float, empty: bool
):
    f.write(struct.pack("<1s", b"f"))
    f.write(struct.pack("<i", int(event_number)))
    f.write(struct.pack("<i", int(ensemble_number)))
    f.write(struct.pack("<d", float(impact_parameter)))
    # single binary flag byte: 0x00 empty, 0x01 not empty
    f.write(struct.pack("<1s", b"\x00" if empty else b"\x01"))


# ----------- YAML config helpers (no skeleton) -----------


def _maybe_num_key(k: str):
    # turn "2212" into int 2212 so YAML keys match your usual structure
    try:
        return int(k)
    except ValueError:
        return k


def set_deep(cfg: dict, dotted_key: str, value):
    """
    Insert a value into a nested dict, creating intermediate dicts.
    Numeric segments become integer keys (e.g. 'Particles.2212' -> {2212: ...}).
    """
    parts = [p for p in dotted_key.split(".") if p]
    d = cfg
    for p in parts[:-1]:
        p2 = _maybe_num_key(p)
        if p2 not in d or not isinstance(d[p2], dict):
            d[p2] = {}
        d = d[p2]
    d[_maybe_num_key(parts[-1])] = value


def write_config_yaml(path: str, overrides: dict):
    """
    Create a YAML file where `overrides` is a dict with dotted keys.
    Example:
      {
        "Modi.Collider.Sqrtsnn": 17.3,
        "Modi.Collider.Projectile.Particles.2212": 82,
        "Output.Particles.Quantities": ["pdg","pz","p0"],
      }
    """
    cfg = {}
    for k, v in (overrides or {}).items():
        set_deep(cfg, k, v)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
