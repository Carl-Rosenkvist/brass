from writing_utils import*
# tests/test_write_config_yaml.py
import yaml

def test_write_config_yaml_roundtrip(tmp_path):
    # Arrange: dotted key overrides
    written = {
        "Modi.Collider.Sqrtsnn": 17.3,
        "Modi.Collider.Projectile.Particles.2212": 82,
        "Modi.Collider.Projectile.Particles.2112": 126,
        "Modi.Collider.Target.Particles.2212": 82,
        "Modi.Collider.Target.Particles.2112": 126,
        
        "Output.Particles.Quantities": ["pdg", "pz", "p0"],
    }
    expected = {
        "Modi": {
            "Collider": {
                "Sqrtsnn": 17.3,
                "Projectile": {"Particles": {2212: 82, 2112: 126}},
                "Target": {"Particles": {2212: 82, 2112: 126}},
            }
        },
        "Output": {
            "Particles": {"Quantities": ["pdg", "pz", "p0"]}
        }
    }

    path = tmp_path / "config.yaml"

    # Act: write and read back
    write_config_yaml(path, written)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    assert data == expected
