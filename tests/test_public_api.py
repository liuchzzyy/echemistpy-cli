from echemistpy.data import RAW_SCHEMA, names


def test_data_schema_names() -> None:
    assert RAW_SCHEMA == "echemistpy-raw-v1"
    assert names("xas") == ("energy_ev", "absorption", "norm_absorption", "e0_ev", "edge_step")


def test_echem_schema_uses_plain_names() -> None:
    echem_names = names("echem")

    assert "neg_im_z_ohm" in echem_names
    assert "q_mah" in echem_names
    assert "work_mode" in echem_names
    assert "energy_uwh" in echem_names
    assert "humidity_percent" in echem_names
    assert "-im_z_ohm" not in echem_names
