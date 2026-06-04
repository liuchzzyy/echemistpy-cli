from echemistpy.io import list_reader_specs, list_supported_formats
from echemistpy.io.plugin_manager import get_plugin_manager


def test_reader_specs_list_formats() -> None:
    formats = list_supported_formats()

    assert ".mpt" in formats
    assert ".mpr" in formats
    assert ".ccs" in formats
    assert "biologic_mpr" in formats[".mpr"]
    assert "biologic_mpt" in formats[".mpt"]
    assert "lanhe_ccs" in formats[".ccs"]


def test_reader_specs_are_declared() -> None:
    specs = {spec.name: spec for spec in list_reader_specs()}

    assert specs["lanhe_ccs"].extensions == (".ccs",)
    assert specs["lanhe_xlsx"].instruments == ("lanhe",)
    assert specs["biologic_mpr"].extensions == (".mpr",)
    assert specs["biologic_mpt"].extensions == (".mpt",)


def test_reader_selection_uses_declared_instruments() -> None:
    manager = get_plugin_manager()

    assert manager.get_loader(".mpr", instrument="mpr") is None
    reader = manager.get_loader(".mpr", instrument="biologic")
    assert reader is not None
    assert reader.__name__ == "BiologicMprReader"
