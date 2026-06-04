from typer.testing import CliRunner

from echemistpy.cli.app import app
from echemistpy.cli.commands.doctor import check_runtime

SAMPLE_CCS = "Samples/Echem/Lanhe/AA.ccs"
SAMPLE_MPR = "Samples/Echem/Biologic/Trial03_EIS/EMD-2V-2mAh-1M+02M-40mL_02_PEIS_C01.mpr"
BIOLOGIC_DIR = "Samples/Echem/Biologic"


def test_cli_formats_command() -> None:
    result = CliRunner().invoke(app, ["echem", "formats"])

    assert result.exit_code == 0
    assert "lanhe_ccs" in result.output
    assert "biologic_mpr" in result.output
    assert "claess_dat" not in result.output
    assert "mspd_xye" not in result.output
    assert "mistral_hdf5" not in result.output
    assert "directory=yes" in result.output


def test_cli_domain_formats_commands() -> None:
    runner = CliRunner()

    xas = runner.invoke(app, ["xas", "formats"])
    xrd = runner.invoke(app, ["xrd", "formats"])
    txm = runner.invoke(app, ["txm", "formats"])

    assert xas.exit_code == 0
    assert "claess_dat" in xas.output
    assert "biologic_mpr" not in xas.output

    assert xrd.exit_code == 0
    assert "mspd_xye" in xrd.output
    assert "lanhe_ccs" not in xrd.output

    assert txm.exit_code == 0
    assert "mistral_hdf5" in txm.output
    assert "biologic_mpt" not in txm.output


def test_doctor_required_checks_pass() -> None:
    checks = {label: state for state, label, _ in check_runtime()}

    assert checks["package"] == "ok"
    assert checks["data_schema"] == "ok"
    assert checks["readers"] == "ok"


def test_cli_inspect_sample_ccs() -> None:
    result = CliRunner().invoke(app, ["echem", "inspect", SAMPLE_CCS, "--instrument", "lanhe"])

    assert result.exit_code == 0
    assert "Schema: echemistpy-raw-v1" in result.output
    assert "cycle_number" in result.output


def test_cli_inspect_sample_mpr() -> None:
    result = CliRunner().invoke(app, ["echem", "inspect", SAMPLE_MPR, "--instrument", "biologic"])

    assert result.exit_code == 0
    assert "Technique: echem,eis,peis" in result.output
    assert "frequency_hz" in result.output


def test_cli_inspect_biologic_directory() -> None:
    result = CliRunner().invoke(app, ["echem", "inspect", BIOLOGIC_DIR, "--instrument", "biologic"])

    assert result.exit_code == 0
    assert "Dims: {'record':" in result.output
    assert "ewe_v" in result.output


def test_cli_convert_sample_ccs(tmp_path) -> None:
    output = tmp_path / "sample.csv"
    result = CliRunner().invoke(app, ["echem", "convert", SAMPLE_CCS, "--instrument", "lanhe", "--out", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    assert "cycle_number" in output.read_text(encoding="utf-8").splitlines()[0]
