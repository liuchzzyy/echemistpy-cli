"""echemistpy 数据的标准名称和 schema 标识。"""

from __future__ import annotations

RAW_SCHEMA = "echemistpy-raw-v1"
ANALYSIS_SCHEMA = "echemistpy-analysis-v1"

GENERAL_NAMES = (
    "record",
    "time_s",
    "timestamp",
    "systime",
)

ECHEM_NAMES = (
    "cycle_number",
    "step_number",
    "work_mode",
    "flags",
    "mode",
    "ox_red",
    "step_in_process",
    "step_duration_s",
    "step_time_s",
    "test_time_s",
    "current_range",
    "half_cycle",
    "control_voltage_v",
    "control_current_ma",
    "voltage_v",
    "ewe_v",
    "abs_ewe_v",
    "ece_v",
    "current_ma",
    "current_ua",
    "abs_current_ma",
    "q_mah",
    "dq_mah",
    "charge_c",
    "dq_c",
    "capacity_mah",
    "capacity_uah",
    "charge_discharge_capacity_mah",
    "charge_capacity_mah",
    "discharge_capacity_mah",
    "specific_capacity_mah_g",
    "specific_capacity_cal_mah_g",
    "energy_uwh",
    "specific_energy_mwh_g",
    "power_uw",
    "frequency_hz",
    "re_z_ohm",
    "neg_im_z_ohm",
    "z_mag_ohm",
    "phase_deg",
    "capacitance_series_uf",
    "capacitance_parallel_uf",
    "dqdv_uah_v",
    "dvdq_v_uah",
    "temperature_c",
    "humidity_percent",
    "mark_1",
    "mark_2",
    "battery_code",
    "data_file",
    "test_name",
    "process_name",
    "thickness_mm",
    "thickness_pressure_g",
    "thickness_temperature_c",
    "channel_number",
    "step_duration_ms",
    "step_time_ms",
    "test_time_ms",
    "step_start_test_time_ms",
    "step_wall_start_ms",
    "state_word",
    "page_offset",
    "record_offset",
    "page_valid_records",
)

XAS_NAMES = (
    "energy_ev",
    "absorption",
    "norm_absorption",
    "e0_ev",
    "edge_step",
)

XRD_NAMES = (
    "two_theta_deg",
    "intensity",
    "intensity_error",
    "d_spacing_angstrom",
)

XPS_NAMES = (
    "be_ev",
    "intensity_cps",
)

TGA_NAMES = (
    "temperature_c",
    "weight_percent",
    "time_min",
)

TXM_NAMES = (
    "energy_ev",
    "x_um",
    "y_um",
    "transmission",
    "optical_density",
    "cluster_label",
)

NAMES_BY_DOMAIN = {
    "general": GENERAL_NAMES,
    "echem": ECHEM_NAMES,
    "tga": TGA_NAMES,
    "xas": XAS_NAMES,
    "xps": XPS_NAMES,
    "xrd": XRD_NAMES,
    "txm": TXM_NAMES,
}

ALL_NAMES = frozenset(name for names in NAMES_BY_DOMAIN.values() for name in names)


def names(domain: str | None = None) -> tuple[str, ...]:
    """返回指定域或全部域的标准名称。"""
    if domain is None:
        return tuple(sorted(ALL_NAMES))
    return NAMES_BY_DOMAIN[domain.lower()]


__all__ = [
    "ALL_NAMES",
    "ANALYSIS_SCHEMA",
    "ECHEM_NAMES",
    "GENERAL_NAMES",
    "NAMES_BY_DOMAIN",
    "RAW_SCHEMA",
    "TGA_NAMES",
    "TXM_NAMES",
    "XAS_NAMES",
    "XPS_NAMES",
    "XRD_NAMES",
    "names",
]
