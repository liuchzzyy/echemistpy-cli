"""Standard column name mappings for data standardization.

本模块定义了不同技术和仪器的标准列名映射关系。
映射按照功能模块化组织，便于维护和扩展。

标准命名约定：
- 时间相关：systime（绝对时间）、time_s（相对时间，秒）
- 电压/电势：ewe_v（工作电极）、ece_v（对电极）、voltage_v（电池电压）
- 电流：current_ma（毫安）、current_ua（微安）
- 容量/电荷：capacity_mah（毫安时）、q_mah（电荷）
- 频率：frequency_hz
- 阻抗：re_z_ohm（实部）、-im_z_ohm（虚部，负值）
"""

from __future__ import annotations

# ============================================================================
# 时间列映射
# ============================================================================

# 绝对时间（无单位，datetime 格式）
ABSOLUTE_TIME_MAPPINGS: dict[str, str] = {
    "Systime": "systime",
    "SysTime": "systime",
    "systime": "systime",
    "abs_time": "systime",
    "Absolute Time": "systime",
    "DateTime": "systime",
}

# 相对时间（标准化为秒）
RELATIVE_TIME_MAPPINGS: dict[str, str] = {
    "time": "time_s",
    "Time": "time_s",
    "TIME": "time_s",
    "t": "time_s",
    "time/s": "time_s",
    "Time/s": "time_s",
    "time_s": "time_s",
    "Time_s": "time_s",
    "test_time": "time_s",
    "TestTime": "time_s",
    "test time": "time_s",
    "Test Time": "time_s",
}

# ============================================================================
# 循环和步骤列映射
# ============================================================================

CYCLE_MAPPINGS: dict[str, str] = {
    "Ns": "cycle_number",
    "ns": "cycle_number",
    "cycle": "cycle_number",
    "Cycle": "cycle_number",
    "cycle number": "cycle_number",
    "cycle_number": "cycle_number",
    "Cycle Number": "cycle_number",
    "Cycle_Number": "cycle_number",
}

STEP_MAPPINGS: dict[str, str] = {
    "step": "step_number",
    "step_number": "step_number",
    "Step": "step_number",
    "Step Number": "step_number",
}

RECORD_MAPPINGS: dict[str, str] = {
    "record": "record",
    "record_number": "record",
    "Record": "record",
}

# ============================================================================
# 电压/电势列映射
# ============================================================================

# 工作电极电势
WORKING_ELECTRODE_MAPPINGS: dict[str, str] = {
    "potential": "ewe_v",
    "Potential": "ewe_v",
    "E": "ewe_v",
    "Ewe": "ewe_v",
    "Ewe/V": "ewe_v",
    "Ewe_V": "ewe_v",
    "potential_V": "ewe_v",
    "voltage_v": "ewe_v",
    "Potential/V": "ewe_v",
    "Potential_V": "ewe_v",
}

# 对电极电势
COUNTER_ELECTRODE_MAPPINGS: dict[str, str] = {
    "Ece/V": "ece_v",
    "Ece_V": "ece_v",
}

# 电池电压（整体电压）
BATTERY_VOLTAGE_MAPPINGS: dict[str, str] = {
    "voltage": "voltage_v",
    "Voltage": "voltage_v",
    "V": "voltage_v",
    "voltage_V": "voltage_v",
    "Voltage_V": "voltage_v",
    "voltage/V": "voltage_v",
    "Voltage/V": "voltage_v",
    "battery_voltage": "voltage_v",
    "Battery_Voltage": "voltage_v",
    "V_batt": "voltage_v",
    "Vbatt": "voltage_v",
    "cell_voltage": "voltage_v",
    "Cell_Voltage": "voltage_v",
}

# ============================================================================
# 电流列映射
# ============================================================================

# 电流（毫安）
CURRENT_MA_MAPPINGS: dict[str, str] = {
    "current": "current_ma",
    "Current": "current_ma",
    "I": "current_ma",
    "i": "current_ma",
    "current_mA": "current_ma",
    "Current_mA": "current_ma",
    "I_mA": "current_ma",
    "<I>/mA": "current_ma",
    "<I>_mA": "current_ma",
    "I/mA": "current_ma",
    "Current/mA": "current_ma",
    "control/V/mA": "current_ma",
    "current_ma": "current_ma",
    "current/mA": "current_ma",
}

# 电流（微安）
CURRENT_UA_MAPPINGS: dict[str, str] = {
    "current_ua": "current_ua",
    "Current_uA": "current_ua",
    "current/uA": "current_ua",
    "Current/uA": "current_ua",
}

# ============================================================================
# 电荷/容量列映射
# ============================================================================

# 电荷
CHARGE_MAPPINGS: dict[str, str] = {
    "charge": "q_mah",
    "Charge": "q_mah",
    "Q": "q_mah",
}

# 容量（毫安时）
CAPACITY_MAH_MAPPINGS: dict[str, str] = {
    "capacity": "capacity_mah",
    "Capacity": "capacity_mah",
    "(Q-Qo)/mA.h": "capacity_mah",
    "capacity_mah": "capacity_mah",
    "capacity/mA.h": "capacity_mah",
    "Capacity/mA.h": "capacity_mah",
    "Capacity/mAh": "capacity_mah",
    "Capacity_mA.h": "capacity_mah",
}

# 容量（微安时）
CAPACITY_UAH_MAPPINGS: dict[str, str] = {
    "capacity_uah": "capacity_uah",
    "Capacity_uAh": "capacity_uah",
    "capacity/uAh": "capacity_uah",
    "capacity/mA.h": "capacity_mah",
    "capacity/uA.h": "capacity_uah",
    "Capacity/uA.h": "capacity_uah",
    "Capacity/uAh": "capacity_uah",
}

# 比容量
SPECIFIC_CAPACITY_MAPPINGS: dict[str, str] = {
    "SpeCap/mAh/g": "specific_capacity_mah_g",
    "SpeCap_mAh_g": "specific_capacity_mah_g",
    "SpeCap_cal/mAh/g": "specific_capacity_cal_mah_g",
    "SpeCap_cal_mAh_g": "specific_capacity_cal_mah_g",
    "SpeCap_cal/mA.h/g": "specific_capacity_cal_mah_g",
    "SpeCap_cal_mA.h_g": "specific_capacity_cal_mah_g",
}

# ============================================================================
# 电化学阻抗谱 (EIS) 列映射
# ============================================================================

EIS_MAPPINGS: dict[str, str] = {
    # 频率
    "freq/Hz": "frequency_hz",
    "freq_Hz": "frequency_hz",
    "Freq_Hz": "frequency_hz",
    "frequency": "frequency_hz",
    "Frequency": "frequency_hz",
    # 阻抗实部
    "Re(Z)/Ohm": "re_z_ohm",
    "Re(Z)_Ohm": "re_z_ohm",
    "Re_Z_Ohm": "re_z_ohm",
    "Z'": "re_z_ohm",
    "Z_real": "re_z_ohm",
    # 阻抗虚部（负值）
    "-Im(Z)/Ohm": "-im_z_ohm",
    "-Im(Z)_Ohm": "-im_z_ohm",
    "-Im_Z_Ohm": "-im_z_ohm",
    "Z''": "-im_z_ohm",
    "Z_imag": "-im_z_ohm",
    # 阻抗模量
    "|Z|/Ohm": "z_mag_ohm",
    "|Z|_Ohm": "z_mag_ohm",
    "Z_mag": "z_mag_ohm",
    # 相位角
    "Phase(Z)/deg": "phase_deg",
    "Phase(Z)_deg": "phase_deg",
    "Phase_Z_deg": "phase_deg",
    "phase": "phase_deg",
    "Phase": "phase_deg",
}

# ============================================================================
# 微分容量列映射
# ============================================================================

DIFFERENTIAL_CAPACITY_MAPPINGS: dict[str, str] = {
    "dQdV/uAh/V": "dqdv_uah_v",
    "dQdV_uAh_V": "dqdv_uah_v",
    "dVdQ/V/uAh": "dvdq_v_uah",
    "dVdQ_V_uAh": "dvdq_v_uah",
}

# ============================================================================
# XRD 列映射
# ============================================================================

XRD_MAPPINGS: dict[str, str] = {
    # 2θ 角度
    "2theta": "2theta_degree",
    "2Theta": "2theta_degree",
    "angle": "2theta_degree",
    # 强度
    "intensity": "intensity",
    "Intensity": "intensity",
    "counts": "intensity",
    "Counts": "intensity",
    # d 间距
    "d-spacing": "d-spacing_angstrom",
    "d_spacing": "d-spacing_angstrom",
}

# ============================================================================
# XPS 列映射
# ============================================================================

XPS_MAPPINGS: dict[str, str] = {
    # 结合能
    "binding_energy": "be_ev",
    "BE": "be_ev",
    "energy": "be_ev",
    # 强度
    "intensity": "intensity_cps",
    "Intensity": "intensity_cps",
    "counts": "intensity_cps",
    "cps": "intensity/cps",
}

# ============================================================================
# TGA 列映射
# ============================================================================

TGA_MAPPINGS: dict[str, str] = {
    # 温度
    "temperature": "T/°C",
    "Temperature": "T/°C",
    "T": "T/°C",
    # 重量
    "weight": "weight/%",
    "Weight": "weight/%",
    "mass": "weight/%",
    # 时间
    "time": "time/min",
    "Time": "time/min",
    "t": "time/min",
}

# ============================================================================
# XAS 列映射
# ============================================================================

XAS_MAPPINGS: dict[str, str] = {
    # 能量
    "energy": "energy_eV",
    "Energy": "energy_eV",
    "energyc": "energy_eV",
    "energy_eV": "energy_eV",
    # 吸收
    "absorption": "absorption_au",
    "Absorption": "absorption_au",
}

# ============================================================================
# TXM 列映射
# ============================================================================

TXM_MAPPINGS: dict[str, str] = {
    # 能量
    "energy": "energy_eV",
    # 位置
    "x": "x_um",
    "y": "y_um",
    # 透射和光学密度
    "transmission": "transmission",
    "optical_density": "optical_density",
}


def get_echem_mappings() -> dict[str, str]:
    """获取电化学技术的完整列名映射。

    Returns:
        包含所有电化学相关列名映射的字典
    """
    echem_map = {}
    echem_map.update(ABSOLUTE_TIME_MAPPINGS)
    echem_map.update(RELATIVE_TIME_MAPPINGS)
    echem_map.update(CYCLE_MAPPINGS)
    echem_map.update(STEP_MAPPINGS)
    echem_map.update(RECORD_MAPPINGS)
    echem_map.update(WORKING_ELECTRODE_MAPPINGS)
    echem_map.update(COUNTER_ELECTRODE_MAPPINGS)
    echem_map.update(BATTERY_VOLTAGE_MAPPINGS)
    echem_map.update(CURRENT_MA_MAPPINGS)
    echem_map.update(CURRENT_UA_MAPPINGS)
    echem_map.update(CHARGE_MAPPINGS)
    echem_map.update(CAPACITY_MAH_MAPPINGS)
    echem_map.update(CAPACITY_UAH_MAPPINGS)
    echem_map.update(SPECIFIC_CAPACITY_MAPPINGS)
    echem_map.update(EIS_MAPPINGS)
    echem_map.update(DIFFERENTIAL_CAPACITY_MAPPINGS)
    return echem_map


def get_xrd_mappings() -> dict[str, str]:
    """获取 XRD 技术的列名映射。

    Returns:
        XRD 列名映射字典
    """
    return XRD_MAPPINGS.copy()


def get_xps_mappings() -> dict[str, str]:
    """获取 XPS 技术的列名映射。

    Returns:
        XPS 列名映射字典
    """
    return XPS_MAPPINGS.copy()


def get_tga_mappings() -> dict[str, str]:
    """获取 TGA 技术的列名映射。

    Returns:
        TGA 列名映射字典
    """
    return TGA_MAPPINGS.copy()


def get_xas_mappings() -> dict[str, str]:
    """获取 XAS 技术的列名映射。

    Returns:
        XAS 列名映射字典
    """
    return XAS_MAPPINGS.copy()


def get_txm_mappings() -> dict[str, str]:
    """获取 TXM 技术的列名映射。

    Returns:
        TXM 列名映射字典
    """
    return TXM_MAPPINGS.copy()


# ============================================================================
# 标准列顺序定义
# ============================================================================

# 电化学数据的标准列顺序（显示优先级）
ECHEM_PREFERRED_ORDER: list[str] = [
    "time_s",
    "systime",
    "cycle_number",
    "step_number",
    "record",
    "ewe_v",
    "ece_v",
    "voltage_v",
    "current_ma",
    "current_ua",
    "capacity_mah",
    "capacity_uah",
    "q_mah",
    "specific_capacity_mah_g",
    "specific_capacity_cal_mah_g",
    "frequency_hz",
    "re_z_ohm",
    "-im_z_ohm",
    "z_mag_ohm",
    "phase_deg",
    "dqdv_uah_v",
    "dvdq_v_uah",
]


__all__ = [
    "ABSOLUTE_TIME_MAPPINGS",
    "BATTERY_VOLTAGE_MAPPINGS",
    "CAPACITY_MAH_MAPPINGS",
    "CAPACITY_UAH_MAPPINGS",
    "CHARGE_MAPPINGS",
    "COUNTER_ELECTRODE_MAPPINGS",
    "CURRENT_MA_MAPPINGS",
    "CURRENT_UA_MAPPINGS",
    "CYCLE_MAPPINGS",
    "DIFFERENTIAL_CAPACITY_MAPPINGS",
    "ECHEM_PREFERRED_ORDER",
    "EIS_MAPPINGS",
    "RECORD_MAPPINGS",
    "RELATIVE_TIME_MAPPINGS",
    "SPECIFIC_CAPACITY_MAPPINGS",
    "STEP_MAPPINGS",
    "TGA_MAPPINGS",
    "TXM_MAPPINGS",
    "WORKING_ELECTRODE_MAPPINGS",
    "XAS_MAPPINGS",
    "XPS_MAPPINGS",
    "XRD_MAPPINGS",
    "get_echem_mappings",
    "get_tga_mappings",
    "get_txm_mappings",
    "get_xas_mappings",
    "get_xps_mappings",
    "get_xrd_mappings",
]
