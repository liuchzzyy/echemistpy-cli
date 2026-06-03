#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa: N999
"""CCS binary reader for LANHE/LAND battery test files."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import xarray as xr

from echemistpy.io.base_reader import BaseReader
from echemistpy.io.plugins.ccs_parser import CCSParser
from echemistpy.io.reader_utils import sanitize_variable_names
from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class LanheCCSReader(BaseReader):
    """Reader for LANHE/LAND ``.ccs`` binary files.

    The binary layout is reverse-engineered from the vendor viewer/exporter. The
    parser produces the same record-level quantities used by the official XLSX
    export for the sample data: voltage, current, cumulative capacity, cumulative
    energy, dQ/dV, dV/dQ, timestamps, and step metadata.
    """

    supports_directories: ClassVar[bool] = True
    instrument: ClassVar[str] = "lanhe"

    MASS_REGEX: ClassVar[re.Pattern[str]] = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(mg|g|ug|µg)", re.IGNORECASE)

    def __init__(self, filepath: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the LANHE CCS reader."""
        if "technique" not in kwargs:
            kwargs["technique"] = ["echem", "gcd"]
        super().__init__(filepath, **kwargs)

    def _load_single_file(self, path: Path, **_kwargs: Any) -> tuple[RawData, RawDataInfo]:
        """Load one LANHE/LAND ``.ccs`` file."""
        parser = CCSParser(str(path)).parse()
        if not parser.measurements:
            raise ValueError(f"No measurement records found in CCS file: {path}")

        mass_g = self._parse_mass_g(self.active_material_mass)
        ds = self._create_dataset(parser, mass_g)
        raw_info = self._create_info(path, parser, mass_g)
        return RawData(data=ds), raw_info

    def _create_dataset(self, parser: CCSParser, mass_g: float | None = None) -> xr.Dataset:
        """Convert parsed CCS measurements to an xarray Dataset."""
        measurements = parser.measurements
        step_duration_ms = self._step_duration_by_key(measurements)

        data: dict[str, list[Any]] = {
            "Cycle": [],
            "Step": [],
            "Record": [],
            "WorkMode": [],
            "StepInProcess": [],
            "StepDuration_s": [],
            "StepTime_s": [],
            "TestTime_s": [],
            "SysTime": [],
            "Voltage_V": [],
            "Current_uA": [],
            "Capacity_uAh": [],
            "Energy_uWh": [],
            "Power_uW": [],
            "dQdV_uAh_V": [],
            "dVdQ_V_uAh": [],
            "Temperature_℃": [],
            "Humidity_%": [],
            "Mark1": [],
            "Mark2": [],
            "DataFile": [],
            "TestName": [],
            "ProcessName": [],
            "ChannelNumber": [],
        }

        if mass_g is not None and mass_g > 0:
            data["SpeCap_mAh_g"] = []
            data["SpeEnergy_mWh_g"] = []

        for measurement in measurements:
            step_key = (measurement["Step"], measurement["StepInProcess"])
            capacity_uah = float(measurement.get("CumCapacity_uAh", 0.0))
            energy_uwh = float(measurement.get("CumEnergy_uWh", 0.0))

            data["Cycle"].append(1)
            data["Step"].append(measurement["Step"])
            data["Record"].append(measurement["Record"])
            data["WorkMode"].append(measurement["WorkMode"])
            data["StepInProcess"].append(measurement["StepInProcess"])
            data["StepDuration_s"].append(step_duration_ms[step_key] / 1000.0)
            data["StepTime_s"].append(float(measurement.get("StepTime_ms", 0)) / 1000.0)
            data["TestTime_s"].append(float(measurement.get("TestTime_ms", 0)) / 1000.0)
            data["SysTime"].append(measurement.get("SysTime"))
            data["Voltage_V"].append(measurement.get("Voltage_V"))
            data["Current_uA"].append(measurement.get("Current_uA"))
            data["Capacity_uAh"].append(capacity_uah)
            data["Energy_uWh"].append(energy_uwh)
            data["Power_uW"].append(measurement.get("Power_uW"))
            data["dQdV_uAh_V"].append(measurement.get("dQdV_uAh_per_V", 0.0))
            data["dVdQ_V_uAh"].append(measurement.get("dVdQ_V_per_uAh", 0.0))
            data["Temperature_℃"].append(measurement.get("Temperature_C", "0"))
            data["Humidity_%"].append(measurement.get("Humidity_pct", "0"))
            data["Mark1"].append(measurement.get("Mark1"))
            data["Mark2"].append(str(measurement.get("Mark2", "")))
            data["DataFile"].append(parser.filepath.replace("\\", "/"))
            data["TestName"].append(parser.metadata.get("test_name", ""))
            data["ProcessName"].append(parser.metadata.get("process", ""))
            data["ChannelNumber"].append(parser.metadata.get("channel", ""))

            if mass_g is not None and mass_g > 0:
                data["SpeCap_mAh_g"].append((capacity_uah / 1000.0) / mass_g)
                data["SpeEnergy_mWh_g"].append((energy_uwh / 1000.0) / mass_g)

        ds = xr.Dataset({key: (("record",), values) for key, values in data.items()})
        ds = sanitize_variable_names(ds)
        if not isinstance(ds, xr.Dataset):
            raise TypeError("Expected sanitized LANHE CCS data to remain an xarray.Dataset.")

        systimes = pd.to_datetime(ds["SysTime"].values)
        ds = ds.assign_coords(systime=(("record",), systimes))
        ds = ds.assign_coords(time_s=(("record",), pd.to_numeric(ds["TestTime_s"].values).astype(float)))
        ds = ds.drop_vars("SysTime")
        ds = ds.set_index(record="Record")

        ds.systime.attrs.update({"long_name": "System Time"})
        ds.time_s.attrs.update({"units": "s", "long_name": "Relative Test Time"})
        ds.attrs["source_format"] = "ccs"
        ds.attrs["reader"] = self.__class__.__name__
        return ds

    def _create_info(self, path: Path, parser: CCSParser, mass_g: float | None = None) -> RawDataInfo:
        """Create RawDataInfo for parsed CCS data."""
        metadata = dict(parser.metadata)
        metadata["file_path"] = str(path)
        metadata["source_format"] = "ccs"
        metadata["Cycle_Summary"] = self._cycle_summary(parser)
        metadata["Step_Summary"] = self._step_summary(parser)
        metadata["Log_Info"] = self._log_info(parser)
        if mass_g is not None:
            metadata["active_material_mass_g"] = mass_g

        start_time = self.start_time or self._format_datetime(metadata.get("start_time"))
        active_material_mass = self.active_material_mass or metadata.get("active_material_mass")

        return RawDataInfo(
            sample_name=self.sample_name or metadata.get("test_name", path.stem),
            start_time=start_time,
            operator=self.operator or metadata.get("user"),
            technique=list(self.technique),
            instrument=self.instrument,
            active_material_mass=active_material_mass,
            wave_number=self.wave_number,
            others=metadata,
        )

    @staticmethod
    def _step_duration_by_key(measurements: list[dict[str, Any]]) -> dict[tuple[Any, Any], int]:
        """Return final step duration in milliseconds for each step."""
        durations: dict[tuple[Any, Any], int] = {}
        for measurement in measurements:
            key = (measurement["Step"], measurement["StepInProcess"])
            durations[key] = max(durations.get(key, 0), int(measurement.get("StepTime_ms", 0)))
        return durations

    @staticmethod
    def _cycle_summary(parser: CCSParser) -> list[dict[str, Any]]:
        """Return cycle summary metadata in official-export-like keys."""
        cycle = parser.cycle
        if not cycle:
            return []
        return [
            {
                "Cycle": cycle.get("Cycle", 1),
                "CapC/uAh": cycle.get("CapC_uAh", 0.0),
                "CapD/uAh": cycle.get("CapD_uAh", 0.0),
                "CoulombEfficiency/%": cycle.get("CoulombEfficiency_pct", 0.0),
                "EnergyC/uWh": cycle.get("EnergyC_uWh", 0.0),
                "EnergyD/uWh": cycle.get("EnergyD_uWh", 0.0),
                "EnergyEfficiency/%": cycle.get("EnergyEfficiency_pct", 0.0),
                "DurationC": cycle.get("DurationC"),
                "DurationD": cycle.get("DurationD"),
                "DataFile": cycle.get("DataFile"),
                "ChannelNumber": cycle.get("ChannelNumber"),
                "AvgVoltC/V": cycle.get("AvgVoltC_V", 0.0),
                "AvgVoltD/V": cycle.get("AvgVoltD_V", 0.0),
            }
        ]

    @staticmethod
    def _step_summary(parser: CCSParser) -> list[dict[str, Any]]:
        """Return parser step groups as metadata."""
        return [dict(step) for step in parser.steps]

    @staticmethod
    def _log_info(parser: CCSParser) -> list[dict[str, Any]]:
        """Return parser log events as metadata."""
        return [dict(event) for event in parser.log_events]

    @classmethod
    def _parse_mass_g(cls, mass_input: Any) -> float | None:
        """Parse active material mass into grams."""
        if not mass_input:
            return None
        match = cls.MASS_REGEX.search(str(mass_input))
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2).lower().replace("µ", "u")
        if unit == "g":
            return value
        if unit == "mg":
            return value * 1e-3
        if unit == "ug":
            return value * 1e-6
        return None

    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        """Format datetime values for RawDataInfo."""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _get_file_extension() -> str:
        """Get the file extension for this reader."""
        return ".ccs"
