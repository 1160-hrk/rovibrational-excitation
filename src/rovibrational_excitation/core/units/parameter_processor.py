"""
Parameter processing utilities for rovibrational excitation calculations.

This module provides high-level parameter processing that combines
unit conversion with automatic parameter detection and processing.
"""

from typing import Any

from .converters import converter


class ParameterProcessor:
    """
    高レベルパラメータ処理クラス

    parameter_converter.pyの機能をunitsモジュールに統合し、
    より拡張性があり保守しやすい設計で提供します。
    """

    def __init__(self):
        """Initialize parameter processor."""
        self.converter = converter

        # Neutral frequency quantities are converted by typed model/field schemas.
        self.frequency_params: list[str] = []

        self.dipole_params = ["mu0_Cm", "transition_dipole_moment"]
        self.field_params = ["amplitude"]
        # The typed TwoLevel schema owns energy-gap conversion and its unit label.
        self.energy_params: list[str] = []
        self.time_params = [
            "duration",
            "t_center",
            "t_start",
            "t_end",
            "dt",
            "coherence_relaxation_time_ps",
        ]

    def auto_convert_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically convert parameters with unit specifications to standard units.

        This method replaces parameter_converter.py's auto_convert_parameters function
        with enhanced functionality.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameter dictionary potentially containing unit specifications

        Returns
        -------
        Dict[str, Any]
            Parameter dictionary with values converted to standard units
        """
        converted_params = params.copy()

        # Process frequency parameters
        for param in self.frequency_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_frequency(
                        original_value, unit, "rad/fs"
                    )
                    converted_params[param] = converted_value
                    print(
                        f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} rad/fs"
                    )

                except ValueError as e:
                    raise ValueError(f"Failed to convert {param}: {e}") from e

        # Process dipole moment parameters
        for param in self.dipole_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_dipole_moment(
                        original_value, unit, "C*m"
                    )
                    converted_params[param] = converted_value
                    print(
                        f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} C·m"
                    )

                except ValueError as e:
                    raise ValueError(f"Failed to convert {param}: {e}") from e

        # Process electric field parameters
        for param in self.field_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_electric_field(
                        original_value, unit, "V/m"
                    )
                    converted_params[param] = converted_value
                    print(
                        f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} V/m"
                    )

                except ValueError as e:
                    raise ValueError(f"Failed to convert {param}: {e}") from e

        # Process energy parameters
        for param in self.energy_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    converted_value = self.converter.convert_energy(
                        original_value, unit, "J"
                    )
                    converted_params[param] = converted_value
                    print(
                        f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} J"
                    )

                except ValueError as e:
                    raise ValueError(f"Failed to convert {param}: {e}") from e

        # Process time parameters
        for param in self.time_params:
            unit_key = f"{param}_units"
            if param in converted_params and unit_key in converted_params:
                original_value = converted_params[param]
                unit = converted_params[unit_key]
                try:
                    if "ps" in param:
                        # Special handling for ps parameters
                        if unit != "ps":
                            converted_value = self.converter.convert_time(
                                original_value, unit, "ps"
                            )
                            converted_params[param] = converted_value
                            print(
                                f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} ps"
                            )
                    else:
                        converted_value = self.converter.convert_time(
                            original_value, unit, "fs"
                        )
                        converted_params[param] = converted_value
                        print(
                            f"✓ Converted {param}: {original_value} {unit} → {converted_value:.6g} fs"
                        )

                except ValueError as e:
                    raise ValueError(f"Failed to convert {param}: {e}") from e

        return converted_params

    def add_parameter_group(
        self, group_name: str, param_list: list, quantity_type: str
    ):
        """
        Add a custom parameter group for automatic processing.

        Parameters
        ----------
        group_name : str
            Name of the parameter group
        param_list : list
            List of parameter names
        quantity_type : str
            Type of physical quantity ("frequency", "energy", etc.)
        """
        if quantity_type not in ["frequency", "dipole", "field", "energy", "time"]:
            raise ValueError(f"Unknown quantity type: {quantity_type}")

        attr_name = f"{quantity_type}_params"
        if hasattr(self, attr_name):
            current_list = getattr(self, attr_name)
            current_list.extend(param_list)
        else:
            setattr(self, attr_name, param_list)

        print(
            f"✓ Added parameter group '{group_name}' with {len(param_list)} parameters"
        )


# Create singleton instance
parameter_processor = ParameterProcessor()
