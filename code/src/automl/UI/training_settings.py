# # Copyright (C) KonaAI - All Rights Reserved
"""Run settings configuration for AutoML with dynamic form generation based on ML parameters schema."""
from typing import Any

import annotated_types
from nicegui import ui
from pydantic.fields import FieldInfo
from src.utils.submodule import Submodule


class TrainingSettingsUI:
    """
    UI class for configuring and displaying machine learning training settings.
    Attributes:
    ----------
        chosen_param (str): The currently selected training parameter.
        submodule (Submodule): The submodule containing ML parameters.
    """

    chosen_param: str = None

    def __init__(self, submodule: Submodule) -> None:
        self.submodule = submodule

    def render(self) -> None:
        """
        Renders the training settings section in the UI.

        This method displays an expandable panel labeled "Training Settings" with a settings icon.
        The panel is styled with full width, a border, and rounded corners. The actual settings
        are drawn by calling the `_draw_settings` method.
        """
        with (
            ui.expansion(text="Training Settings", icon="settings")
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            self._draw_settings()

    @ui.refreshable_method
    def _draw_settings(self):
        """Draw the settings widgets."""
        params = self.submodule.ml_params

        param_options = {}
        for field_name, field_info in params.model_fields.items():
            fi: FieldInfo = field_info
            param_options[field_name] = fi.title or field_name

        select_param = (
            ui.select(
                label="Select Training Parameter",
                value=self.chosen_param,
                options=param_options,
            )
            .classes("w-full")
            .props("outlined")
            .bind_value(self, "chosen_param")
        )
        select_param.on_value_change(
            lambda: self._draw_settings.refresh()  # pylint: disable=unnecessary-lambda
        )

        # show the selected parameter description
        if self.chosen_param:
            field_info = params.model_fields[self.chosen_param]
            ui.label(f"{field_info.description}").classes("w-full")
            self._draw_value_widget(self.chosen_param, field_info)

    def _draw_value_widget(self, field_name: str, field_info: FieldInfo):
        """Draw the value widget based on field type."""
        value_widget = None
        if field_info.annotation is bool:
            value_widget = ui.switch(
                value=getattr(self.submodule.ml_params, field_name),
                on_change=lambda e: setattr(
                    self.submodule.ml_params, field_name, e.value
                ),
            ).classes("w-full")
        elif field_info.annotation in (int, float):
            _min = (
                next(
                    v.ge
                    for v in field_info.metadata
                    if isinstance(v, annotated_types.Ge)
                )
                if field_info.metadata
                else 0
            )
            _max = (
                next(
                    v.le
                    for v in field_info.metadata
                    if isinstance(v, annotated_types.Le)
                )
                if field_info.metadata
                else 100
            )
            value = (
                getattr(self.submodule.ml_params, field_name) or field_info.default
                if field_info.default is not None
                else 0
            )

            value_widget = (
                ui.slider(
                    min=_min if _min is not None else 0,
                    max=_max if _max is not None else 100,
                    value=value,
                )
                .classes("w-full mt-4")
                .props("label-always switch-marker-labels-side")
            )

        if value_widget:
            ui.button(
                text="Save",
                icon="save",
                on_click=lambda: self._save_param(field_name, value_widget.value),
            ).classes()

    def _save_param(self, field_name: str, value: Any):
        """Save the parameter value."""
        setattr(self.submodule.ml_params, field_name, value)
        self.submodule.save_configuration()
        ui.notify(f"Parameter updated to {value}", type="positive")
