# # Copyright (C) KonaAI - All Rights Reserved
"""LLM Configuration UI Component Module"""
from nicegui import ui
from src.insight_agent import constants
from src.utils.llm_config import BaseLLMConfig


class LLMConfigUI:
    """LLM Configuration UI Component"""

    llm_config: BaseLLMConfig = None

    def __init__(self, llm_config: BaseLLMConfig = None):
        self.llm_config = llm_config
        self.container = None
        self.widgets = {}

    def clear_widgets(self):
        """Clear existing widgets to prevent stacking issues."""
        if self.container:
            self.container.clear()
        self.widgets.clear()

    def render(self):  # sourcery skip: extract-method
        """Render the LLM configuration UI."""
        # Clear any existing widgets first
        self.clear_widgets()

        # Create main container with proper structure
        self.container = ui.column().classes("w-full max-w-2xl mx-auto")

        with self.container:
            # Configuration form card
            with ui.card().classes("w-full p-6"):
                # Use column layout for form elements
                with ui.column().classes("w-full gap-4"):
                    # LLM Provider Selection
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("LLM Provider:").classes("w-24 text-sm font-medium")
                        llm_providers = list(constants.LLM_MODELS.keys())
                        self.widgets["llm_name"] = (
                            ui.select(
                                options=llm_providers,
                                value=(
                                    getattr(self.llm_config, "llm_name", None)
                                    if self.llm_config
                                    else None
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "llm_name")
                            if self.llm_config
                            else ui.select(options=llm_providers).classes("flex-1")
                        )

                    # Model Name
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Model Name:").classes("w-24 text-sm font-medium")
                        self.widgets["model_name"] = (
                            ui.input(
                                placeholder="Enter model name",
                                value=(
                                    getattr(self.llm_config, "model_name", "")
                                    if self.llm_config
                                    else ""
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "model_name")
                            if self.llm_config
                            else ui.input(placeholder="Enter model name").classes(
                                "flex-1"
                            )
                        )

                    # Temperature
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Temperature:").classes("w-24 text-sm font-medium")
                        self.widgets["temperature"] = (
                            ui.number(
                                placeholder="0.7",
                                min=0.0,
                                max=2.0,
                                step=0.1,
                                value=(
                                    getattr(self.llm_config, "temperature", 0.7)
                                    if self.llm_config
                                    else 0.7
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "temperature")
                            if self.llm_config
                            else ui.number(
                                placeholder="0.7", min=0.0, max=2.0, step=0.1, value=0.7
                            ).classes("flex-1")
                        )

                    # API Key
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("API Key:").classes("w-24 text-sm font-medium")
                        self.widgets["api_key"] = (
                            ui.input(
                                placeholder="Enter API key",
                                password=True,
                                password_toggle_button=True,
                                value=(
                                    getattr(self.llm_config, "api_key", "")
                                    if self.llm_config
                                    else ""
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "api_key")
                            if self.llm_config
                            else ui.input(
                                placeholder="Enter API key", password=True
                            ).classes("flex-1")
                        )

                    # Endpoint
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("Endpoint:").classes("w-24 text-sm font-medium")
                        self.widgets["endpoint"] = (
                            ui.input(
                                placeholder="Enter endpoint URL",
                                value=(
                                    getattr(self.llm_config, "endpoint", "")
                                    if self.llm_config
                                    else ""
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "endpoint")
                            if self.llm_config
                            else ui.input(placeholder="Enter endpoint URL").classes(
                                "flex-1"
                            )
                        )

                    # API Version
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label("API Version:").classes("w-24 text-sm font-medium")
                        self.widgets["api_version"] = (
                            ui.input(
                                placeholder="Enter API version",
                                value=(
                                    getattr(self.llm_config, "api_version", "")
                                    if self.llm_config
                                    else ""
                                ),
                            )
                            .classes("flex-1")
                            .bind_value_to(self.llm_config, "api_version")
                            if self.llm_config
                            else ui.input(placeholder="Enter API version").classes(
                                "flex-1"
                            )
                        )

                    # Action buttons
                    with ui.row().classes("w-full justify-center gap-4 mt-6"):
                        ui.button("Reset", on_click=self._reset_form).classes(
                            "bg-gray-500 text-white px-6 py-2"
                        )

    def _reset_form(self):
        """Reset form to default values."""
        if self.llm_config:
            # Reset to default values
            for widget_name, widget in self.widgets.items():
                if hasattr(widget, "value"):
                    if widget_name == "temperature":
                        widget.value = 0.7
                    elif widget_name == "llm_name":
                        widget.value = None
                    else:
                        widget.value = ""
        ui.notify("Form reset to default values.", color="info")
