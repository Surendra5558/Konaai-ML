# # Copyright (C) KonaAI - All Rights Reserved
"""Submodule Selector UI Component"""
import contextlib
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import List

from nicegui import run
from nicegui import ui
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.metadata import Metadata
from src.utils.submodule import Submodule


class SubModuleSelectorUI:
    """Class to handle the submodule selection UI."""

    instance: Instance = Instance()
    submodule: Submodule = Submodule(
        module=None, submodule=None, instance_id=GlobalSettings().active_instance_id
    )
    next_function: ClassVar[Callable] = None
    next_function_kwargs: ClassVar[Dict[str, Any]] = {}
    spinner: ClassVar[ui.element] = None
    drawn_placeholder: ClassVar[bool] = False
    submodule_container: ClassVar[ui.element] = None  # Container for submodule dropdown

    def __init__(self):
        """Initialize the SubModuleSelectorUI."""
        self.md = None

    def reset(self):
        """Reset the submodule selector state."""
        self.submodule = Submodule(
            module=None, submodule=None, instance_id=GlobalSettings().active_instance_id
        )
        self.next_function = None
        self.next_function_kwargs = {}
        self.drawn_placeholder = False
        self.submodule_container = None

    async def render(self, next_func: Callable, **kwargs):
        """
        Renders the submodule selector UI and sets up the next function to be called.

        Args:
            next_func (Callable): The function to be called after the selection is made.
            **kwargs: Additional keyword arguments to be passed to the next function.

        Side Effects:
            Stores the next function and its arguments for later use.
            Calls the internal method to draw available modules.
        """
        if not GlobalSettings().active_instance_id:
            ui.label("No active instance found.").classes("text-red-500")
            return None

        self.instance = GlobalSettings.instance_by_id(
            GlobalSettings().active_instance_id
        )
        if not self.instance:
            raise ValueError("No active instance found.")
        self.md = Metadata(GlobalSettings().active_instance_id)
        self.next_function = next_func
        self.next_function_kwargs = kwargs
        await self._draw_modules()
        return None

    async def _draw_modules(self):
        self._start_spinner()
        modules: List[str] = await run.io_bound(lambda: self.md.modules)
        self._stop_spinner()
        if not modules:
            ui.label("No modules found.").classes("text-red-500")
            return

        # Create a select dropdown for modules
        ui.select(
            label="Select Module",
            options=modules,
            on_change=lambda e: self._draw_submodules(e.value),
        ).classes("w-full mb-1").props("outlined")

        # Create a dedicated container for submodule dropdown right after module dropdown
        # This ensures it appears immediately below the module dropdown
        self.submodule_container = ui.column().classes("w-full mb-1")

        # Call _draw_placeholder initially to show buttons (even if disabled)
        # This ensures buttons are always visible from the start
        self._draw_placeholder()

    async def _draw_submodules(self, module_name: str):
        # Reset submodule selection when module changes
        self.submodule.module = module_name
        self.submodule.submodule = None

        # Clear the submodule container to remove any existing dropdown
        if self.submodule_container is not None:
            with contextlib.suppress(Exception):
                self.submodule_container.clear()
        # Refresh the data dictionary widget to update button states (disable them)
        # Buttons will remain visible but disabled until submodule is selected
        if self.drawn_placeholder:
            with contextlib.suppress(Exception):
                self._draw_placeholder.refresh()
        # Get submodules for the selected module
        submodules: List[str] = await run.io_bound(
            lambda: self.md.get_submodule_names(module_name)
        )

        # Create submodule dropdown in the dedicated container
        # This ensures it appears immediately below the module dropdown
        with self.submodule_container:
            if not submodules:
                ui.label("No submodules found.").classes("text-red-500")
                return

            sub = (
                ui.select(label="Select Submodule", options=submodules)
                .classes("w-full")
                .props("outlined")
            )
            sub.on_value_change(
                lambda e: self._create_submodule_object(module_name, e.value)
            )

    def _create_submodule_object(self, module_name: str, submodule_name: str):
        """Create a Submodule object based on the selected module and submodule."""
        if not module_name or not submodule_name:
            ui.notify("Please select both a module and a submodule.", color="red")
            return

        # Create the Submodule object
        self.submodule.module = module_name
        self.submodule.submodule = submodule_name
        if self.drawn_placeholder:
            # If the placeholder was already drawn, remove it
            self._draw_placeholder.refresh()
        else:
            self._draw_placeholder()

    @ui.refreshable_method
    def _draw_placeholder(self):
        self.drawn_placeholder = True

        if self.next_function:
            # Update kwargs with current submodule to ensure we pass the latest values
            updated_kwargs = (
                self.next_function_kwargs.copy()
                if isinstance(self.next_function_kwargs, dict)
                else {}
            )
            if "submodule" in updated_kwargs:
                updated_kwargs["submodule"] = self.submodule
            # call the next function if provided
            self.next_function(**updated_kwargs)

    def _start_spinner(self) -> ui.element:
        # show an overlay with a spinner while the upload is in progress
        with ui.element("div").classes(
            "fixed inset-0 flex items-center justify-center bg-black/20 z-50"
        ) as overlay:
            ui.spinner(size="xl", color="primary")
        overlay.visible = True  # Initially hide the overlay
        self.spinner = overlay
        return overlay

    def _stop_spinner(self):
        if self.spinner:
            self.spinner.visible = False
            self.spinner.delete()
