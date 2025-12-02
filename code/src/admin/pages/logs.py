# # Copyright (C) KonaAI - All Rights Reserved
"""All the logs related to the application are stored here"""
import asyncio
import glob
import os
import zipfile
from datetime import date
from datetime import datetime

from nicegui import run
from nicegui import ui
from src.admin import theme
from src.admin.components.spinners import create_loader
from src.utils.conf import Setup
from src.utils.file_mgmt import file_handler
from src.utils.status import Status


def make_archive(log_files):
    """
    Creates a zip archive containing the specified log files.

    Args:
        log_files (list): A list of file paths to be included in the archive.

    Returns:
        str: The name of the created zip file.
    """
    _, zip_name = file_handler.get_new_file_name(file_extension=".zip")
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for file in log_files:
            zipf.write(file, arcname=os.path.basename(file))
    return zip_name


def logs():
    """
    Page to display logs.
    """
    with theme.frame("Logs"):
        ui.markdown("# Logs")
        with ui.column().classes("w-full"):
            log_parser()


def log_parser():
    """
    Parses and displays log files from the configured logs directory.
    This function retrieves all JSON log files, determines the available date range,
    and provides a user interface for selecting a date range, viewing logs, and downloading logs.
    It handles UI interactions, including loading spinners and notifications, and displays results
    or error messages as appropriate.

    Raises:
        Displays a notification if no logs are found.
        Handles and reports exceptions during log parsing or UI setup.
    """
    try:
        logs_dir = Setup().log_path
        log_files = glob.glob(os.path.join(logs_dir, "*.json"))
        if not log_files:
            ui.notify("No logs found", type="negative")
            return

        min_date, max_date = get_log_date_range(log_files)

        with ui.row().classes("items-center"):
            _, date_picker = create_date_range_input(min_date, max_date)
            view_btn = ui.button("View", icon="visibility").props("outlined")
            download_btn = ui.button("Download", icon="download").props("outlined")
            download_spinner = create_loader()
            download_spinner.visible = False

            result_label = ui.label("").style("color: green")
            result_label.visible = False
            download_btn.on(
                "click",
                lambda: download_logs(
                    date_picker.value, result_label, download_spinner
                ),
            )

        with ui.row().classes("items-center"):
            view_spinner = create_loader()
            view_spinner.visible = False

        log_display_area = ui.column().classes("w-full")
        view_btn.on_click(
            lambda: view_logs(date_picker.value, log_display_area, view_spinner)
        )

    except BaseException as _e:
        Status.FAILED("Failed to parse logs", str(_e))

    log_files = []


async def view_logs(
    date_range_value: dict, display_area: ui.column, spinner: ui.element
):
    """
    Displays logs within the specified date range (async, with spinner).
    """
    if not date_range_value:
        ui.notify("Select a date range", type="warning")
        return
    spinner.visible = True
    try:
        await asyncio.sleep(0.5)  # Increased delay for more visible spinner

        start_date, end_date = parse_date_range(date_range_value)

        if not isinstance(start_date, date) or not isinstance(end_date, date):
            ui.notify("Provide valid dates", type="warning")
            return

        if start_date > end_date:
            ui.notify("Start date cannot be greater than end date", type="negative")
            return

        # IO-bound: filter logs by date range
        def get_log_files():
            log_files_list = glob.glob(os.path.join(Setup().log_path, "*.json"))
            log_files_list = [
                i
                for i in log_files_list
                if start_date
                <= datetime.fromtimestamp(os.path.getctime(i)).date()
                <= end_date
            ]
            log_files_list.sort(key=os.path.getctime, reverse=True)
            return log_files_list

        log_files_list = await run.io_bound(get_log_files)

        display_area.clear()
        with display_area:
            if not log_files_list:
                ui.label("No logs found for the selected date range!").classes(
                    "text-red-500"
                )
                return
            for log_file in log_files_list:
                long_dt_format = datetime.fromtimestamp(
                    os.path.getctime(log_file)
                ).strftime("%A, %d %B %Y")
                with (
                    ui.expansion(f"Log for {long_dt_format}", value=False)
                    .classes("w-full border-2 rounded-md")
                    .props("outlined") as log_ex
                ):
                    with ui.row().classes("w-full h-80 relative"):
                        log_spinner = create_loader()
                        log_spinner.visible = False

                        log_viewer = ui.column().classes(
                            "w-full h-80 overflow-auto gap-1"
                        )

                    def on_expand(value, lg=log_viewer, lf=log_file, sp=log_spinner):
                        if value:
                            sp.visible = True
                            lg.clear()
                            asyncio.create_task(show_log(lf, lg, sp))
                        else:
                            sp.visible = False
                            lg.clear()

                    log_ex.on_value_change(on_expand)
    except BaseException as _e:
        Status.FAILED("Failed to view logs", str(_e))
        ui.notify("No logs", type="negative")
    finally:
        spinner.visible = False


async def show_log(log_path: str, log_viewer: ui.column, spinner: ui.element):
    """
    Displays the content of the specified log file in the log viewer UI component (async).
    """
    try:
        with log_viewer:
            spinner.visible = True
        if not os.path.exists(log_path):
            ui.notify("Log not found", type="negative")
            return

        def read_lines():
            with open(log_path, encoding="utf-8") as f:
                return f.readlines()

        lines = await run.io_bound(read_lines)
        with log_viewer:
            for i, line in enumerate(lines):
                bg_color = "bg-gray-100" if i % 2 == 0 else "white"
                ui.label(line).classes(f"{bg_color}")
                ui.separator()
    except BaseException as _e:
        Status.FAILED("Failed to show log", str(_e))
        ui.notify("Failed to show log", type="negative")
    finally:
        spinner.visible = False


async def download_logs(date_range_value: dict, result_label, spinner: ui.element):
    """
    Downloads logs within the specified date range (async, with spinner).
    """
    result_label.visible = False
    spinner.visible = True
    try:
        if not date_range_value:
            ui.notify("Select a date range", type="warning")
            return

        start_date, end_date = parse_date_range(date_range_value)

        if not isinstance(start_date, date) or not isinstance(end_date, date):
            ui.notify("Provide valid dates", type="warning")
            return

        if start_date > end_date:
            ui.notify("Start date cannot be greater than end date", type="negative")
            return

        def get_log_files():
            return filter_logs_by_date_range(start_date, end_date)

        log_files = await run.io_bound(get_log_files)

        if not log_files:
            ui.notify("No logs found for the selected date range.", type="warning")
            return

        def zip_logs():
            return make_archive(log_files)

        await asyncio.sleep(0.3)

        zip_file = await run.io_bound(zip_logs)
        file_name = (
            f"logs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.zip"
        )
        with open(zip_file, "rb") as f:
            ui.download(f.read(), filename=file_name)

        result_label.text = "Logs downloaded. Check downloads directory."
        result_label.visible = True
    except BaseException as _e:
        Status.FAILED("Failed to download logs", str(_e))
        ui.notify("Failed to download logs", type="negative")
    finally:
        spinner.visible = False


def filter_logs_by_date_range(start_date, end_date):
    """
    Filter log files by date range.

    Args:
        start_date (datetime.date): the start date
        end_date (datetime.date): the end date

    Returns:
        list: a list of file paths to the filtered log files
    """
    log_files = glob.glob(os.path.join(Setup().log_path, "*.json"))
    log_files = [
        i
        for i in log_files
        if start_date <= datetime.fromtimestamp(os.path.getctime(i)).date() <= end_date
    ]
    return log_files


def get_log_date_range(log_files):
    """
    Returns the minimum and maximum dates from the log files.
    """
    min_date_timestamp = min(os.path.getctime(i) for i in log_files)
    max_date_timestamp = max(os.path.getctime(i) for i in log_files)
    return (
        datetime.fromtimestamp(min_date_timestamp).date(),
        datetime.fromtimestamp(max_date_timestamp).date(),
    )


def create_date_range_input(min_date, max_date):
    """
    Creates a date range input UI component with a calendar picker.

    Args:
        min_date (datetime.date): The minimum selectable date.
        max_date (datetime.date): The maximum selectable date.

    Returns:
        tuple: A tuple containing the date input component and the date picker component.
    """

    with ui.input("Date range").classes("w-64 mr-2") as date_input:
        with ui.menu().props("no-parent-event") as menu:
            date_picker = ui.date().props("range")
            date_picker.bind_value(
                date_input,
                forward=lambda x: (
                    f'{x["from"]} - {x["to"]}' if isinstance(x, dict) else None
                ),
                backward=lambda x: (
                    {"from": x.split(" - ")[0], "to": x.split(" - ")[1]}
                    if " - " in (x or "")
                    else None
                ),
            )
            date_picker.props(
                f'min={min_date.strftime("%Y-%m-%d")} max={max_date.strftime("%Y-%m-%d")}'
            )
            with ui.row().classes("justify-end"):
                ui.button("Close", on_click=menu.close).props("flat")
        with date_input.add_slot("append"):
            ui.icon("edit_calendar").on("click", menu.open).style(
                "align-self: center;cursor:pointer"
            )
    return date_input, date_picker


def parse_date_range(date_range_value: dict):
    """
    Parses a dictionary containing 'from' and 'to' date strings into Python date objects.

    Args:
        date_range_value (dict): A dictionary with keys 'from' and 'to', each mapping to a date string in the format '%Y-%m-%d'.

    Returns:
        tuple: A tuple containing two datetime.date objects: (start_date, end_date).
    """
    start_date = datetime.strptime(date_range_value["from"], "%Y-%m-%d").date()
    end_date = datetime.strptime(date_range_value["to"], "%Y-%m-%d").date()
    return start_date, end_date
