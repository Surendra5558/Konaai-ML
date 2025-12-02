# # Copyright (C) KonaAI - All Rights Reserved
"""This Module handles the login page for the application."""
import os

import bcrypt
import yaml
from nicegui import app
from nicegui import ui
from src.admin import theme
from src.admin.utils import config
from src.utils.conf import Setup


# Authenticate user
def authenticate(user_name, password):
    """
    Authenticates a user by validating the provided username and password against credentials stored in a YAML file.

    Args:
        user_name (str): The username to authenticate.
        password (str): The password to authenticate.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """

    # read credentials from yaml file
    yaml_path = os.path.join(Setup().assets_path, config.get("settings", "CREDS_FILE"))
    with open(yaml_path, encoding="utf-8") as file:
        yml = yaml.load(file, Loader=yaml.SafeLoader)

    users = yml["credentials"]["usernames"]
    secret_key = yml["secret"]["salt"].encode("ASCII")

    # Validate user name and password
    if user_name in users.keys():
        hashed_password = users[user_name]["password"].encode("ASCII")
        if hashed_password == bcrypt.hashpw(password.encode("ASCII"), secret_key):
            return True
    return False


# Login setup
def login():
    """Handle user login.

    This function sets up the login page with username and password input fields,
    and a login button. Upon clicking the login button, it authenticates the user
    and redirects to the home page if successful. Otherwise, it displays an
    error notification.
    """
    with theme.frame("Login"):
        ui.page_title("Login")
        ui.markdown("# KonaAI Intelligence Server")
        ui.separator()

        def on_login_click():
            username = username_input.value
            password = password_input.value
            if authenticate(username, password):

                # if login successful, open the home page
                ui.navigate.to("/", new_tab=False)

                ui.timer(
                    0.1,
                    lambda: (
                        ui.notify("Login successful!", type="positive"),
                        ui.navigate.to("/", new_tab=True),
                    ),
                    once=True,
                )

                # save logged in status
                app.storage.user["authenticated"] = True

            else:
                ui.notify("Incorrect username or password", type="negative")

        with ui.column().classes("absolute-center items-center"):
            ui.label("Login").style("font-size: 2em; font-weight: bold;")
            # username input
            username_input = (
                ui.input(label="Username", placeholder="Enter username")
                .classes("w-full border-2 rounded-md flex-grow")
                .props("outlined")
            )

            # password input
            password_input = (
                ui.input(
                    label="Password",
                    password=True,
                    placeholder="Enter password",
                    password_toggle_button=True,
                )
                .classes("w-full border-2 rounded-md flex-grow")
                .props("outlined")
            )
            ui.button("Login", on_click=on_login_click).classes("mt-4").style(
                "width: 200px;"
            )
