# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a form for configuring JWT settings."""
import os

from nicegui import ui
from nicegui.events import UploadEventArguments
from pydantic import SecretStr
from pydantic import ValidationError
from src.utils.conf import Setup
from src.utils.instance import Instance
from src.utils.instance import InstanceSettings
from src.utils.jwt_model import JWTModel
from src.utils.status import Status


class JWTSettingsForm:
    """
     JWTSettingsForm is a reusable UI component for configuring JSON Web Token (JWT) settings within an administrative interface.
     This form provides fields and controls for setting JWT parameters such as Audience, Issuer, Certificate Type, Certificate Name, Azure KeyVault Name, and Secret Key. It dynamically updates its UI based on the selected certificate type, supporting three modes: SecretKey, PemKeys (with file upload for public/private PEM keys), and AzureKeyVault.

     Key Features:
     - Renders a card-based form UI for JWT configuration.
     - Dynamically shows/hides fields and upload controls based on certificate type.
     - Handles secure upload and validation of PEM key files.
     - Validates input and saves JWT settings to the associated instance.
     - Provides user feedback and error notifications for validation and file upload issues.
     Attributes:
    -----------
         model (JWTModel): The JWT model instance representing current settings.
         instance (Instance): The parent instance to which settings are applied.
         widgets (dict): Dictionary of UI widget references for form fields.
         upload_container: UI container for PEM key upload controls.
         pem_upload_status (dict): Tracks upload status of private and public PEM keys.
    """

    model: JWTModel = JWTModel()

    def __init__(self, instance: Instance):
        self.instance = instance
        if not instance.settings:
            instance.settings = InstanceSettings()

        self.model = instance.settings.jwt or JWTModel()
        self.widgets = {}
        self.upload_container = None
        self.pem_upload_status = {"private_uploaded": False, "public_uploaded": False}

    def render(self):
        """
        Renders the JWT settings form UI.

        This method creates a card layout containing input fields for JWT configuration,
        including Audience, Issuer, Certificate Type, Certificate Name, Azure KeyVault Name,
        and Secret Key. The Certificate Type selection dynamically updates the visibility
        of related fields. A Save Settings button is provided to submit the form.
        UI elements are stored in the `self.widgets` dictionary for later access.
        """
        with ui.card().classes("w-full p-6 space-y-1 bg-white shadow-md rounded-lg"):

            valid_types = ["SecretKey", "PemKeys", "AzureKeyVault"]
            cert_type = self.model.CertificateType or "SecretKey"
            if cert_type not in valid_types:
                cert_type = "SecretKey"

            self.widgets = {
                "Audience": ui.input("Audience", value=self.model.Audience).classes(
                    "w-full"
                ),
                "Issuer": ui.input("Issuer", value=self.model.Issuer).classes("w-full"),
                "CertificateType": ui.select(
                    valid_types, value=cert_type, label="Certificate Type"
                ).classes("w-full"),
                "CertificateName": ui.input(
                    "Certificate Name", value=self.model.CertificateName
                ).classes("w-full"),
                "AzureKeyVaultName": ui.input(
                    "Azure KeyVault Name", value=self.model.AzureKeyVaultName
                ).classes("w-full"),
                "SecretKey": ui.input(
                    "Secret Key",
                    value=(
                        self.model.SecretKey.get_secret_value()
                        if self.model.SecretKey
                        else ""
                    ),
                    password=True,
                    password_toggle_button=True,
                ).classes("w-full"),
            }

            self.upload_container = ui.column().classes("w-full")

            self.widgets["CertificateType"].on(
                "update:model-value", lambda _: self._update_jwt_visibility()
            )
            self._update_jwt_visibility()

            with ui.row().classes("w-full justify-end mt-4"):
                ui.button(
                    text="Save Settings",
                    icon="save",
                    on_click=self.on_submit,
                    color="primary",
                )

    def _file_upload(self, upload: UploadEventArguments, key_type: str):
        """
        Handles the upload of a file for a specified key type.
        Args:
            upload (UploadEventArguments): The event arguments containing the uploaded file data.
            key_type (str): The type of key being uploaded (e.g., "public", "private").
        Returns:
            bool: True if the file upload was successful, False otherwise.
        Raises:
            Exception: If an error occurs during file upload, it is caught and handled internally.
        """
        try:
            return self._handle_file_upload(upload, key_type)
        except Exception as e:
            Status.FAILED(f"Error uploading {key_type} key", error=e, traceback=False)
            ui.notify(f"Failed to upload {key_type} key: {e}", type="negative")
            return False

    def _handle_file_upload(self, upload: UploadEventArguments, key_type: str):
        if not upload or not upload.name:
            ui.notify(f"No file uploaded for {key_type} key.", type="warning")
            return False

        filename = upload.name
        if not filename.lower().endswith(".pem"):
            ui.notify(
                f"Invalid file type: {filename}. Please upload a .pem file.",
                type="negative",
            )
            return False

        file_path = os.path.join(
            Setup().db_path,
            self.instance.instance_id,
            Setup().global_constants.get("JWT", {}).get(f"{key_type}_KEY_FILE"),
        )
        Status.INFO(f"Uploading {key_type} Key to {file_path}")
        with open(file_path, "wb") as file:
            if hasattr(upload.content, "read"):
                file.write(upload.content.read())
            else:
                file.write(upload.content)
        ui.notify(
            f"{key_type.capitalize()} key '{filename}' uploaded and saved.",
            type="positive",
        )
        return True

    def _update_jwt_visibility(self):
        """Updates the visibility of JWT settings based on the selected certificate type."""
        cert_type = self.widgets["CertificateType"].value
        # Hide all optional widgets
        self.widgets["CertificateName"].set_visibility(False)
        self.widgets["AzureKeyVaultName"].set_visibility(False)
        self.widgets["SecretKey"].set_visibility(False)
        self.upload_container.clear()
        self.pem_upload_status["private_uploaded"] = False
        self.pem_upload_status["public_uploaded"] = False

        if cert_type == "AzureKeyVault":
            self.widgets["AzureKeyVaultName"].set_visibility(True)
            self.widgets["CertificateName"].set_visibility(True)
        elif cert_type == "PemKeys":
            with self.upload_container:
                ui.upload(
                    label="Upload Private Key (.pem)",
                    auto_upload=True,
                    on_upload=lambda e: self.pem_upload_status.__setitem__(
                        "private_uploaded", self._file_upload(e, "PRIVATE")
                    ),
                ).classes("w-full").props("accept=.pem")

                ui.upload(
                    label="Upload Public Key (.pem)",
                    auto_upload=True,
                    on_upload=lambda e: self.pem_upload_status.__setitem__(
                        "public_uploaded", self._file_upload(e, "PUBLIC")
                    ),
                ).classes("w-full").props("accept=.pem")
        else:  # SecretKey
            self.widgets["SecretKey"].set_visibility(True)

    def on_submit(self):
        """
        Handles the submission of the JWT configuration form.
        This method collects input values from the form widgets, validates the presence and type of the secret key,
        checks for required PEM key uploads if the certificate type is set to 'PemKeys', and attempts to save the JWT
        settings to the instance. It provides user notifications for success, validation errors, missing PEM keys,
        and other exceptions encountered during the process.
        Raises:
            ValidationError: If the JWT input data fails model validation.
            Exception: For any other errors encountered during saving.
        """
        try:
            jwt_input = {k: v.value for k, v in self.widgets.items()}

            if jwt_input.get("SecretKey"):
                jwt_input["SecretKey"] = SecretStr(jwt_input["SecretKey"])
            else:
                jwt_input["SecretKey"] = None

            # PEM upload validation
            if jwt_input.get("CertificateType") == "PemKeys" and not (
                self.pem_upload_status["private_uploaded"]
                and self.pem_upload_status["public_uploaded"]
            ):
                ui.notify(
                    "Please upload both private and public PEM keys.",
                    type="warning",
                    position="top",
                )
                return

            # Validate and save
            jwt_model = JWTModel(**jwt_input)
            self.instance.settings.jwt = jwt_model
            if self.instance.save_settings():
                ui.notify(
                    "JWT settings saved successfully.",
                    type="positive",
                    position="bottom",
                )
            else:
                Status.FAILED("Failed to save JWT settings.")
                ui.notify(
                    "Failed to save JWT settings. Contact support.",
                    type="negative",
                )
        except ValidationError as e:
            ui.notify(e.errors(), type="negative", position="top")
        except Exception as ex:
            Status.FAILED("Error saving JWT settings", error=ex, traceback=False)
            ui.notify("Failed to save JWT settings.", type="negative", position="top")
