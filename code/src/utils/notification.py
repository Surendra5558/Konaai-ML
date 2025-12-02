# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the email notification class"""
import base64
import pathlib
import re
import smtplib
import time
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Union

import pandas as pd
import validators
from src.utils.conf import Setup
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.status import Status


class EmailNotification:
    """
    EmailNotification is a utility class for sending email notifications with support for SMTP configuration,
    content formatting, attachments, and retry logic.
    Attributes:
    ----------
        smtp_server (str): The SMTP server address.
        smtp_port (int): The SMTP server port (default: 25).
        username (str): The username for SMTP authentication.
        _password (str): The base64-encoded password for SMTP authentication (use the password property).
        recipient_emails (List[str]): List of recipient email addresses.
        from_email (str): The sender's email address.
        footer (str): Footer text appended to every email (default: automated email notice).
    """

    smtp_server: str = None
    smtp_port: int = 25
    username: str = None
    _password: str = None
    recipient_emails: List[str] = []
    from_email: str = None
    footer: str = "This is an automated email. Please do not reply to this email."

    def __str__(self):
        return f"EmailNotification(Server: {self.smtp_server}, Port: {self.smtp_port})"

    def __init__(self, instance_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_id = instance_id
        self.contents: List[Any] = []
        self.attachments: List[str] = []

        self.load_config()

    @property
    def password(self) -> str:
        """
        Decodes the stored base64 encoded password.

        Returns:
            str: The decoded password as a UTF-8 string.
        """
        # return url safe base64 encoded password with utf-8 encoding
        if not self._password:
            return ""
        return base64.urlsafe_b64decode(self._password).decode("utf-8")

    @password.setter
    def password(self, value: str):
        """
        Sets the password after encoding it in a URL-safe base64 format with UTF-8 encoding.

        Args:
            value (str): The password to be encoded and set.
        """
        if not value:
            self._password = ""  # nosec
        else:
            # set url safe base64 encoded password with utf-8 encoding
            self._password = base64.urlsafe_b64encode(value.encode("utf-8")).decode(
                "utf-8"
            )

    @property
    def header(self) -> Union[pathlib.Path, None]:
        """
        Retrieves the path to the logo file if it exists.

        Returns:
            Union[pathlib.Path, None]: The path to the logo file if it exists, otherwise None.
        """
        logo_file_name = Setup().global_constants.get("ASSETS", {}).get("LOGO_FILE", "")
        logo_file_path = pathlib.Path(Setup().assets_path, logo_file_name)
        return logo_file_path if logo_file_path.exists() else None

    def load_config(self) -> bool:
        """
        Loads the email configuration.

        This method attempts to load the email configuration by calling the
        internal `_load_config` method. If an exception occurs during the
        loading process, it logs the failure and returns `False`.

        Returns:
            bool: `True` if the configuration is loaded successfully,
                  `False` otherwise.
        """
        try:
            self._load_config()
        except BaseException as e:
            Status.FAILED(
                "Failed to load email configuration",
                self,
                error=str(e),
                traceback=False,
            )
            return False
        return True

    def _load_config(self):
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            raise ValueError(f"Instance with ID {self.instance_id} not found.")

        if not instance.settings.notification:
            raise ValueError(
                f"No notification settings found for instance {self.instance_id}"
            )

        self.smtp_server = instance.settings.notification.SMTPServer
        self.smtp_port = instance.settings.notification.SMTPPort
        self.username = instance.settings.notification.SMTPUsername
        self.password = instance.settings.notification.SMTPPassword.get_secret_value()
        self.from_email = instance.settings.notification.FromEmail

        _vars = [
            self.smtp_server,
            self.smtp_port,
            self.from_email,
        ]
        if [var for var in _vars if not var]:
            raise ValueError("Missing required SMTP configuration")

        try:
            instance = Instance(instance_id=self.instance_id)
            self.recipient_emails = list(instance.settings.notification.RecipientEmails)
        except Exception:
            self.recipient_emails = []
        # incase recipient emails is entered as text in the settings file instead of a list
        # convert it to a list
        if isinstance(self.recipient_emails, str):
            self.recipient_emails = re.split(r"[;,\s]\s*", self.recipient_emails)

        # validate emails
        if invalid_emails := [
            email for email in self.recipient_emails if not validators.email(email)
        ]:
            Status.WARNING(
                f"Invalid recipient email addresses found: {', '.join(invalid_emails)}"
            )

        # remove invalid emails
        self.recipient_emails = [
            email for email in self.recipient_emails if validators.email(email)
        ]
        self.recipient_emails = list(set(self.recipient_emails))

        if not self.recipient_emails:
            Status.WARNING("No recipient emails found")

    def is_connected(self) -> bool:
        """
        Checks if the SMTP server is connected successfully.

        This method attempts to connect to the SMTP server using the provided
        server address, port, username, and password. If the connection is
        successful, it logs an informational message and returns True. If the
        connection fails, it logs a failure message with the exception details
        and returns False.

        Returns:
            bool: True if the connection to the SMTP server is successful, False otherwise.
        """
        try:
            return self._is_connected()
        except smtplib.SMTPServerDisconnected:
            Status.FAILED(
                "SMTP server connection failed: server disconnected", traceback=False
            )
            return False
        except Exception as e:
            Status.FAILED(
                "SMTP server connection failed", error=str(e), traceback=False
            )
            return False

    def _is_connected(self):
        if not self.smtp_server or not self.smtp_port:
            Status.FAILED("SMTP server or port not configured")
            return False
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.ehlo()  # Identifies your client to the server
        server.starttls()
        server.ehlo()  # Re-identify
        if self.password:
            server.login(self.username, self.password)
        server.quit()
        Status.INFO("SMTP server connection successful")
        return True

    def add_content(self, title: str, content: Any, position: int = None) -> None:
        """
        Adds content to the notification list with an optional position.
        Args:
        -----
            title (str): The title of the content, which will be converted to bold.
            content (Any): The content to be added.
            position (int, optional): The position at which to insert the content. If None, the content is appended to the end.

        Returns:
            None
        Raises:
            Status.WARNING: If duplicate content is found, a warning is issued and the content is not added.
        """
        # make sure we are not adding duplicate content
        is_duplicate = any(content == item for item in self.contents)
        if is_duplicate:
            Status.WARNING("Duplicate content found, skipping.")
            return

        # convert title to bold
        title = f"<b>{self._clean_string(title)}</b>"

        # check if content is a Status object
        if isinstance(content, Status):
            content = content.to_dict()

        # insert at specific position
        if position is not None:
            self.contents.insert(position, title)
            self.contents.insert(position + 1, content)
        else:
            # append to the end
            self.contents.append(title)
            self.contents.append(content)

    def _clean_string(self, data: Any) -> str:
        """
        Cleans the input string by removing leading and trailing whitespace,
        replacing underscores with spaces, and conditionally capitalizing it.
        Args:
            data (Any): The input data to be cleaned. If the input is not a string,
                        it will be returned as is.
        Returns:
            str: The cleaned and conditionally capitalized string.
        """
        if not isinstance(data, str):
            return data

        data = str(data).strip()
        # Remove _ from string
        data = data.replace("_", " ")

        return self._capitalize_conditionally(data)

    def _capitalize_conditionally(self, data: str) -> str:
        """
        Capitalizes words conditionally in a given string.
        This method processes each word in the input string and capitalizes it if it
        contains one or fewer uppercase letters. If a word contains more than one
        uppercase letter, it is left unchanged.
        Args:
            data (str): The input string containing words to be processed.
        Returns:
            str: A new string with words capitalized conditionally.
        """

        def process_word(word: str):
            # Check if the word has more than one uppercase letter
            if sum(bool(char.isupper()) for char in word) > 1:
                return word  # Leave the word as it is

            return word.capitalize()  # Capitalize the word

        return " ".join([process_word(word) for word in data.split()])

    def _str_to_html(self, data: str) -> str:
        """
        Convert a plain text string to an HTML paragraph.

        Args:
            data (str): The plain text string to be converted.

        Returns:
            str: The HTML formatted string.
        """
        # Convert string to HTML
        return f"<p>{data}</p>"

    def _dict_to_html(self, data: Dict, border_width: int = 1) -> str:
        """
        Convert a dictionary to an HTML table.
        Args:
            data (Dict): The dictionary to convert to HTML.
            border_width (int, optional): The width of the table border. Defaults to 1.
        Returns:
            str: The HTML representation of the dictionary.
        """
        # Convert dictionary to HTML table
        # Create index column, and sub columns if its a nested dictionary
        html = f"<table border={border_width} style='border: none; border-collapse: collapse; width: 100%;'>"
        for key, value in data.items():
            if isinstance(value, Dict):
                html += f"<tr><td>{self._clean_string(key)}</td><td>{self._dict_to_html(value, 0)}</td></tr>"
            elif isinstance(value, (List, Set)):
                html += f"<tr><td>{self._clean_string(key)}</td><td>{self._list_to_html(value, 0)}</td></tr>"
            else:
                html += (
                    f"<tr><td>{self._clean_string(key)}</td><td>{str(value)}</td></tr>"
                )

        html += "</table>"

        return html

    def _list_to_html(self, data: Union[List, Set], border_width: int = 1) -> str:
        """
        Convert a list or set to an HTML table representation.
        Args:
            data (Union[List, Set]): The list or set to be converted to HTML.
            border_width (int, optional): The width of the table border. Defaults to 1.
        Returns:
            str: The HTML string representing the table.
        """
        # Convert list to HTML table
        # Create index column, and sub columns if its a nested list
        html = f"<table border={border_width} style='border: none; border-collapse: collapse; width: 100%;'>"
        for value in data:
            if isinstance(value, (List, Set)):
                html += f"<tr><td>{self._list_to_html(value, 0)}</td></tr>"
            elif isinstance(value, Dict):
                html += f"<tr><td>{self._dict_to_html(value, 0)}</td></tr>"
            else:
                html += f"<tr><td>{str(value)}</td></tr>"

        html += "</table>"

        return html

    def _df_to_html(self, data: pd.DataFrame) -> str:
        """
        Converts a pandas DataFrame to an HTML table string.
        Args:
            data (pd.DataFrame): The DataFrame to convert.
        Returns:
            str: The HTML table representation of the DataFrame.
        """
        # clean column names
        data.columns = [self._clean_string(col) for col in data.columns]

        # Convert DataFrame to HTML table
        return data.to_html(index=False, border=0)

    def _format_email_body(self) -> Union[str, None]:
        """
        Formats the email body by converting the contents into HTML format.
        Returns:
            Union[str, None]: The formatted email body as an HTML string, or None if there are no contents to send.
        Raises:
            Status.NOT_FOUND: If no contents are found or the instance is not found.
            Status.NOT_IMPLEMENTED: If an unsupported content type is encountered.
        Notes:
            - Adds a header with the KonaAI logo.
            - Converts each content item to HTML based on its type (str, dict, list, pd.DataFrame).
            - Adds a footer to the email body.
        """
        # check items in contents
        if not self.contents:
            Status.NOT_FOUND("No contents to send in email")
            return None

        # get instance details
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND(f"Instance with ID {self.instance_id} not found.")
            return None

        # keep only relevant instance data
        instance_data = {
            "instance_id": self.instance_id,
            "client_name": instance.client_name,
            "project_name": instance.project_name,
        }
        self.add_content("Project Instance", instance_data, position=0)

        # create email body
        email_body = '<img src="cid:KonaAI Logo" style="width: 50%; height: auto;"><br>'
        for content in self.contents:
            if isinstance(content, str):
                email_body += self._str_to_html(content)
            elif isinstance(content, Dict):
                email_body += self._dict_to_html(content)
            elif isinstance(content, list):
                email_body += self._list_to_html(content)
            elif isinstance(content, pd.DataFrame):
                email_body += self._df_to_html(content)
            else:
                Status.NOT_IMPLEMENTED(f"Unsupported content type: {type(content)}")

            email_body += "<p></p>"  # add space between contents

        # add footer
        email_body += "<hr>"  # add horizontal line
        email_body += self._str_to_html(self.footer)
        return email_body

    def attach(self, file_path: str):
        """
        Attach a file to the email.

        Args:
        ----
            file_path (str): The path to the file to be attached.
        """
        # Attach file to email
        if not pathlib.Path(file_path).exists():
            Status.NOT_FOUND(f"File not found: {file_path}")
            return
        self.attachments.append(file_path)

    def send(
        self, subject: str, retry_count: int = 3, retry_delay_min: int = 2
    ) -> bool:
        """
        Sends an email with the specified subject, attachments, and body content.
        Args:
        -----
            subject (str): The subject of the email.
            retry_count (int, optional): The number of times to retry sending the email in case of failure. Defaults to 3.
            retry_delay_min (int, optional): The delay in minutes between retries. Defaults to 2.

        Returns:
            bool: True if the email was successfully sent, False otherwise.
        """

        # check connection
        if not self.is_connected():
            Status.FAILED(f"Failed to connect to email server. {self}")
            return False

        msg = MIMEMultipart()
        msg["From"] = self.from_email
        msg["To"] = ",".join(self.recipient_emails)
        msg["Subject"] = f"{subject} [Instance ID {self.instance_id}]"  # Email subject

        # add header
        if self.header:
            with open(self.header, "rb") as file:
                part = MIMEImage(file.read(), Name="KonaAI Logo")
                part.add_header("Content-ID", "KonaAI Logo")
                msg.attach(part)

        # add attachments
        if self.attachments:
            for file_path in self.attachments:
                # attach file to email
                file_name = pathlib.Path(file_path).name

                with open(file_path, "rb") as file:
                    part = MIMEApplication(file.read(), Name=file_name)
                    part["Content-Disposition"] = f'attachment; filename="{file_name}"'
                    msg.attach(part)

        # add body
        email_body = self._format_email_body()
        if not email_body:
            Status.NOT_FOUND("Email body is empty")
            return False
        msg.attach(MIMEText(email_body, "html"))

        # send email
        is_delivered = False
        for _ in range(retry_count):
            try:
                self._send_email(msg)
                is_delivered = True
                break
            except BaseException as e:
                Status.FAILED(
                    f"Email send failed. Retrying in {retry_delay_min} minutes.",
                    error=str(e),
                )
                time.sleep(retry_delay_min * 60)

        # clear all contents and attachments
        self.contents.clear()
        return bool(is_delivered)

    def _send_email(
        self,
        msg,
    ) -> None:
        """
        Sends an email message using the configured SMTP server.
        Args:
            msg (email.message.EmailMessage): The email message to be sent.
        Raises:
            ValueError: If the connection to the email server fails.
        Returns:
            None
        """
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        if self.password:
            server.login(self.username, self.password)
        server.sendmail(self.from_email, self.recipient_emails, msg.as_string())
        server.quit()
        Status.INFO("Email sent successfully", recipients=self.recipient_emails)
