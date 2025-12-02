# # Copyright (C) KonaAI - All Rights Reserved
"""This file is used to configure pytest for the entire project."""
import logging
import pathlib
import sys
import warnings

import pytest
from fastapi.testclient import TestClient
from src.utils.auth import generate_token
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.webserver import app as webapp

# ignore warnings
warnings.filterwarnings("ignore")

# List all active loggers
for logger_name in logging.Logger.manager.loggerDict:
    if logger_name != "KonaAIML":
        # set all loggers to warning level except KonaAIML
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# this file path
cwd = pathlib.Path(__file__).parent.absolute()
# add src to path
if str(cwd.parent) not in sys.path:
    sys.path.append(str(cwd.parent))


# create temporary Instance for testing
@pytest.fixture(scope="session", autouse=True)
def create_instance():
    """
    Fixture that creates a temporary instance for testing.

    This fixture is automatically used in all tests. It creates an instance with a unique ID and
    client/project information, and then deletes it after the test is completed.
    """
    current_active_instance = GlobalSettings().active_instance_id
    instance = GlobalSettings.instance_by_id(current_active_instance)

    if not instance:
        pytest.exit("Instance not created")

    Status.INFO(
        f"Instance created with ID: {instance.instance_id}, "
        f"Client Name: {instance.client_name}, Project Name: {instance.project_name}"
    )

    # set the instance as active
    gs = GlobalSettings()
    gs.active_instance_id = instance.instance_id
    gs.save_settings()
    Status.INFO(f"Active instance set to: {str(instance.instance_id)}")

    # yield the instance for testing
    yield instance


@pytest.fixture(scope="session")
def client(_app=webapp):
    """
    Fixture that provides a test client for the Flask application.

    Args:
        _app: The Fast API application instance.

    Returns:
        A test client for the Flask application.
    """
    with TestClient(_app) as c:
        yield c


@pytest.fixture(scope="session")
def authorization():
    """Fixture that provides a JWT token for the FastAPI application."""

    gs = GlobalSettings()
    active_instance_id = gs.active_instance_id
    instance = GlobalSettings.instance_by_id(active_instance_id)
    if not instance:
        pytest.exit("No active instance found")

    # backup current settings
    current_settings = instance.settings.model_copy(deep=True)

    # update JWT values in instance settings
    instance.settings.jwt.CertificateType = "secretkey"
    instance.settings.jwt.Issuer = "test"
    instance.settings.jwt.Audience = "test"
    instance.settings.jwt.SecretKey = "test"
    instance.save_settings()

    _auth = {}
    client_id = GlobalSettings.instance_by_id(
        GlobalSettings().active_instance_id
    ).ClientUID
    project_id = GlobalSettings.instance_by_id(
        GlobalSettings().active_instance_id
    ).ProjectUID
    if token := generate_token(
        client_id=client_id, project_id=project_id, expiry_minutes=180
    ):
        _auth = {"Authorization": token}
    else:
        pytest.exit("Could not generate JWT token")

    yield _auth

    # restore original instance settings
    instance.settings = current_settings
    instance.save_settings()


# specify order of each application module
def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument
    """Modify the order of the tests."""
    pass
