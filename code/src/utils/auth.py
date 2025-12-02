# # Copyright (C) KonaAI - All Rights Reserved
"""This module handles authentication related tasks"""
import base64
import os
import time
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.keys import KeyClient
from azure.keyvault.secrets import SecretClient
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs12
from jose import jwt
from src.utils.global_config import GlobalSettings

from .conf import Setup
from .status import Status


def generate_token(
    client_id: str, project_id: str, expiry_minutes: int = 30
) -> Optional[str]:
    """
    Generates a JWT token for the specified client and project.
    Args:
        client_id (str): The client identifier.
        project_id (str): The project identifier.
        expiry_minutes (int, optional): Token expiry time in minutes. Defaults to 30.
    Returns:
        Optional[str]: The generated JWT token as a Bearer string, or None if generation fails.
    Raises:
        None explicitly. Handles exceptions internally and returns None on failure.
    Notes:
        - Uses global settings to retrieve JWT configuration and private key.
        - Supports both HS256 and RS256 algorithms based on certificate type.
        - Logs status messages for informational and error events.
    """
    try:
        Status.INFO("Generating JWT token...")
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            Status.NOT_FOUND(
                f"Instance not found for client_id: {client_id}, project_id: {project_id}"
            )
            return None

        # Get the certificate type and other parameters from global settings
        jwt_settings = instance.settings.jwt
        if not jwt_settings:
            Status.FAILED("JWT settings not found in instance configuration.")
            return None

        if not jwt_settings.Issuer or not jwt_settings.Audience:
            Status.FAILED("Issuer or Audience is not configured in JWT settings.")
            return None

        cert_type = jwt_settings.CertificateType
        issuer = jwt_settings.Issuer
        audience = jwt_settings.Audience
        private_key = get_private_key(client_id, project_id)

        if not all([cert_type, issuer, audience, private_key]):
            Status.FAILED("Missing required parameters to generate JWT token")
            return None

        payload = {
            "iss": issuer,
            "aud": audience,
            "iat": int(time.time()),
            "exp": int(time.time()) + (expiry_minutes * 60),
        }
        token = jwt.encode(
            claims=payload,
            key=private_key,
            algorithm="HS256" if "secretkey" in str(cert_type).lower() else "RS256",
        )

        return f"Bearer {token}"
    except BaseException as _e:
        Status.FAILED("Can not generate JWT token.", error=str(_e), traceback=False)
    return None


def get_private_key(client_id: str, project_id: str) -> Optional[bytes]:
    """
    Retrieves the private key for a given client and project.
    The method determines the certificate type from global settings and fetches the private key accordingly:
    - If the certificate type is 'SecretKey', returns the secret key as bytes.
    - If the certificate type is 'AzureKeyVault', retrieves the key from Azure Key Vault.
    - If the certificate type is 'PemKeys', loads the private key from a PEM file and returns it as a UTF-8 string.
    Args:
        client_id (str): The client identifier.
        project_id (str): The project identifier.
    Returns:
        Optional[bytes]: The private key in bytes or UTF-8 string, or None if not found or on error.
    """
    try:
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            raise ValueError(
                f"Instance not found for client_id: {client_id}, project_id: {project_id}"
            )

        jwt_settings = instance.settings.jwt
        if not jwt_settings:
            raise ValueError("JWT settings not found in instance configuration.")

        if not jwt_settings.Issuer or not jwt_settings.Audience:
            raise ValueError("Issuer or Audience is not configured in JWT settings.")

        if not (cert_type := jwt_settings.CertificateType):
            raise ValueError("No certificate type configured.")

        Status.INFO(f"Certificate type: {cert_type}")

        # if global settings certificate type is SecretKey
        if "secretkey" in str(cert_type).lower():
            key = jwt_settings.SecretKey.get_secret_value()
            return str(key).encode(encoding="utf-8")

        # if global settings certificate type is AzureKeyVault
        if "keyvault" in str(cert_type).lower():
            key = get_azure_private_key(
                jwt_settings.AzureKeyVaultName, jwt_settings.CertificateName
            )
            return key

        private_key_file = None
        # if global settings certificate type is PemKeys
        if "pemkeys" in str(cert_type).lower():
            private_key_file = os.path.join(
                Setup().db_path,
                instance.instance_id,
                Setup().global_constants.get("JWT", {}).get("PRIVATE_KEY_FILE"),
            )

        if os.path.exists(private_key_file):
            with open(private_key_file, "rb") as _f:
                private_key = serialization.load_pem_private_key(
                    _f.read(), password=None, backend=default_backend()
                )

                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                return private_pem.decode("utf-8")
        else:
            raise FileNotFoundError("Private key file does not exist")
    except BaseException as _e:
        Status.FAILED(
            "Error while fetching private key.", error=str(_e), traceback=False
        )
    return None


def get_azure_private_key(vault_name: str, certificate_name: str) -> Optional[bytes]:
    """
    Retrieves the private key from an Azure Key Vault certificate.
    Args:
        vault_name (str): The name of the Azure Key Vault.
        certificate_name (str): The name of the certificate stored in the Key Vault.
    Returns:
        Optional[bytes]: The private key in bytes if found, otherwise None.
    Raises:
        azure.core.exceptions.ResourceNotFoundError: If the specified certificate is not found in the Key Vault.
        azure.core.exceptions.ClientAuthenticationError: If authentication to Azure Key Vault fails.
        ValueError: If the certificate cannot be decoded or the private key cannot be extracted.
    """
    vault_url = f"https://{vault_name.strip()}.vault.azure.net"
    certificate_name = certificate_name.strip()

    client = SecretClient(
        vault_url=vault_url,
        credential=DefaultAzureCredential(logging_enable=False),
    )
    secret = client.get_secret(certificate_name)

    # extract private key from certificate in PEM format
    cert_bytes = base64.b64decode(secret.value)
    private_key, _, _ = pkcs12.load_key_and_certificates(cert_bytes, None)
    return private_key


def get_public_key(client_id: str, project_id: str) -> Optional[bytes]:
    """
    Retrieves the public key used for JWT authentication based on the provided client and project IDs.
    The method determines the certificate type from the global settings and fetches the public key accordingly:
    - For "SecretKey" (HS256): Returns the secret key as bytes.
    - For "KeyVault" (RS256): Fetches the public key from Azure Key Vault.
    - For "PemKeys" (RS256): Loads the public key from a PEM file, supporting both certificate and public key formats.
    Args:
        client_id (str): The client identifier.
        project_id (str): The project identifier.
    Returns:
        Optional[bytes]: The public key in PEM format as bytes, or None if not found or an error occurs.
    Raises:
        ValueError: If the PEM file is neither a certificate nor a public key.
    """
    try:
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        jwt_settings = instance.settings.jwt
        cert_type = str(jwt_settings.CertificateType).lower()

        # SecretKey (HS256)
        if "secretkey" in cert_type:
            secret = jwt_settings.SecretKey.get_secret_value()
            return str(secret).encode("utf-8")

        # Azure Key Vault (RS256)
        if "keyvault" in cert_type:
            return get_azure_public_key(
                jwt_settings.AzureKeyVaultName, jwt_settings.CertificateName
            )

        # PEM Keys (RS256)
        if "pemkeys" in cert_type:
            pem_path = os.path.join(
                Setup().db_path,
                instance.instance_id,
                Setup().global_constants.get("JWT", {}).get("PUBLIC_KEY_FILE"),
            )
            with open(pem_path, "rb") as pem_file:
                pem_data = pem_file.read()

                if b"-----BEGIN CERTIFICATE-----" in pem_data:
                    # Its a certificate file
                    cert = x509.load_pem_x509_certificate(pem_data, default_backend())
                    public_key = cert.public_key()
                elif b"-----BEGIN PUBLIC KEY-----" in pem_data:
                    # Load the public key from the PEM file
                    public_key = serialization.load_pem_public_key(
                        pem_file.read(), backend=default_backend()
                    )
                else:
                    raise ValueError(
                        "PEM file is neither a certificate nor a public key."
                    )

                # Check if public_bytes is a public key object
                if not public_key:
                    Status.FAILED("Public key not found in PEM file.")
                    return None

                # Convert to PEM format
                return public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
        Status.NOT_FOUND("No valid certificate type configured.")
    except BaseException as _e:
        Status.FAILED(
            "Error while fetching public key.", error=str(_e), traceback=False
        )
    return None


def get_azure_public_key(vault_name: str, certificate_name: str) -> Optional[bytes]:
    """
    Retrieves the public key from an Azure Key Vault certificate.
    Args:
        vault_name (str): The name of the Azure Key Vault.
        certificate_name (str): The name of the certificate to retrieve.
    Returns:
        Optional[bytes]: The public key in PEM format, or None if not found.
    Raises:
        azure.core.exceptions.ResourceNotFoundError: If the specified certificate does not exist.
        azure.identity.CredentialUnavailableError: If authentication fails.
        Any other exceptions raised by the Azure SDK.
    """
    vault_name = vault_name.strip()
    cert_name = certificate_name.strip()

    client = KeyClient(
        vault_url=f"https://{vault_name}.vault.azure.net",
        credential=DefaultAzureCredential(),
    )
    key = client.get_key(cert_name)
    return key.key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
