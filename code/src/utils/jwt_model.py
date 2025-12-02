# # Copyright (C) KonaAI - All Rights Reserved
"""JWTModel defines the configuration for JWT (JSON Web Token) authentication."""
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import SecretStr


class JWTModel(BaseModel):
    """
    JWTModel defines the configuration for JWT (JSON Web Token) authentication.
    Attributes:
    ----------
        Audience (Optional[str]): The audience for the JWT token.
        Issuer (Optional[str]): The issuer of the JWT token.
        CertificateType (Literal["SecretKey", "PemKeys", "AzureKeyVault"]):
            Type of certificate used for JWT signing. Defaults to "SecretKey".
        CertificateName (Optional[str]):
            Name of the certificate (used for PEMKeys or AzureKeyVault).
        AzureKeyVaultName (Optional[str]):
            Azure Key Vault name (used if CertificateType is AzureKeyVault).
        SecretKey (Optional[SecretStr]):
            Secret key for signing JWT (used if CertificateType is SecretKey).
    Config:
        model_config: Custom JSON encoder for SecretStr fields.
    """

    Audience: Optional[str] = Field(
        default=None, description="The audience for the JWT token"
    )
    Issuer: Optional[str] = Field(
        default=None, description="The issuer of the JWT token"
    )
    CertificateType: Literal["SecretKey", "PemKeys", "AzureKeyVault"] = Field(
        default="SecretKey", description="Type of certificate used for JWT signing"
    )
    CertificateName: Optional[str] = Field(
        default=None,
        description="Name of the certificate (used for PEMKeys or AzureKeyVault)",
    )
    AzureKeyVaultName: Optional[str] = Field(
        default=None,
        description="Azure Key Vault name (used if CertificateType is AzureKeyVault)",
    )
    SecretKey: Optional[SecretStr] = Field(
        default=None,
        description="Secret key for signing JWT (used if CertificateType is SecretKey)",
    )

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: (
                v.get_secret_value() if isinstance(v, SecretStr) else v
            ),
        },
    )

    @field_validator("CertificateType", mode="before")
    @classmethod
    def validate_certificate_type(cls, v):
        """Validates the provided certificate type."""
        valid_types = {"SecretKey", "PemKeys", "AzureKeyVault"}
        return "SecretKey" if not v or v not in valid_types else v
