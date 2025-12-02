# # Copyright (C) KonaAI - All Rights Reserved
"""FastAPI app configuration and security setup"""
import logging

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from jose import jwt
from src.utils.auth import get_public_key
from src.utils.global_config import GlobalSettings
from src.utils.status import Status

# Initialize FastAPI
app = FastAPI(
    title="KonaAI Intelligence API's",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

security = HTTPBearer()
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT Authentication dependency
async def validate_token(
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
):
    """
    Validates a JWT token provided in the request headers for a specific client and project.
    Args:
        client_id (str): The client ID, extracted from the "clientId" header.
        project_id (str): The project ID, extracted from the "projectId" header.
        credentials (HTTPAuthorizationCredentials): The HTTP authorization credentials containing the JWT token.
    Returns:
        dict: A dictionary containing the validated token and its decoded payload.
    Raises:
        HTTPException: If the token is missing, the client/project instance is not found, JWT configuration is missing,
                       public key is not found, or the token validation fails.
    """
    try:
        # get the token from credentials
        token = credentials.credentials
        if not token:
            Status.FAILED("No JWT token provided in the request")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No authentication credentials provided",
            )

        # Validate client_id and project_id
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            Status.NOT_FOUND(
                f"Instance not found for client_id: {client_id}, project_id: {project_id}"
            )
            raise HTTPException(
                status_code=404,
                detail="Instance not found for the provided client and project IDs",
            )
        config = instance.settings.jwt
        if config is None:
            Status.FAILED("JWT configuration is missing or corrupted.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT configuration is missing or corrupted.",
            )

        # Decode the JWT token
        public_key = get_public_key(client_id, project_id)
        if not public_key:
            Status.FAILED("Public key not found for JWT validation.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Public key not found for JWT validation.",
            )

        payload = jwt.decode(
            token,
            key=public_key,
            algorithms=[
                (
                    "HS256"
                    if "secretkey" in str(config.CertificateType).lower()
                    else "RS256"
                )
            ],
            audience=config.Audience,
            issuer=config.Issuer,
        )
        return {"token": token, "payload": payload}
    except BaseException as e:
        Status.FAILED("JWT validation failed: ", error=str(e), traceback=False)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication credentials",
        ) from e
