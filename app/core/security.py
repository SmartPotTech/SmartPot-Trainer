import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import TrainerSettings, get_settings

bearer = HTTPBearer(auto_error=False)


def require_service_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    settings: Annotated[TrainerSettings, Depends(get_settings)],
) -> None:
    """Internal service guard: only the API and SmartPot-AI with the shared token may access it."""
    # compare_digest sobre bytes: una cabecera no ASCII debe dar 401, nunca un TypeError/500.
    supplied = credentials.credentials.encode("utf-8") if credentials is not None else b""
    accepted = [settings.smartpot_trainer_token.encode("utf-8")]
    if settings.smartpot_ai_token:
        accepted.append(settings.smartpot_ai_token.encode("utf-8"))
    if credentials is None or not any(secrets.compare_digest(supplied, token) for token in accepted):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
