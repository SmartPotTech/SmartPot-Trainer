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
    """El servicio es interno: solo la API y SmartPot-AI, con el token compartido, pueden consultarlo."""
    if credentials is None or not (
        secrets.compare_digest(credentials.credentials, settings.smartpot_trainer_token)
        or (settings.smartpot_ai_token and secrets.compare_digest(credentials.credentials, settings.smartpot_ai_token))
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de servicio inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
