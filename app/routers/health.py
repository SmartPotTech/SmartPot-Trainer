from fastapi import APIRouter, Request

router = APIRouter(tags=["Salud"])


@router.get("/health", summary="Estado del servicio de entrenamiento")
def health(request: Request) -> dict[str, str]:
    service = getattr(request.app.state, "service", None)
    persistent = service.store.persistent if service else False
    return {
        "status": "UP",
        "service": "smartpot-trainer",
        "storage": "PERSISTENT" if persistent else "MEMORY",
        "trainer": "READY" if service else "INITIALIZING",
    }
