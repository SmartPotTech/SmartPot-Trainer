from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Training service status")
def health(request: Request) -> dict[str, str]:
    service = getattr(request.app.state, "service", None)
    persistent = service.store.persistent if service else False
    return {
        "status": "UP",
        "service": "smartpot-trainer",
        "storage": "PERSISTENT" if persistent else "MEMORY",
        "trainer": "READY" if service else "INITIALIZING",
    }
