import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.learning.service import TrainerConfig, TrainerService
from app.routers import health, learning


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = TrainerService(
            TrainerConfig(
                data_dir=settings.data_dir or None,
                min_samples=settings.learning_min_samples,
                retrain_every=settings.learning_retrain_every,
                check_seconds=settings.learning_check_seconds,
                retention_days=settings.learning_retention_days,
                max_rows=settings.learning_max_rows,
                training_rows=settings.learning_training_rows,
                separate_process=settings.learning_separate_process,
                smartpot_ai_url=settings.smartpot_ai_url,
                smartpot_ai_token=settings.notify_token,
                database_url=settings.database_url,
                timezone=settings.ai_timezone,
                model_export_onnx=settings.model_export_onnx,
                model_signing_key_path=settings.model_signing_key_path,
            )
        )
        app.state.service.start()
        yield
        app.state.service.stop()

    app = FastAPI(
        title="SmartPot Trainer",
        version="1.0.0",
        description="SmartPot decoupled continuous learning and model training service "
        "(Champion-Challenger, K-Means, Isolation Forest, LoRA adapters, and model persistence).",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid data", "errors": errors})

    app.include_router(health.router)
    app.include_router(learning.router)
    return app


app = create_app()
