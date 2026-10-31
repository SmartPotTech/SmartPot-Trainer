from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.core.security import require_service_token
from app.learning.service import TrainerService, to_stored
from app.schemas.learning import IngestResult, LearningBatch, LearningStatus, TrainRequest

router = APIRouter(prefix="/v1/learning", tags=["Entrenamiento"], dependencies=[Depends(require_service_token)])


def _service(request: Request) -> TrainerService:
    return request.app.state.service


@router.post(
    "/readings",
    response_model=IngestResult,
    summary="Recibir lecturas reales para entrenamiento",
    description="La API envía por lotes las lecturas de los cultivos reales. Se guardan seudonimizadas para "
    "alimentar los próximos entrenamientos de modelos.",
)
def ingest(batch: LearningBatch, request: Request) -> IngestResult:
    service = _service(request)
    stored = service.ingest(
        [
            to_stored(
                item.crop_id,
                item.crop_type,
                item.measured_at.timestamp(),
                item.local_hour,
                item.measures.as_dict(),
            )
            for item in batch.readings
        ]
    )
    return IngestResult(received=len(batch.readings), stored=stored, total=service.store.total())


@router.get(
    "/status",
    response_model=LearningStatus,
    summary="Estado de modelos y aprendizaje continuo",
    description="Lecturas por especie, calidad de datos, comparación de modelos vigentes, estados de operación "
    "y detector de lecturas atípicas.",
)
def status(request: Request) -> LearningStatus:
    return LearningStatus.model_validate(_service(request).status())


@router.post(
    "/train",
    response_model=LearningStatus,
    summary="Entrenar modelos ahora",
    description="Entrena la especie indicada o todas las que tengan suficientes datos nuevos acumulados.",
)
async def train(payload: TrainRequest, request: Request) -> LearningStatus:
    service = _service(request)
    crop_types = [payload.crop_type.upper()] if payload.crop_type else service.due()
    for crop_type in crop_types:
        if crop_type not in service.counts():
            raise HTTPException(status_code=404, detail=f"No hay lecturas de {crop_type}")
        await run_in_threadpool(service.train, crop_type)
    return LearningStatus.model_validate(service.status())


@router.delete(
    "/crops/{crop_id}",
    summary="Olvidar un cultivo",
    description="Borra las lecturas de un cultivo eliminado para garantizar privacidad y evitar sesgo.",
)
def forget(crop_id: str, request: Request) -> dict[str, int]:
    return {"deleted": _service(request).forget(crop_id)}


@router.post(
    "/prune",
    summary="Podar lecturas antiguas",
    description="Elimina lecturas que superen los días de retención o el tope de filas por especie.",
)
def prune(request: Request) -> dict[str, int]:
    service = _service(request)
    removed = service.store.prune(service.config.retention_days, service.config.max_rows)
    return {"removed": removed}
