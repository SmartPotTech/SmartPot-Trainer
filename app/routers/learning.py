from fastapi import APIRouter, Depends, HTTPException, Path, Request
from starlette.concurrency import run_in_threadpool

from app.core.security import require_service_token
from app.learning.service import TrainerService, to_stored
from app.schemas.learning import IngestResult, LearningBatch, LearningStatus, TrainRequest

router = APIRouter(prefix="/v1/learning", tags=["Training"], dependencies=[Depends(require_service_token)])


def _service(request: Request) -> TrainerService:
    return request.app.state.service


@router.post(
    "/readings",
    response_model=IngestResult,
    summary="Ingest real readings for training",
    description="The API submits batches of real crop readings. They are stored pseudonymized to "
    "feed upcoming model training runs.",
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
    summary="Model and continuous learning status",
    description="Readings per species, data quality, active model comparisons, operating states, "
    "and outlier detector status.",
)
def status(request: Request) -> LearningStatus:
    return LearningStatus.model_validate(_service(request).status())


@router.post(
    "/train",
    response_model=LearningStatus,
    summary="Train models now",
    description="Trains the specified crop species or all species with enough newly accumulated readings.",
)
async def train(payload: TrainRequest, request: Request) -> LearningStatus:
    service = _service(request)
    crop_types = [payload.crop_type.upper()] if payload.crop_type else service.due()
    for crop_type in crop_types:
        if crop_type not in service.counts():
            raise HTTPException(status_code=404, detail=f"No readings for {crop_type}")
        await run_in_threadpool(service.train, crop_type)
    return LearningStatus.model_validate(service.status())


@router.delete(
    "/crops/{crop_id}",
    summary="Forget a crop",
    description="Deletes stored readings for a removed crop to preserve privacy and prevent bias.",
)
def forget(
    request: Request,
    crop_id: str = Path(..., pattern=r"^[0-9a-f]{24}$"),
) -> dict[str, int]:
    return {"deleted": _service(request).forget(crop_id)}


@router.post(
    "/prune",
    summary="Prune old readings",
    description="Removes readings exceeding the retention window or per-species row cap.",
)
def prune(request: Request) -> dict[str, int]:
    return {"removed": _service(request).prune()}
