from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from app.knowledge.profiles import PROFILES


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Measures(CamelModel):
    temperature: float | None = Field(None, ge=-20, le=60)
    humidity: float | None = Field(None, ge=0, le=100)
    brightness: float | None = Field(None, ge=0, le=200_000)
    ph: float | None = Field(None, ge=0, le=14)
    tds: float | None = Field(None, ge=0, le=10_000)
    atmosphere: float | None = Field(None, ge=300, le=1_100)
    soil_moisture: float | None = Field(None, ge=0, le=100)
    measured_at: datetime | None = None

    def as_dict(self) -> dict[str, float | None]:
        return self.model_dump(by_alias=True, exclude={"measured_at"})


class LearningReading(CamelModel):
    crop_id: str = Field(min_length=1, max_length=64)
    crop_type: str
    measured_at: datetime
    local_hour: int | None = Field(None, ge=0, le=23)
    measures: Measures

    @field_validator("crop_type")
    @classmethod
    def known_crop(cls, value: str) -> str:
        if value.upper() not in PROFILES:
            raise ValueError(f"Unknown crop type: {value}")
        return value.upper()


class LearningBatch(CamelModel):
    readings: list[LearningReading] = Field(max_length=2000)


class IngestResult(CamelModel):
    received: int
    stored: int
    total: int


class TrainRequest(CamelModel):
    crop_type: str | None = None


class CandidateOut(CamelModel):
    model: str
    score: float | None = None
    std: float | None = None


class ModelCardOut(CamelModel):
    task: str
    label: str
    status: str
    reason: str | None = None
    metric: str
    model: str | None = None
    score: float | None = Field(None, description="Time-series cross-validation mean score")
    std: float | None = None
    holdout: float | None = Field(None, description="Score on the most recent 20% holdout readings")
    baseline: float | None = Field(None, description="Baseline score on the same holdout readings")
    samples: int
    positives: int | None = None
    version: int
    trained_at: datetime | None = None
    params: dict[str, float | int | str | None] = Field(default_factory=dict)
    candidates: list[CandidateOut] = Field(default_factory=list)


class ClusterOut(CamelModel):
    label: str
    description: str
    share: float


class StatesOut(CamelModel):
    k: int
    silhouette: float
    clusters: list[ClusterOut]


class AnomalyOut(CamelModel):
    samples: int
    contamination: float


class QualityOut(CamelModel):
    rows: int
    completeness: float
    validity: float
    outliers: int
    score: float


class CropTypeLearning(CamelModel):
    crop_type: str
    name: str
    readings: int
    crops: int
    new_since_training: int
    training: bool
    trained_at: datetime | None = None
    quality: QualityOut | None = None
    models: list[ModelCardOut] = Field(default_factory=list)
    states: StatesOut | None = None
    anomaly: AnomalyOut | None = None


class LearningStatus(CamelModel):
    enabled: bool
    persistent: bool
    stored_readings: int
    min_samples: int
    retrain_every: int
    crop_types: list[CropTypeLearning]
