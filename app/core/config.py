from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Autenticación: token compartido con la API y SmartPot-AI
    smartpot_trainer_token: str = Field(
        min_length=16,
        description="Bearer authentication token for training endpoints",
    )
    smartpot_ai_token: str | None = Field(
        None,
        description="Token used to authenticate when notifying SmartPot-AI of new models",
    )

    # Red y URLs
    port: int = Field(8001, ge=1024, le=65535)
    smartpot_ai_url: str = Field(
        "http://ai-smartpot:8000",
        description="Base URL of SmartPot-AI for triggering model reloads",
    )
    ai_grpc_socket: str = Field(
        "/var/run/smartpot/ai.sock",
        description="Unix gRPC socket toward SmartPot-AI",
    )
    valkey_socket: str = Field(
        "/var/run/smartpot/valkey.sock",
        description="Unix socket toward Valkey",
    )
    docs_enabled: bool = False
    log_level: str = "INFO"
    model_seed: int = 42

    # Directorio de datos y modelos compartidos
    data_dir: str = "data"
    database_url: str | None = Field(
        None,
        description="ADBC/PostgreSQL connection string toward TimescaleDB (read-only role)",
    )
    # Misma zona que usa la API (AI_TIMEZONE) para la hora local de cada lectura.
    ai_timezone: str = "America/Bogota"
    model_export_onnx: bool = Field(True, description="Export Champion models to signed ONNX format")
    model_signing_key_path: str | None = Field(None, description="Path to Ed25519 private key")

    # Parámetros de entrenamiento
    learning_min_samples: int = Field(200, ge=50)
    learning_retrain_every: int = Field(300, ge=10)
    learning_check_seconds: int = Field(120, ge=0, description="Retraining loop interval in seconds (0=disabled)")
    learning_retention_days: int = Field(60, ge=1)
    learning_max_rows: int = Field(200_000, ge=1_000)
    learning_training_rows: int = Field(12_000, ge=500)
    learning_separate_process: bool = True

    @property
    def notify_token(self) -> str:
        return self.smartpot_ai_token or self.smartpot_trainer_token


@lru_cache
def get_settings() -> TrainerSettings:
    return TrainerSettings()
