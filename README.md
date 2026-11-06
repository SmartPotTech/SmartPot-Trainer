# SmartPot Trainer (Continuous Learning & Model Training Plane)

> **Language / Idioma:** [English](#english) | [Español](#español)

---

<a id="english"></a>

## English

Decoupled continuous learning and model training microservice for SmartPot. Responsible for batch ingestion of sensor telemetry, Data Quality Score (DQS) evaluation with Polars, Champion-Challenger model tournaments with time-series cross-validation (`TimeSeriesSplit`), hyperparameter tuning via `GridSearchCV`, fast in-memory adaptation via LoRA (Low-Rank Adaptation), and atomic model persistence (native `smartpot.learned-models/1` JSON with Ed25519-signed manifests, plus ONNX and joblib inspection sidecars) on the shared volume.

---

### 1. Architecture & Plane Decoupling

To guarantee that the inference service (`SmartPot-AI`) maintains sub-millisecond latency (< 0.05 ms) free of CPU and memory contention jitter caused by heavy training jobs, the architecture is split into two complementary planes:

```
                  [ SmartPot-API / Clients ]
                                │
          ┌─────────────────────┴─────────────────────┐
          │ (Hot Inference)                           │ (Ingestion & Retraining)
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│       SmartPot-AI         │               │     SmartPot-Trainer      │
│  (Serving Plane / :8000)  │               │ (Training Plane / :8001)  │
│                           │               │                           │
│  • Rust + ONNX Core       │               │  • Polars LazyFrame DQS   │
│  • Base Models (Zero Cold)│               │  • TimeSeriesSplit (cv=4) │
│  • Hot Reload (< 5 ms)    │◄──(Reload)────│  • GridSearchCV           │
│  • Pure Inference Mode    │   Webhook     │  • LoRA Online Updates    │
└─────────────┬─────────────┘               └─────────────┬─────────────┘
              │                                           │
              └─────────────► [ /data ] ◄─────────────────┘
                              ai_data
                     ├── learning.db (SQLite WAL)
                     └── models/
                         ├── LETTUCE.joblib
                         └── TOMATO.joblib
```

#### Shared Volume Contract (`/data`)
- **`/data/learning.db`**: SQLite database managed with temporal indexes and cryptographic SHA-256 pseudonymization of crop identifiers (`crop_key`), ensuring privacy and data retention compliance.
- **`/data/models/{SPECIES}_inference.json`**: canonical native package with the three champion pipelines, K-Means states, Isolation Forest, LoRA adapters and exactly 16 scikit-learn control cases.
- **`/data/models/{SPECIES}_manifest.json`**: BLAKE2b-256 hashes and the Ed25519 signature; SmartPot-AI rejects anything not signed by `TRUSTED_MODEL_PUBLIC_KEY_HEX`.
- **`/data/models/{crop_type}.joblib` & `*.onnx`**: inspection sidecars only. joblib is never deserialized without a verified manifest signed by the pinned key.
- **`/data/keys/ed25519_private.pem`**: private signing key in a separate `trainer_keys` volume, mode `0600`, never inside `ai_data`.

---

### 2. Learning Capabilities & Models

1. **DQS (Data Quality Score) Evaluation**:
   - Parallel processing with `polars.LazyFrame`.
   - Strict physical bounds validation per agronomic variable.
   - Statistical outlier detection via Interquartile Range (IQR with $k=3.0$).
   - Physically impossible per-minute jumps in temperature, humidity and substrate, plus an Isolation Forest secondary filter before building the training set.

2. **Champion-Challenger Tournament**:
   - Forward-chaining cross-validation (`TimeSeriesSplit(n_splits=4)`) respecting temporal ordering.
   - Tournament across 7 classifiers (Logistic Regression, K-NN, Decision Tree, 120-tree Random Forest, HistGradientBoosting, and MLP Neural Network) against the Baseline.
   - Tournament across 5 regressors (Baseline, Ridge Regression, K-NN, Random Forest, HistGradientBoosting).
   - Automatic hyperparameter tuning via `GridSearchCV` on the top candidate.
   - A new model only replaces the active champion if it beats its metric on the most recent 20% holdout readings.

3. **Low-Rank Adapters (LoRA) in NumPy**:
   - Ultra-fast analytical adaptation (< 1 ms) per species:
     $$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
   - Adapts probabilities online without blocking CPU cores.

4. **Unsupervised Models**:
   - **Operating States**: K-Means clustering optimized via Silhouette analysis ($k \in [2, 6]$) describing crop state and time of day in natural language.
   - **Anomaly Detector**: 150-estimator `IsolationForest` to flag atypical variable combinations.

---

### 2.1 Model supply chain and fail-closed security

- The native `{SPECIES}_inference.json` package contains exactly 16 deterministic control cases; SmartPot-AI rejects a bundle with a missing, duplicated or malformed case.
- The Ed25519 private key lives in `MODEL_SIGNING_KEY_PATH` (default `/data/keys/ed25519_private.pem`, separate `trainer_keys` volume, mode `0600`). An existing corrupt key aborts startup; no ephemeral key is generated.
- `verify_manifest` compares against `TRUSTED_MODEL_PUBLIC_KEY_HEX`, not only the key declared inside the manifest.
- Persisted joblib files are never deserialized without a manifest signature and BLAKE2b artifact hash verified against that pinned key; without a trust anchor the Trainer retrains instead of loading them.
- A classifier whose training window (or a time fold) has only one class returns `PENDING` with an explanatory reason instead of raising HTTP 500.
- `DELETE /v1/learning/crops/{crop_id}` only accepts 24-character lowercase ObjectId hex; any other value returns 422. Non-ASCII Bearer headers return 401, never 500.

### 3. REST Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Public | Health probe (`UP`, `PERSISTENT` or `MEMORY` storage). |
| `POST` | `/v1/learning/readings` | Bearer Token | Batch ingestion of pseudonymized crop telemetry. |
| `GET` | `/v1/learning/status` | Bearer Token | Learned model status, readings per species, and DQS quality. |
| `POST` | `/v1/learning/train` | Bearer Token | On-demand training trigger by species or due species. |
| `DELETE` | `/v1/learning/crops/{id}` | Bearer Token | Forget stored readings for a deleted crop (right to erasure). |
| `POST` | `/v1/learning/prune` | Bearer Token | Forced pruning of old readings exceeding retention or row cap. |

---

### 4. Local Execution & Development

#### One-Command Deployment

The image is public: `ghcr.io/smartpottech/smartpot-trainer`. The complete platform, including this service, starts with a single command without cloning the repository:

```bash
curl -fsSL https://raw.githubusercontent.com/SmartPotTech/.github/main/docker/demo/compose.yaml | docker compose -f - up -d --wait
```

Open http://localhost:5173 and sign in with `demo@smartpot.app` / `SmartPot2026`. Stop with `docker compose -p smartpot-demo down` (add `-v` to remove data). More details in the [demo](https://github.com/SmartPotTech/.github/tree/main/docker/demo).

```bash
# Sync dependencies with uv
uv sync

# Run unit test suite
uv run pytest -v

# Validate code quality
uv run ruff check .

# Start development server on port 8001
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

### 5. Docker Container

The production image uses a multi-stage build based on `python:3.13-slim-bookworm` and `uv`, running as a non-privileged user (`1000:1000`):

```bash
docker pull ghcr.io/smartpottech/smartpot-trainer:latest
```

---

### 6. Performance Plan Status

| Component | Status |
|-------|--------|
| ADBC (Apache Arrow) reading from TimescaleDB | Optional, using read-only role `smartpot_reader`, real crops only, and pseudonymized key; without `DATABASE_URL` the trainer uses its local SQLite fed by the API |
| Polars pipeline | Filtering, rolling windows, and self-supervised labels |
| Fast LoRA path & deep Champion-Challenger path | Implemented |
| Signed native export | `smartpot.learned-models/1` JSON, 16 controls, BLAKE2b hashes, and Ed25519 manifest signature (ONNX is an optional sidecar) |

---

<a id="español"></a>

## Español

Microservicio desacoplado de entrenamiento y aprendizaje continuo de SmartPot. Responsable de la ingesta por lotes de telemetría de sensores, evaluación de calidad de datos con Polars, torneos de modelos con validación cruzada temporal (`TimeSeriesSplit`), ajuste de hiperparámetros con `GridSearchCV`, adaptación rápida en memoria mediante LoRA (Low-Rank Adaptation) y persistencia atómica de modelos (JSON nativo `smartpot.learned-models/1` con manifiestos firmados con Ed25519, más sidecars ONNX y joblib para inspección) en el volumen compartido.

---

### 1. Arquitectura y Desacoplamiento de Planos

Para garantizar que el servicio de inferencia (`SmartPot-AI`) mantenga latencias sub-milisegundo (< 0.05 ms) libres de fluctuaciones (*jitter*) causadas por la contención de CPU y memoria de entrenamientos masivos, la arquitectura se divide en dos planos complementarios:

```
                  [ SmartPot-API / Clientes ]
                                │
          ┌─────────────────────┴─────────────────────┐
          │ (Inferencia en caliente)                  │ (Ingesta y reentrenamiento)
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│       SmartPot-AI         │               │     SmartPot-Trainer      │
│  (Serving Plane / :8000)  │               │ (Training Plane / :8001)  │
│                           │               │                           │
│  • Núcleo Rust + ONNX     │               │  • Polars LazyFrame DQS   │
│  • Base Models (Zero Cold)│               │  • TimeSeriesSplit (cv=4) │
│  • Hot Reload (< 5 ms)    │◄──(Reload)────│  • GridSearchCV           │
│  • Pure Inference Mode    │   Webhook     │  • LoRA Online Updates    │
└─────────────┬─────────────┘               └─────────────┬─────────────┘
              │                                           │
              └─────────────► [ /data ] ◄─────────────────┘
                              ai_data
                     ├── learning.db (SQLite WAL)
                     └── models/
                         ├── LETTUCE.joblib
                         └── TOMATO.joblib
```

#### Contrato del Volumen Compartido (`/data`)
- **`/data/learning.db`**: Base de datos SQLite gestionada con índices temporales y seudonimización criptográfica SHA-256 de los identificadores de cultivo (`crop_key`), garantizando privacidad y cumplimiento de retención de datos.
- **`/data/models/{SPECIES}_inference.json`**: paquete nativo canónico con los tres pipelines campeones, estados K-Means, Isolation Forest, adaptadores LoRA y exactamente 16 casos de control de scikit-learn.
- **`/data/models/{SPECIES}_manifest.json`**: hashes BLAKE2b-256 y firma Ed25519; SmartPot-AI rechaza todo lo que no esté firmado con `TRUSTED_MODEL_PUBLIC_KEY_HEX`.
- **`/data/models/{crop_type}.joblib` y `*.onnx`**: sidecars de inspección. joblib nunca se deserializa sin un manifiesto verificado y firmado con la clave fijada.
- **`/data/keys/ed25519_private.pem`**: clave privada de firma en el volumen separado `trainer_keys`, modo `0600`, nunca dentro de `ai_data`.

---

### 2. Capacidades de Aprendizaje y Modelos

1. **Evaluación de Calidad DQS (Data Quality Score)**:
   - Procesamiento en paralelo con `polars.LazyFrame`.
   - Validación física estricta por variable agronómica.
   - Detección de atípicos estadísticos por Rango Intercuartílico (IQR con $k=3.0$).
   - Saltos físicamente imposibles por minuto en temperatura, humedad y sustrato, más un filtro secundario Isolation Forest antes de construir el conjunto de entrenamiento.

2. **Torneo Champion-Challenger**:
   - Validación cruzada hacia adelante (`TimeSeriesSplit(n_splits=4)`) respetando la flecha temporal.
   - Competencia entre 7 clasificadores (Regresión Logística, K-NN, Árbol de Decisión, Bosque Aleatorio de 120 árboles, HistGradientBoosting y Red Neuronal MLP) frente a la Línea Base.
   - Competencia entre 5 regresores (Línea Base, Regresión Ridge, K-NN, Bosque Aleatorio, HistGradientBoosting).
   - Ajuste automático de hiperparámetros con `GridSearchCV` sobre el mejor candidato.
   - El nuevo modelo solo reemplaza al campeón vigente si supera su métrica en el 20 % de lecturas más recientes (*holdout*).

3. **Adaptadores de Bajo Rango (LoRA) en NumPy**:
   - Adaptación analítica ultrarrápida (< 1 ms) por especie:
     $$W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
   - Permite adaptar probabilidades sin bloquear núcleos de CPU.

4. **Modelos No Supervisados**:
   - **Estados de Operación**: Agrupamiento K-Means optimizado mediante análisis de Silueta ($k \in [2, 6]$) describiendo en lenguaje natural el estado de los cultivos y momento del día.
   - **Detector de Anomalías**: `IsolationForest` de 150 estimadores para alertar sobre combinaciones atípicas de variables.

---

### 2.1 Cadena de suministro de modelos y seguridad fail-closed

- El paquete nativo `{SPECIES}_inference.json` contiene exactamente 16 casos de control deterministas; SmartPot-AI rechaza un bundle con casos faltantes, duplicados o mal formados.
- La clave privada Ed25519 vive en `MODEL_SIGNING_KEY_PATH` (por defecto `/data/keys/ed25519_private.pem`, volumen separado `trainer_keys`, modo `0600`). Una clave existente corrupta aborta el arranque; no se genera una clave efímera.
- `verify_manifest` compara contra `TRUSTED_MODEL_PUBLIC_KEY_HEX`, no solo contra la clave declarada dentro del manifiesto.
- Los joblib persistidos nunca se deserializan sin firma de manifiesto y hash BLAKE2b verificados con esa clave fijada; sin ancla de confianza el Trainer reentrena en vez de cargarlos.
- Un clasificador cuya ventana de entrenamiento (o un fold temporal) tenga una sola clase devuelve `PENDING` con una razón explicativa en vez de provocar HTTP 500.
- `DELETE /v1/learning/crops/{crop_id}` solo acepta ObjectId hexadecimal de 24 caracteres en minúsculas; cualquier otro valor devuelve 422. Una cabecera Bearer no ASCII devuelve 401, nunca 500.

### 3. Endpoints REST

| Método | Ruta | Autenticación | Descripción |
|---|---|---|---|
| `GET` | `/health` | Pública | Sonda de salud (`UP`, almacenamiento `PERSISTENT` o `MEMORY`). |
| `POST` | `/v1/learning/readings` | Bearer Token | Ingesta por lotes de telemetría de cultivos seudonimizada. |
| `GET` | `/v1/learning/status` | Bearer Token | Estado de modelos aprendidos, lecturas por especie y calidad DQS. |
| `POST` | `/v1/learning/train` | Bearer Token | Disparo bajo demanda de entrenamiento por especie o pendientes. |
| `DELETE` | `/v1/learning/crops/{id}` | Bearer Token | Olvido de lecturas de un cultivo eliminado (derecho al olvido). |
| `POST` | `/v1/learning/prune` | Bearer Token | Poda forzada de lecturas antiguas que superan retención o tope. |

---

### 4. Ejecución y Desarrollo Local

#### Despliegue en un comando

La imagen es pública: `ghcr.io/smartpottech/smartpot-trainer`. La plataforma completa, con este servicio incluido, se levanta con un solo comando y sin descargar el código:

```bash
curl -fsSL https://raw.githubusercontent.com/SmartPotTech/.github/main/docker/demo/compose.yaml | docker compose -f - up -d --wait
```

Abre http://localhost:5173 e ingresa con `demo@smartpot.app` / `SmartPot2026`. Se detiene con `docker compose -p smartpot-demo down` (con `-v` borra también los datos). Más detalle en la [demo local](https://github.com/SmartPotTech/.github/tree/main/docker/demo).

```bash
# Sincronizar dependencias con uv
uv sync

# Ejecutar suite de pruebas unitarias
uv run pytest -v

# Validar calidad de código
uv run ruff check .

# Iniciar servidor de desarrollo en puerto 8001
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

### 5. Contenedor Docker

La imagen de producción utiliza construcción multietapa basada en `python:3.13-slim-bookworm` y `uv`, ejecutándose bajo usuario no privilegiado (`1000:1000`):

```bash
docker pull ghcr.io/smartpottech/smartpot-trainer:latest
```

---

### 6. Estado del plan de rendimiento

| Pieza | Estado |
|-------|--------|
| Lectura por ADBC (Apache Arrow) desde TimescaleDB | Opcional, con el rol de solo lectura `smartpot_reader`, solo cultivos reales y clave seudonimizada; sin `DATABASE_URL` el entrenador usa su SQLite local, que alimenta la API |
| Pipeline en Polars | Filtrado, ventanas móviles y etiquetas autosupervisadas |
| Ruta rápida LoRA y ruta profunda campeón-retador | En el código |
| Exportación nativa firmada | JSON smartpot.learned-models/1, 16 controles, hashes BLAKE2b y firma Ed25519 del manifiesto (ONNX es un sidecar opcional) |
