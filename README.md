# SmartPot Trainer (Plano de Entrenamiento y Aprendizaje Continuo)

Microservicio desacoplado de entrenamiento y aprendizaje continuo de SmartPot. Responsable de la ingesta por lotes de telemetría de sensores, evaluación de calidad de datos con Polars, torneos de modelos con validación cruzada temporal (`TimeSeriesSplit`), ajuste de hiperparámetros con `GridSearchCV`, adaptación rápida en memoria mediante LoRA (Low-Rank Adaptation) y persistencia atómica de modelos (`joblib`) en el volumen compartido.

---

## 1. Arquitectura y Desacoplamiento de Planos

Para garantizar que el servicio de inferencia (`SmartPot-AI`) mantenga latencias sub-milisegundo (< 0.05 ms) libres de fluctuaciones (*jitter*) causadas por la contención de CPU y memoria de entrenamientos masivos, la arquitectura se divide en dos planos complementarios:

```
                  [ SmartPot-API-Go / Clientes ]
                                │
          ┌─────────────────────┴─────────────────────┐
          │ (Inferencia en caliente)                  │ (Ingesta y reentrenamiento)
          ▼                                           ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│       SmartPot-AI         │               │     SmartPot-Trainer      │
│  (Serving Plane / :8000)  │               │ (Training Plane / :8001)  │
│                           │               │                           │
│  • Cython C++20 Core      │               │  • Polars LazyFrame DQS   │
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

### Contrato del Volumen Compartido (`/data`)
- **`/data/learning.db`**: Base de datos SQLite gestionada con índices temporales y seudonimización criptográfica SHA-256 de los identificadores de cultivo (`crop_key`), garantizando privacidad y cumplimiento de retención de datos.
- **`/data/models/{crop_type}.joblib`**: Artefactos de modelos serializados con metadatos de versión de Scikit-Learn, tarjetas de modelo (`ModelCard`), agrupamiento K-Means (`StatesModel`), bosque de aislamiento (`AnomalyModel`) y adaptadores LoRA (`LowRankCropAdapter`).

---

## 2. Capacidades de Aprendizaje y Modelos

1. **Evaluación de Calidad DQS (Data Quality Score)**:
   - Procesamiento en paralelo con `polars.LazyFrame`.
   - Validación física estricta por variable agronómica.
   - Detección de atípicos estadísticos por Rango Intercuartílico (IQR con $k=3.0$).

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

## 3. Endpoints REST

| Método | Ruta | Autenticación | Descripción |
|---|---|---|---|
| `GET` | `/health` | Pública | Sonda de salud (`UP`, almacenamiento `PERSISTENT` o `MEMORY`). |
| `POST` | `/v1/learning/readings` | Bearer Token | Ingesta por lotes de telemetría de cultivos seudonimizada. |
| `GET` | `/v1/learning/status` | Bearer Token | Estado de modelos aprendidos, lecturas por especie y calidad DQS. |
| `POST` | `/v1/learning/train` | Bearer Token | Disparo bajo demanda de entrenamiento por especie o pendientes. |
| `DELETE` | `/v1/learning/crops/{id}` | Bearer Token | Olvido de lecturas de un cultivo eliminado (derecho al olvido). |
| `POST` | `/v1/learning/prune` | Bearer Token | Poda forzada de lecturas antiguas que superan retención o tope. |

---

## 4. Ejecución y Desarrollo Local

### Despliegue en un comando

La imagen es pública: `ghcr.io/smartpottech/smartpot-trainer`. La plataforma completa, con este servicio incluido, se
levanta con un solo comando y sin descargar el código:

```bash
curl -fsSL https://raw.githubusercontent.com/SmartPotTech/.github/main/docker/demo-beta/compose.yaml | docker compose -f - up -d --wait
```

Abre http://localhost:5173 e ingresa con `demo@smartpot.app` / `SmartPot2026`. Se detiene con
`docker compose -p smartpot-demo-beta down` (con `-v` borra también los datos). Más detalle en la
[demo beta](https://github.com/SmartPotTech/.github/tree/main/docker/demo-beta).


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

## 5. Contenedor Docker

La imagen de producción utiliza construcción multietapa basada en `python:3.13-slim-bookworm` y `uv`, ejecutándose bajo usuario no privilegiado (`1000:1000`):

```bash
docker pull ghcr.io/smartpottech/smartpot-trainer:latest
```

---

## 6. Estado del plan de rendimiento

| Pieza | Estado |
|-------|--------|
| Lectura por ADBC (Apache Arrow) desde TimescaleDB | Opcional, con el rol de solo lectura `smartpot_reader`, solo cultivos reales y clave seudonimizada; sin `DATABASE_URL` el entrenador usa su SQLite local, que alimenta la API |
| Pipeline en Polars | Filtrado, ventanas móviles y etiquetas autosupervisadas |
| Ruta rápida LoRA y ruta profunda campeón-retador | En el código |
| Exportación ONNX firmada | `skl2onnx`, hash BLAKE2b y firma Ed25519 del manifiesto; la IA solo verifica la firma si se define `TRUSTED_MODEL_PUBLIC_KEY_HEX` y todavía carga los modelos con `joblib` |
| Recarga en caliente de la IA por Valkey | Pendiente |

