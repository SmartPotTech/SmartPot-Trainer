# SmartPot-Trainer: native learned-model supply chain

[English](#english) | [Español](#español)

<a id="english"></a>

## English

### Native package

`export_crop_models` writes `{SPECIES}_inference.json` with the `smartpot.learned-models/1` schema:

- `cards`: the three champion pipelines (`needs_water`, `overheat`, `moisture_1h`) serialized as scaling plus native parameters (`logistic`, `linear`, `mlp`, `forest`, `hist_gradient_boosting`, `knn`, `constant`).
- `states`: the K-Means scaler, scaled centers, raw-to-state mapping and cluster metadata.
- `anomaly`: the Isolation Forest trees with `left`, `right`, `feature`, `threshold`, `value`, `denominator` and `offset`.
- `lora`: exact `Adapter` fields (`d_in`, `k_out`, `rank`, `alpha`, `w0`, `b0`, `a`, `b`, `bias`, `updates_count`).
- `checks`: exactly 16 deterministic 10-feature vectors evaluated with scikit-learn (`needs_water`, `overheat`, `moisture_1h` delta, mapped state, raw anomaly decision).

The manifest declares the BLAKE2b-256 hash of the bundle and is signed with Ed25519 over Python's canonical JSON (`sort_keys=True`, compact separators, ASCII escapes, no `signature_hex`).

### Security and robustness

- The private key lives in `MODEL_SIGNING_KEY_PATH` (default `/data/keys/ed25519_private.pem`), outside `/data/models`, with mode `0600`. A corrupt existing key aborts startup; no ephemeral fallback.
- `verify_manifest(path, expected_public_key_hex)` pins the trust anchor. Persisted joblib files are never deserialized without a verified signature and hash; without `TRUSTED_MODEL_PUBLIC_KEY_HEX` the service retrains.
- `train_classifier` returns `PENDING` when the window or a fold has one class, instead of raising HTTP 500.
- Data quality adds physically impossible per-minute jumps in temperature, humidity and substrate, plus an Isolation Forest secondary filter before building the training set.
- `DELETE /v1/learning/crops/{crop_id}` requires a 24-character lowercase ObjectId hex; non-ASCII Bearer headers return 401.

<a id="español"></a>

## Español

### Paquete nativo

`export_crop_models` escribe `{SPECIES}_inference.json` con el esquema `smartpot.learned-models/1`:

- `cards`: los tres pipelines campeones (`needs_water`, `overheat`, `moisture_1h`) serializados como escalado más parámetros nativos (`logistic`, `linear`, `mlp`, `forest`, `hist_gradient_boosting`, `knn`, `constant`).
- `states`: escalador de K-Means, centros escalados, mapeo de grupos a estados y metadatos de cada clúster.
- `anomaly`: árboles del Isolation Forest con `left`, `right`, `feature`, `threshold`, `value`, `denominator` y `offset`.
- `lora`: campos exactos de `Adapter` (`d_in`, `k_out`, `rank`, `alpha`, `w0`, `b0`, `a`, `b`, `bias`, `updates_count`).
- `checks`: exactamente 16 vectores deterministas de 10 features evaluados con scikit-learn (`needs_water`, `overheat`, delta de `moisture_1h`, estado mapeado y decisión cruda de anomalía).

El manifiesto declara el hash BLAKE2b-256 del paquete y se firma con Ed25519 sobre el JSON canónico de Python (`sort_keys=True`, separadores compactos, escapes ASCII, sin `signature_hex`).

### Seguridad y robustez

- La clave privada vive en `MODEL_SIGNING_KEY_PATH` (por defecto `/data/keys/ed25519_private.pem`), fuera de `/data/models`, con modo `0600`. Una clave existente corrupta aborta el arranque; no hay clave efímera de respaldo.
- `verify_manifest(path, expected_public_key_hex)` fija el ancla de confianza. Los joblib persistidos nunca se deserializan sin firma y hash verificados; sin `TRUSTED_MODEL_PUBLIC_KEY_HEX` el servicio reentrena.
- `train_classifier` devuelve `PENDING` cuando la ventana o un fold tiene una sola clase, en lugar de provocar HTTP 500.
- La calidad de datos añade saltos físicamente imposibles por minuto en temperatura, humedad y sustrato, más un filtro secundario Isolation Forest antes de construir el conjunto de entrenamiento.
- `DELETE /v1/learning/crops/{crop_id}` exige un ObjectId hexadecimal de 24 caracteres en minúsculas; una cabecera Bearer no ASCII devuelve 401.
