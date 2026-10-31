"""Entrenamiento con datos reales, en el mismo orden de un proyecto de aprendizaje automático:

1. Calidad de datos y exclusión de atípicos.
2. Etiquetas autosupervisadas (qué pasó en la hora siguiente).
3. Comparación de modelos con validación cruzada temporal (TimeSeriesSplit), frente a una línea base.
4. Ajuste de hiperparámetros del mejor con GridSearchCV.
5. Campeón y retador: el modelo nuevo reemplaza al vigente solo si mejora en las lecturas más recientes.
6. No supervisado: estados de operación con K-Means (k por silueta) y lecturas atípicas con Isolation Forest.
"""

import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.knowledge.profiles import PARAMETERS, phrase
from app.learning.features import Dataset

log = logging.getLogger(__name__)

BASELINE = "Línea base"
HOLDOUT_SHARE = 0.2
MIN_POSITIVES = 10
MAX_CLUSTERS = 6
CLUSTER_SAMPLE = 4000
SEED = 7

CLASSIFIERS: dict[str, Any] = {
    BASELINE: DummyClassifier(strategy="most_frequent"),
    "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "K vecinos más cercanos": KNeighborsClassifier(n_neighbors=15),
    "Árbol de decisión": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=SEED
    ),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=120, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=1
    ),
    "Gradient boosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=SEED),
    "Red neuronal (MLP)": MLPClassifier(
        hidden_layer_sizes=(16,), alpha=0.01, max_iter=300, early_stopping=True, random_state=SEED
    ),
}
CLASSIFIER_GRIDS = {
    "Regresión logística": {"model__C": [0.1, 1.0, 10.0]},
    "K vecinos más cercanos": {"model__n_neighbors": [7, 15, 31]},
    "Árbol de decisión": {"model__max_depth": [4, 6, 10]},
    "Bosque aleatorio": {"model__max_depth": [6, 10], "model__n_estimators": [80, 160]},
    "Gradient boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 4]},
    "Red neuronal (MLP)": {"model__alpha": [0.001, 0.01]},
}
REGRESSORS: dict[str, Any] = {
    BASELINE: DummyRegressor(strategy="mean"),
    "Regresión ridge": Ridge(alpha=1.0),
    "K vecinos más cercanos": KNeighborsRegressor(n_neighbors=15),
    "Bosque aleatorio": RandomForestRegressor(
        n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=1
    ),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=150, random_state=SEED),
}
REGRESSOR_GRIDS = {
    "Regresión ridge": {"model__alpha": [0.1, 1.0, 10.0]},
    "K vecinos más cercanos": {"model__n_neighbors": [7, 15, 31]},
    "Bosque aleatorio": {"model__max_depth": [6, 10], "model__n_estimators": [80, 160]},
    "Gradient boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 4]},
}

TASKS = {
    "needs_water": "¿Se secará el sustrato en la próxima hora?",
    "overheat": "¿Pasará la temperatura del máximo en la próxima hora?",
    "moisture_1h": "Humedad del sustrato dentro de una hora",
}


@dataclass
class Candidate:
    model: str
    score: float
    std: float


@dataclass
class ModelCard:
    """Resultado de una tarea: el modelo vigente y cómo se eligió."""

    task: str
    label: str
    status: str
    metric: str
    reason: str | None = None
    model: str | None = None
    score: float | None = None
    std: float | None = None
    holdout: float | None = None
    baseline: float | None = None
    samples: int = 0
    positives: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    version: int = 0
    trained_at: float | None = None
    estimator: Any = None

    @property
    def usable(self) -> bool:
        return self.status == "TRAINED" and self.estimator is not None


@dataclass
class Cluster:
    id: int
    label: str
    description: str
    share: float
    center: list[float]


@dataclass
class StatesModel:
    k: int
    silhouette: float
    clusters: list[Cluster]
    scaler: StandardScaler
    kmeans: KMeans
    mapping: dict[int, int]

    def assign(self, positions: np.ndarray) -> Cluster:
        raw = int(self.kmeans.predict(self.scaler.transform(positions))[0])
        target = self.mapping.get(raw, raw)
        return next(c for c in self.clusters if c.id == target)


@dataclass
class AnomalyModel:
    samples: int
    forest: IsolationForest

    def score(self, positions: np.ndarray) -> tuple[float, bool]:
        decision = float(self.forest.decision_function(positions)[0])
        probability = 1.0 / (1.0 + np.exp(25.0 * decision))
        return round(float(probability), 3), bool(self.forest.predict(positions)[0] == -1)


def _pipeline(model: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", clone(model))])


@contextmanager
def _quiet():
    with warnings.catch_warnings():
        for category in (ConvergenceWarning, UndefinedMetricWarning, UserWarning, RuntimeWarning):
            warnings.simplefilter("ignore", category)
        yield


def _split(n: int) -> int:
    return max(1, int(n * (1 - HOLDOUT_SHARE)))


def train_classifier(
    task: str, x: np.ndarray, y: np.ndarray, min_samples: int, current: ModelCard | None
) -> ModelCard:
    card = ModelCard(
        task=task,
        label=TASKS[task],
        status="PENDING",
        metric="F1 macro",
        samples=int(x.shape[0]),
        positives=int(y.sum()),
    )
    negatives = card.samples - card.positives
    if card.samples < min_samples:
        card.reason = f"Faltan lecturas con una hora de futuro: hay {card.samples} de {min_samples}."
        return _keep(card, current)
    if card.positives < MIN_POSITIVES or negatives < MIN_POSITIVES:
        rare = "positivos" if card.positives < MIN_POSITIVES else "negativos"
        card.reason = (
            f"Aún no hay suficientes casos {rare} ({min(card.positives, negatives)} de {MIN_POSITIVES}): "
            "el modelo aprende cuando la situación ocurre."
        )
        return _keep(card, current)

    cut = _split(card.samples)
    x_train, y_train, x_hold, y_hold = x[:cut], y[:cut], x[cut:], y[cut:]
    folds = TimeSeriesSplit(n_splits=4)
    with _quiet():
        for name, model in CLASSIFIERS.items():
            scores = cross_val_score(
                _pipeline(model), x_train, y_train, cv=folds, scoring="f1_macro", error_score=np.nan
            )
            card.candidates.append(Candidate(name, _round(np.nanmean(scores)), _round(np.nanstd(scores))))
        best = _best(card.candidates, higher=True)
        tuned = _tune(best.model, CLASSIFIERS, CLASSIFIER_GRIDS, x_train, y_train, "f1_macro")
        baseline = _pipeline(CLASSIFIERS[BASELINE]).fit(x_train, y_train)
        card.baseline = _round(f1_score(y_hold, baseline.predict(x_hold), average="macro", zero_division=0))
        card.holdout = _round(f1_score(y_hold, tuned.predict(x_hold), average="macro", zero_division=0))
        rival = (
            _round(f1_score(y_hold, current.estimator.predict(x_hold), average="macro", zero_division=0))
            if current and current.usable
            else None
        )

    return _decide(card, best, tuned, x, y, current, rival, higher=True)


def train_regressor(x: np.ndarray, y: np.ndarray, min_samples: int, current: ModelCard | None) -> ModelCard:
    card = ModelCard(
        task="moisture_1h",
        label=TASKS["moisture_1h"],
        status="PENDING",
        metric="Error medio (%)",
        samples=int(x.shape[0]),
    )
    if card.samples < min_samples:
        card.reason = f"Faltan lecturas con su valor una hora después: hay {card.samples} de {min_samples}."
        return _keep(card, current)

    cut = _split(card.samples)
    x_train, y_train, x_hold, y_hold = x[:cut], y[:cut], x[cut:], y[cut:]
    folds = TimeSeriesSplit(n_splits=4)
    with _quiet():
        for name, model in REGRESSORS.items():
            scores = -cross_val_score(
                _pipeline(model), x_train, y_train, cv=folds, scoring="neg_mean_absolute_error", error_score=np.nan
            )
            card.candidates.append(Candidate(name, _round(np.nanmean(scores)), _round(np.nanstd(scores))))
        best = _best(card.candidates, higher=False)
        tuned = _tune(best.model, REGRESSORS, REGRESSOR_GRIDS, x_train, y_train, "neg_mean_absolute_error")
        card.baseline = _round(mean_absolute_error(y_hold, np.zeros_like(y_hold)))
        card.holdout = _round(mean_absolute_error(y_hold, tuned.predict(x_hold)))
        rival = (
            _round(mean_absolute_error(y_hold, current.estimator.predict(x_hold)))
            if current and current.usable
            else None
        )

    return _decide(card, best, tuned, x, y, current, rival, higher=False)


def _tune(name: str, catalog: dict, grids: dict, x: np.ndarray, y: np.ndarray, scoring: str) -> Pipeline:
    if name in grids and x.shape[0] >= 120:
        search = GridSearchCV(
            _pipeline(catalog[name]), grids[name], cv=TimeSeriesSplit(n_splits=3), scoring=scoring, error_score=np.nan
        )
        search.fit(x, y)
        return search.best_estimator_
    return _pipeline(catalog[name]).fit(x, y)


def _best(candidates: list[Candidate], higher: bool) -> Candidate:
    ranked = [c for c in candidates if c.model != BASELINE and not np.isnan(c.score)]
    if not ranked:
        return next(c for c in candidates if c.model == BASELINE)
    return max(ranked, key=lambda c: c.score) if higher else min(ranked, key=lambda c: c.score)


def _decide(
    card: ModelCard,
    best: Candidate,
    tuned: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    current: ModelCard | None,
    rival: float | None,
    higher: bool,
) -> ModelCard:
    beats_baseline = card.holdout > card.baseline if higher else card.holdout < card.baseline
    if not beats_baseline or best.model == BASELINE:
        card.reason = "Ningún modelo supera a la línea base en las lecturas más recientes; se sigue aprendiendo."
        return _keep(card, current)
    if rival is not None and (rival >= card.holdout if higher else rival <= card.holdout):
        current.candidates, current.samples = card.candidates, card.samples
        current.reason = (
            f"El modelo vigente sigue siendo mejor en las lecturas recientes ({rival} frente a {card.holdout})."
        )
        return current

    final = clone(tuned).fit(x, y)
    card.status = "TRAINED"
    card.model = best.model
    card.score, card.std = best.score, best.std
    card.params = {
        key.removeprefix("model__"): value
        for key, value in tuned.get_params().items()
        if key.startswith("model__") and key.removeprefix("model__") in _tuned_keys(best.model)
    }
    card.version = (current.version if current else 0) + 1
    card.trained_at = time.time()
    card.estimator = final
    card.reason = None
    return card


def _tuned_keys(name: str) -> set[str]:
    grid = CLASSIFIER_GRIDS.get(name, {}) | REGRESSOR_GRIDS.get(name, {})
    return {key.removeprefix("model__") for key in grid}


def _keep(card: ModelCard, current: ModelCard | None) -> ModelCard:
    if current and current.usable:
        current.reason = card.reason
        current.samples = card.samples
        if card.candidates:
            current.candidates = card.candidates
        return current
    return card


def _round(value: float) -> float:
    return round(float(value), 4)


def train_states(positions: np.ndarray, hours: np.ndarray) -> StatesModel | None:
    """Estados típicos de operación de la especie: K-Means con cantidad de grupos por silueta."""
    if positions.shape[0] < 60:
        return None
    rng = np.random.default_rng(SEED)
    sample = (
        positions
        if positions.shape[0] <= CLUSTER_SAMPLE
        else positions[rng.choice(positions.shape[0], CLUSTER_SAMPLE, replace=False)]
    )
    scaler = StandardScaler().fit(sample)
    scaled = scaler.transform(sample)
    best: tuple[float, KMeans] | None = None
    for k in range(2, min(MAX_CLUSTERS, len(np.unique(scaled, axis=0)) - 1) + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled)
        score = silhouette_score(scaled, kmeans.labels_, sample_size=min(2000, scaled.shape[0]), random_state=SEED)
        if best is None or score > best[0]:
            best = (float(score), kmeans)
    if best is None:
        return None
    silhouette, kmeans = best
    labels = kmeans.predict(scaler.transform(positions))
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    clusters: dict[str, Cluster] = {}
    mapping: dict[int, int] = {}
    for raw, center in enumerate(centers):
        members = labels == raw
        label, description = describe(center, hours[members])
        share = float(members.mean())
        if label in clusters:
            clusters[label].share = round(clusters[label].share + share, 3)
            mapping[raw] = clusters[label].id
            continue
        cluster = Cluster(
            id=len(clusters),
            label=label,
            description=description,
            share=round(share, 3),
            center=[round(float(v), 3) for v in center],
        )
        clusters[label] = cluster
        mapping[raw] = cluster.id
    ordered = sorted(clusters.values(), key=lambda c: c.share, reverse=True)
    return StatesModel(
        k=kmeans.n_clusters,
        silhouette=round(silhouette, 3),
        clusters=ordered,
        scaler=scaler,
        kmeans=kmeans,
        mapping=mapping,
    )


def describe(center: np.ndarray, hours: np.ndarray) -> tuple[str, str]:
    """Nombre corto del estado a partir de su centro y su hora típica."""
    deviations = []
    for parameter, value in zip(PARAMETERS, center, strict=True):
        if value < -0.1:
            deviations.append((-value, phrase(parameter, "LOW")))
        elif value > 1.1:
            deviations.append((value - 1, phrase(parameter, "HIGH")))
    deviations.sort(reverse=True)
    moment = _moment(hours)
    if not deviations:
        return f"Condiciones ideales {moment}".strip(), "Todas las variables quedan dentro de su rango ideal."
    text = " y ".join(name for _, name in deviations[:2])
    label = f"{text[0].upper()}{text[1:]} {moment}".strip()
    return label, "Lo más alejado del ideal: " + ", ".join(name for _, name in deviations) + "."


def _moment(hours: np.ndarray) -> str:
    hours = hours[~np.isnan(hours)]
    if hours.size == 0:
        return ""
    angle = np.arctan2(np.sin(hours / 24 * 2 * np.pi).mean(), np.cos(hours / 24 * 2 * np.pi).mean())
    typical = (angle / (2 * np.pi) * 24) % 24
    if 6 <= typical < 12:
        return "en la mañana"
    if 12 <= typical < 18:
        return "en la tarde"
    return "en la noche"


def train_anomaly(positions: np.ndarray) -> AnomalyModel | None:
    if positions.shape[0] < 100:
        return None
    forest = IsolationForest(n_estimators=150, contamination=0.01, random_state=SEED).fit(positions)
    return AnomalyModel(samples=int(positions.shape[0]), forest=forest)


def train_all(dataset: Dataset, min_samples: int, current: dict[str, ModelCard]) -> dict:
    """Entrena las tres tareas supervisadas y los dos modelos no supervisados de una especie."""
    water = dataset.water_mask
    heat = dataset.heat_mask
    moisture = ~np.isnan(dataset.moisture_delta)
    complete = ~np.isnan(dataset.states).any(axis=1)
    return {
        "needs_water": train_classifier(
            "needs_water", dataset.x[water], dataset.water[water], min_samples, current.get("needs_water")
        ),
        "overheat": train_classifier(
            "overheat", dataset.x[heat], dataset.heat[heat], min_samples, current.get("overheat")
        ),
        "moisture_1h": train_regressor(
            dataset.x[moisture], dataset.moisture_delta[moisture], min_samples, current.get("moisture_1h")
        ),
        "states": train_states(dataset.states[complete], _hours_from_features(dataset.x[complete])),
        "anomaly": train_anomaly(dataset.states[complete]),
    }


def _hours_from_features(x: np.ndarray) -> np.ndarray:
    sine, cosine = x[:, len(PARAMETERS)], x[:, len(PARAMETERS) + 1]
    return (np.arctan2(sine, cosine) / (2 * np.pi) * 24) % 24
