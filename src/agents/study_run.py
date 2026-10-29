"""Corrida sellada del estudio A vs B vs placebo sobre BTCUSDT diario. ADR 0012.

Ejecuta, no juzga: el criterio (seccion 6) es del ensamblado. Tres brazos (A, B y
placebo) sobre la geometria del ADR 0006, con la configuracion de ``agents.study``
y las siete columnas de noticias de ``features.news``. A no ve las columnas; B ve
las de la barra ``t``; el placebo, las de ``t - 364 d``.

El test se toca una vez. ``SEAL.json`` se escribe de forma exclusiva **antes** del
primer entrenamiento (seccion 9.3) y la corrida se niega a empezar si el ADR en
disco no es el sellado o si la latencia en vivo medida supera la usada. La unidad
es **un** (brazo, fold, semilla): se persiste al terminar (append + fsync) y
``resume`` solo salta las ya escritas, sin reentrenar ninguna.

ASSUMPTION: el p99 de la latencia en vivo lo mide Kevin con los datos del grabador
(viven en la VM) y entra como argumento; este modulo no lee la VM.
ASSUMPTION: ``run_commit`` entra al sello, asi que reanudar con otro commit falla:
un arreglo despues del primer toque es otra corrida (seccion 9.4).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from agents.policy import Policy
from agents.ppo import PPOConfig, SB3Policy
from agents.protocol import STUDY_SEEDS
from agents.study import STUDY_OBSERVATION, STUDY_PPO
from agents.walkforward import (
    FoldSeedRecord,
    PolicyFactory,
    SealError,
    WalkForwardConfig,
    acquire_seal,
    evaluate_fold_seed,
    folds,
    load_dataset,
)
from data.manifest import read_manifest
from data.news_aggregate import read_aggregates
from data.schema import BarSeries
from envs.observation import ObservationSpec
from eval.walkforward import Window
from features.finbert import MODEL_FILES, MODEL_REVISION
from features.finbert_dataset import attach_finbert, read_finbert_aggregates
from features.news import (
    NEWS_EXOG,
    NEWS_LATENCY,
    PLACEBO_LAG,
    NewsFeatureTable,
    build_news_features,
    with_news,
)

ADR_PATH = Path("docs/adr/0012-pre-registro-del-agente-b.md")
# Sello del ADR 0012 tal como esta en main. Cambiar el ADR, aunque sea una coma,
# obliga a cambiar esta constante, y eso queda en el diff.
ADR_SHA256 = "5bd0b4079e922f3bb77c63c560038e514cd44271886211b283be93133fed9c40"
NEWS_MANIFEST = Path("datasets/gdelt/gdelt-batches.csv.gz.manifest.json")
FINBERT_MANIFEST = Path("datasets/gdelt/finbert-batches.csv.gz.manifest.json")

ARM_A, ARM_B, ARM_PLACEBO = "A", "B", "placebo"
ARMS: tuple[str, ...] = (ARM_A, ARM_B, ARM_PLACEBO)


@dataclass(frozen=True)
class StudyRunConfig:
    """Todo lo que no fija el ADR 0012 y cambia el resultado. Entra al sello.

    La geometria, los costos y el capital son los del ADR 0006 (``WalkForwardConfig``);
    el agente es ``C*`` (``agents.study``), que ``WalkForwardConfig`` no conoce.
    """

    walk: WalkForwardConfig = field(
        default_factory=lambda: WalkForwardConfig(
            total_timesteps=STUDY_PPO.total_timesteps,
            reward_scale=STUDY_PPO.reward_scale,
            seeds=STUDY_SEEDS,
        )
    )
    ppo: PPOConfig = STUDY_PPO
    observation: ObservationSpec = STUDY_OBSERVATION
    news_latency_min: int = int(NEWS_LATENCY / np.timedelta64(1, "m"))
    placebo_lag_days: int = int(PLACEBO_LAG / np.timedelta64(1, "D"))
    # Politicas de B que se guardan para el re-diagnostico sobre datos reales
    # (ADR 0012 §5): las de las dos primeras semillas, en cada fold.
    saved_policy_seeds: tuple[int, ...] = STUDY_SEEDS[:2]

    def describe(self) -> dict[str, object]:
        walk = self.walk.describe()
        # ``WalkForwardConfig`` describe un PPO por defecto que aqui no se usa.
        walk.pop("ppo")
        return {
            "walk": walk,
            "ppo": self.ppo.describe(),
            "observation": self.observation.describe(),
            "news_exog": [vars(c) for c in NEWS_EXOG],
            "news_latency_min": self.news_latency_min,
            "placebo_lag_days": self.placebo_lag_days,
            "saved_policy_seeds": list(self.saved_policy_seeds),
            "arms": list(ARMS),
        }

    def digest(self) -> str:
        crudo = json.dumps(self.describe(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(crudo).hexdigest()


@dataclass(frozen=True)
class Unit:
    arm: str
    fold: int
    seed: int


def units(config: StudyRunConfig, n_bars: int) -> tuple[Unit, ...]:
    """Fold por fold: una corrida cortada deja los tres brazos parejos."""
    return tuple(
        Unit(arm, ventana.index, seed)
        for ventana in folds(n_bars, config.walk)
        for arm in ARMS
        for seed in config.walk.seeds
    )


def arm_observation(arm: str, config: StudyRunConfig) -> ObservationSpec:
    if arm == ARM_A:
        return config.observation
    return replace(config.observation, exog=NEWS_EXOG)


def arm_series(
    arm: str, series: BarSeries, table: NewsFeatureTable, config: StudyRunConfig
) -> BarSeries:
    """A: la serie tal cual. B: columnas de ``t``. Placebo: las de ``t - 364 d``."""
    if arm == ARM_A:
        return series
    if arm == ARM_B:
        return with_news(series, table)
    if arm == ARM_PLACEBO:
        return with_news(series, table, np.timedelta64(config.placebo_lag_days, "D"))
    raise SealError(f"brazo desconocido: {arm!r}")


def load_news_table(root: Path, config: StudyRunConfig) -> tuple[NewsFeatureTable, str]:
    """Tabla de features con la latencia de la configuracion, y su SHA256 de datos."""
    gdelt = read_aggregates(root / NEWS_MANIFEST)
    frame = attach_finbert(gdelt, read_finbert_aggregates(root / FINBERT_MANIFEST))
    tabla = build_news_features(frame, np.timedelta64(config.news_latency_min, "m"))
    return tabla, read_manifest(root / NEWS_MANIFEST).sha256


def verify_adr(root: Path, expected: str = ADR_SHA256) -> str:
    real = hashlib.sha256((root / ADR_PATH).read_bytes()).hexdigest()
    if real != expected:
        raise SealError(
            f"el ADR 0012 en disco tiene SHA256 {real} y el sellado es {expected}: "
            "el criterio cambio despues de sellarse"
        )
    return real


def seal_payload(
    config: StudyRunConfig,
    *,
    adr_sha: str,
    dataset_sha: str,
    news_sha: str,
    finbert_sha: str,
    run_commit: str,
    live_latency_p99_min: float,
) -> dict[str, Any]:
    """Seccion 9.3: lo que se fija antes del primer entrenamiento."""
    return {
        "adr_sha256": adr_sha,
        "dataset_sha256": dataset_sha,
        "news_dataset_sha256": news_sha,
        "finbert_batches_sha256": finbert_sha,
        "finbert_revision": MODEL_REVISION,
        "finbert_files": dict(MODEL_FILES),
        "config": config.digest(),
        "commit": run_commit,
        "latency_min": config.news_latency_min,
        "live_latency_p99_min": live_latency_p99_min,
    }


def policy_path(out_dir: Path, unit: Unit) -> Path:
    """Donde queda la politica de una unidad de B (SB3 le agrega ``.zip``)."""
    return out_dir / "policies" / f"{unit.arm}-fold{unit.fold}-seed{unit.seed}"


def _leer(path: Path) -> list[tuple[str, FoldSeedRecord]]:
    if not path.exists():
        return []
    filas = []
    for linea in path.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            datos = json.loads(linea)
            filas.append((datos.pop("arm"), FoldSeedRecord.from_dict(datos)))
    return filas


def read_records(path: Path) -> list[tuple[str, FoldSeedRecord]]:
    """Los registros guardados como (brazo, registro), en orden de escritura."""
    return _leer(path)


def run_study(
    root: Path,
    out_dir: Path,
    policy_factory: Callable[[ObservationSpec], PolicyFactory],
    *,
    config: StudyRunConfig | None = None,
    run_commit: str,
    live_latency_p99_min: float,
    resume: bool = False,
    expected_adr_sha: str = ADR_SHA256,
) -> int:
    """Corre todas las unidades que faltan y devuelve cuantas entreno.

    ``policy_factory(observation)`` devuelve la fabrica de politicas de un brazo,
    que recibe ``(make_env, seed)`` como en ``agents.walkforward``.
    """
    config = config or StudyRunConfig()
    adr_sha = verify_adr(root, expected_adr_sha)
    if live_latency_p99_min > config.news_latency_min:
        raise SealError(
            f"el p99 de latencia en vivo ({live_latency_p99_min} min) supera la "
            f"usada ({config.news_latency_min} min): sube la latencia al siguiente "
            "multiplo de 15 antes de correr (ADR 0012, 2.2), nunca despues"
        )
    serie, marcas, dataset_sha = load_dataset(root)
    tabla, news_sha = load_news_table(root, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_seal(
        out_dir / "SEAL.json",
        seal_payload(
            config,
            adr_sha=adr_sha,
            dataset_sha=dataset_sha,
            news_sha=news_sha,
            finbert_sha=read_manifest(root / FINBERT_MANIFEST).sha256,
            run_commit=run_commit,
            live_latency_p99_min=live_latency_p99_min,
        ),
        resume=resume,
    )
    registros_path = out_dir / "records.jsonl"
    listos = {(a, r.fold, r.seed) for a, r in _leer(registros_path)}
    ventanas: dict[int, Window] = {v.index: v for v in folds(len(serie), config.walk)}
    por_brazo = {a: arm_series(a, serie, tabla, config) for a in ARMS}
    fabricas = {a: policy_factory(arm_observation(a, config)) for a in ARMS}
    entrenadas = 0
    for u in units(config, len(serie)):
        if (u.arm, u.fold, u.seed) in listos:
            continue

        def guardar(politica: Policy, unidad: Unit = u) -> None:
            if (
                unidad.arm == ARM_B
                and unidad.seed in config.saved_policy_seeds
                and isinstance(politica, SB3Policy)
            ):
                destino = policy_path(out_dir, unidad)
                destino.parent.mkdir(parents=True, exist_ok=True)
                politica.save(str(destino))

        registro = evaluate_fold_seed(
            por_brazo[u.arm],
            marcas,
            ventanas[u.fold],
            u.seed,
            config.walk,
            fabricas[u.arm],
            arm_observation(u.arm, config),
            guardar,
        )
        with registros_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"arm": u.arm, **registro.to_dict()}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        entrenadas += 1
    return entrenadas
