"""Re-diagnostico de B sobre datos reales (N8e, ADR 0012, 5).

Corre sobre **las politicas de B que guardo la corrida sellada** (2 semillas x 11
folds = 22), en las barras de test de su fold. No reentrena nada ni cambia el
veredicto de K3: es descriptivo.

Para cada politica se fija, de a un grupo, la observacion en su media del episodio de
test (``diagnosis.diagnose_policy``, los cinco grupos) y se mide cuanto cambia la
accion. Ademas, para el grupo ``noticias``, se mide cuanto cambia la accion al
**reemplazar** las noticias reales por las del placebo (``t - 364 d``).

Regla de lectura. El ADR 0012 fija solo una mitad: si fijar las noticias cambia la
accion en menos de ``NEWS_USE_MIN_CHANGE`` (0.02) de media absoluta, "B no usa las
noticias". Para afirmar lo contrario el cambio se lee contra el piso de ruido de la
puerta 5b (``results/puerta/REPORTE.md``), declarado aqui antes de correr:

- mediana <= ``NOISE_FLOOR_MEDIAN``: no se distingue de lo que producen siete
  columnas de ruido puro.
- entre ese piso y ``NOISE_FLOOR_MAX`` (la politica mas reactiva al ruido): indicio
  debil, no concluyente.
- sobre ``NOISE_FLOOR_MAX``: mas que cualquier politica ante ruido puro.

ASSUMPTION: el piso se midio sobre fixtures sinteticos (5b), con otra distribucion de
las entradas que la real; es la referencia disponible, no una equivalencia. El
cambio de accion mide **sensibilidad**, no utilidad: que B reaccione a las noticias
no dice que le sirvan. Las politicas del placebo no se guardaron, asi que la
comparacion con el placebo es el reemplazo de entradas, no un segundo agente.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agents import calibration as cal
from agents import diagnosis as dg
from agents.experiment import jsonable
from agents.gate_diagnosis import NEWS_GROUP, NEWS_USE_MIN_CHANGE, Loader
from agents.ppo import SB3Policy
from agents.runner import build_env, run_policy, scaler_for
from agents.study_run import (
    ARM_B,
    ARM_PLACEBO,
    StudyRunConfig,
    arm_observation,
    arm_series,
    load_news_table,
    policy_path,
)
from agents.study_run import Unit as StudyUnit
from agents.walkforward import SealError, folds, load_dataset, training_span
from envs.observation import ObservationBuilder
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig
from eval.distribution import MIN_SEEDS, summarize
from sim.sizing import TargetWeightSizer

#: Puerta 5b (N6c): cambio de accion medio al fijar 7 columnas de puro ruido.
NOISE_FLOOR_MEDIAN = 0.0252
NOISE_FLOOR_MAX = 0.100
#: Dispersion de la accion bajo la cual la politica es constante.
CONSTANT_STD = 1e-9


def _default_loader(path: Path) -> Any:  # pragma: no cover - necesita torch
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


def read_news_change(median: float) -> str:
    """Lectura del cambio mediano al fijar las noticias (ver el docstring)."""
    if median < NEWS_USE_MIN_CHANGE:
        return f"menos de {NEWS_USE_MIN_CHANGE}: B no usa las noticias (regla del ADR)"
    if median <= NOISE_FLOOR_MEDIAN:
        return "no se distingue del piso de ruido de la puerta 5b"
    if median <= NOISE_FLOOR_MAX:
        return (
            "sobre el piso mediano de ruido pero dentro del rango de una politica "
            "que reacciona al ruido: indicio debil, no concluyente"
        )
    return "mas que cualquier politica ante ruido puro: B reacciona a las noticias"


def diagnose_unit(
    series: Any,
    placebo: Any,
    window: Any,
    unit: StudyUnit,
    config: StudyRunConfig,
    *,
    policy_dir: Path,
    loader: Loader = _default_loader,
) -> dict[str, Any]:
    """Diagnostica la politica guardada de (fold, semilla) sobre su tramo de test."""
    archivo = Path(f"{policy_path(policy_dir, unit)}.zip")
    if not archivo.is_file():
        raise SealError(f"falta la politica guardada {archivo}")
    spec = arm_observation(ARM_B, config)
    escalador = scaler_for(series.slice(*training_span(window)), spec)
    env_config = EnvConfig(
        observation=spec, sizer=TargetWeightSizer(safety=config.walk.safety)
    )
    costos = config.walk.sim_config()
    inicio = window.test[0] - spec.warmup

    def entorno(fuente: Any) -> Any:
        return build_env(
            fuente.slice(inicio, window.test[1]),
            costos,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=config.walk.reward_scale),
        )

    politica = SB3Policy(loader(archivo), recurrent=False)
    grupos = dg.feature_groups(ObservationBuilder(spec).names)
    diagnostico = dg.diagnose_policy(politica, lambda: entorno(series), grupos)

    # Reemplazo en lazo abierto: las observaciones de B con las columnas de noticias
    # del placebo en las mismas barras. El resto de las columnas no se toca.
    grabadora = dg._Recorder(politica)
    base = run_policy(entorno(series), grabadora)
    obs_b = np.asarray(grabadora.observations, dtype=np.float64)
    otra = dg._Recorder(politica)
    run_policy(entorno(placebo), otra)
    obs_p = np.asarray(otra.observations, dtype=np.float64)
    n = min(len(obs_b), len(obs_p))
    cols = grupos[NEWS_GROUP]
    cambiado = obs_b[:n].copy()
    cambiado[:, cols] = obs_p[:n, cols]
    acciones = np.asarray(base.actions, dtype=np.float64)[:n]
    nuevas = np.array([politica.act(f) for f in cambiado], dtype=np.float64)
    diagnostico["placebo_swap_mean_abs_change"] = float(
        np.abs(nuevas - acciones).mean()
    )
    return {
        "fold": unit.fold,
        "seed": unit.seed,
        "n_steps": diagnostico["n_steps"],
        "diagnosis": diagnostico,
    }


def summarize_policies(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribucion entre politicas por grupo, por fold, y la lectura."""
    ids = list(range(len(registros)))

    def dist(nombre: str, valores: Sequence[float]) -> dict[str, object]:
        # Con menos de 10 politicas (solo en tests) el resumen queda marcado.
        return summarize(
            nombre, ids, list(valores), allow_fewer_seeds=len(ids) < MIN_SEEDS
        ).describe()

    def cambio(r: dict[str, Any], grupo: str) -> float:
        return float(r["diagnosis"]["groups"][grupo]["open_loop_mean_abs_change"])

    grupos = registros[0]["diagnosis"]["groups"]
    mediana = float(np.median([cambio(r, NEWS_GROUP) for r in registros]))
    por_fold: dict[int, list[float]] = {}
    for r in registros:
        por_fold.setdefault(r["fold"], []).append(cambio(r, NEWS_GROUP))
    swap = [r["diagnosis"]["placebo_swap_mean_abs_change"] for r in registros]
    # Una politica de accion constante ignora todo por construccion: se cuenta aparte.
    vivas = [r for r in registros if r["diagnosis"]["action_std"] > CONSTANT_STD]
    return {
        "n_constant_policies": len(registros) - len(vivas),
        "news_median_abs_change_non_constant": (
            float(np.median([cambio(r, NEWS_GROUP) for r in vivas])) if vivas else None
        ),
        "n_policies": len(registros),
        "baseline_action_std": dist(
            "baseline", [r["diagnosis"]["action_std"] for r in registros]
        ),
        "groups": {
            g: {
                "open_loop_mean_abs_change": dist(g, [cambio(r, g) for r in registros]),
                "closed_loop_action_std": dist(
                    g,
                    [
                        r["diagnosis"]["groups"][g]["closed_loop_action_std"]
                        for r in registros
                    ],
                ),
            }
            for g in grupos
        },
        "news_median_abs_change": mediana,
        "news_change_by_fold": {
            f: float(np.median(v)) for f, v in sorted(por_fold.items())
        },
        "placebo_swap_mean_abs_change": dist("placebo_swap", swap),
        "noise_floor": {"median": NOISE_FLOOR_MEDIAN, "max": NOISE_FLOOR_MAX},
        "uses_news_by_adr_rule": mediana >= NEWS_USE_MIN_CHANGE,
        "reading": (
            f"fijar las noticias cambia la accion {mediana:.4f} de media absoluta "
            f"(mediana entre {len(registros)} politicas): {read_news_change(mediana)}"
        ),
    }


def run(
    root: Path,
    policy_dir: Path,
    out: Path,
    *,
    config: StudyRunConfig | None = None,
    loader: Loader = _default_loader,
    progress: Callable[[str], None] = lambda _m: None,
) -> dict[str, Any]:
    """Diagnostica las politicas guardadas de B y escribe ``rediagnosis.json``."""
    config = config or StudyRunConfig()
    serie, _marcas, _sha = load_dataset(root)
    tabla, _news_sha = load_news_table(root, config)
    b = arm_series(ARM_B, serie, tabla, config)
    p = arm_series(ARM_PLACEBO, serie, tabla, config)
    registros = []
    for ventana in folds(len(serie), config.walk):
        for semilla in config.saved_policy_seeds:
            u = StudyUnit(ARM_B, ventana.index, semilla)
            registros.append(
                diagnose_unit(
                    b, p, ventana, u, config, policy_dir=policy_dir, loader=loader
                )
            )
            progress(f"fold {ventana.index} semilla {semilla}")
    resumen = summarize_policies(registros)
    resumen["policies"] = [
        {
            "fold": r["fold"],
            "seed": r["seed"],
            "action_std": r["diagnosis"]["action_std"],
            "news_change": r["diagnosis"]["groups"][NEWS_GROUP][
                "open_loop_mean_abs_change"
            ],
            "placebo_swap": r["diagnosis"]["placebo_swap_mean_abs_change"],
        }
        for r in registros
    ]
    cal.write_atomic(out / "rediagnosis.json", jsonable(resumen))
    return resumen
