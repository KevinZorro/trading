"""Re-diagnostico de la puerta con las noticias como grupo propio (ADR 0012, 5).

Corre sobre **las politicas de B que guardo la puerta** (``gate.SAVED_POLICY_SEEDS``,
10 caminos por nivel), no sobre otras reentrenadas. Para cada una se fija, de a un
grupo, la observacion en su media del episodio (``diagnosis.diagnose_policy``) y se
mide cuanto cambia la accion y, en el 5a, el ``capture``. Los grupos son los
cinco de siempre mas ``noticias``.

Regla de lectura, fijada en el ADR antes de correr: si fijar las noticias cambia la
accion en menos de ``NEWS_USE_MIN_CHANGE`` (0.02) de media absoluta, el reporte
dice "B no usa las noticias". Es descriptivo y no decide el veredicto de la puerta.
En el 5a ademas se informa cuantas politicas pierden ``capture`` al fijar las
noticias; el ADR dice "tiene que hundirlo" sin un numero, y aqui no se inventa uno.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agents import calibration as cal
from agents import diagnosis as dg
from agents import gate
from agents.experiment import jsonable
from agents.ppo import SB3Policy
from agents.protocol import _sim_config
from agents.runner import build_env, scaler_for
from agents.study import STUDY_PPO
from data.fixtures import DEFAULT_N_BARS, LEVELS
from envs.observation import ObservationBuilder
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig
from eval.distribution import summarize
from sim.sizing import TargetWeightSizer

#: ADR 0012, 5: debajo de esto "B no usa las noticias".
NEWS_USE_MIN_CHANGE = 0.02
NEWS_GROUP = "noticias"
B_ARMS = tuple(a for a in gate.ARMS if not a.blind)

Loader = Callable[[Path], Any]


def _default_loader(path: Path) -> Any:  # pragma: no cover - necesita torch
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


def saved_units(out: Path, point: str) -> tuple[cal.Unit, ...]:
    """Las unidades de B cuya politica guardo la puerta, en orden fijo."""
    return tuple(
        u for u in gate.arm_units(point) if u.agent_seed in gate.SAVED_POLICY_SEEDS
    )


def diagnose_unit(
    out: Path, unit: cal.Unit, *, loader: Loader = _default_loader
) -> dict[str, Any]:
    """Diagnostica la politica guardada de una unidad sobre su tramo de validacion."""
    a = gate.arm(unit.point)
    archivo = Path(f"{gate.policy_path(out / 'policies', unit)}.zip")
    if not archivo.is_file():
        raise cal.CalibrationError(f"falta la politica guardada {archivo}")
    fixture = LEVELS[a.level](DEFAULT_N_BARS, seed=unit.path_seed)
    train, validacion, _ = fixture.split()
    spec = gate.observation_for(a)
    escalador = scaler_for(train.series, spec)
    config_sim = _sim_config(fixture, 100_000.0, 0.0)
    env_config = EnvConfig(observation=spec, sizer=TargetWeightSizer(safety=0.98))
    techos = validacion.ceilings(
        safety=0.98, cash_rate=0.0, initial_cash=100_000.0, first_decision=spec.warmup
    )
    politica = SB3Policy(loader(archivo), recurrent=False)
    diagnostico = dg.diagnose_policy(
        politica,
        lambda: build_env(
            validacion.series,
            config_sim,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=STUDY_PPO.reward_scale),
        ),
        dg.feature_groups(ObservationBuilder(spec).names),
        ceilings=techos,
    )
    return {
        "unit": unit.key,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        "diagnosis": diagnostico,
    }


def summarize_arm(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribucion sobre las politicas de un brazo y la lectura de la regla."""
    ids = list(range(len(registros)))

    def dist(nombre: str, valores: Sequence[float]) -> dict[str, object]:
        return summarize(nombre, ids, list(valores)).describe()

    grupos = registros[0]["diagnosis"]["groups"]
    por_grupo = {
        nombre: {
            "open_loop_mean_abs_change": dist(
                nombre,
                [
                    r["diagnosis"]["groups"][nombre]["open_loop_mean_abs_change"]
                    for r in registros
                ],
            ),
            "closed_loop_action_std": dist(
                nombre,
                [
                    r["diagnosis"]["groups"][nombre]["closed_loop_action_std"]
                    for r in registros
                ],
            ),
        }
        for nombre in grupos
    }
    cambio = float(
        np.median(
            [
                r["diagnosis"]["groups"][NEWS_GROUP]["open_loop_mean_abs_change"]
                for r in registros
            ]
        )
    )
    salida: dict[str, Any] = {
        "n_policies": len(registros),
        "baseline_action_std": dist(
            "baseline", [r["diagnosis"]["action_std"] for r in registros]
        ),
        "groups": por_grupo,
        "news_median_abs_change": cambio,
        "uses_news": cambio >= NEWS_USE_MIN_CHANGE,
        "reading": (
            f"fijar las noticias cambia la accion {cambio:.4f} de media absoluta "
            f"(mediana entre politicas): "
            + (
                "B usa las noticias"
                if cambio >= NEWS_USE_MIN_CHANGE
                else f"menos de {NEWS_USE_MIN_CHANGE}, B no usa las noticias"
            )
        ),
    }
    capturas = [r["diagnosis"]["capture"] for r in registros]
    if all(c is not None for c in capturas):
        base = [float(c) for c in capturas]
        fijo = [
            float(r["diagnosis"]["groups"][NEWS_GROUP]["closed_loop_capture"])
            for r in registros
        ]
        caen = sum(f < b for f, b in zip(fijo, base, strict=True))
        salida["capture"] = {
            "baseline": dist("capture", base),
            "news_fixed": dist("capture_news_fixed", fijo),
            "policies_that_lose_capture": caen,
        }
        salida["reading"] += (
            f". Capture: {caen} de {len(registros)} politicas lo pierden al fijar "
            "las noticias"
        )
    return salida


def run(out: Path, *, loader: Loader = _default_loader) -> dict[str, Any]:
    """Diagnostica las politicas guardadas de 5a_B y 5b_B y escribe el resumen."""
    resumen: dict[str, Any] = {}
    for a in B_ARMS:
        registros = [
            diagnose_unit(out, u, loader=loader) for u in saved_units(out, a.point)
        ]
        resumen[a.label] = summarize_arm(registros)
    cal.write_atomic(out / "rediagnosis.json", jsonable(resumen))
    return resumen
