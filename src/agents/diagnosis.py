"""Diagnostico de por que 4b no pasa a ningun presupuesto (calibracion, ADR 0010).

G1 a G4 fallaron con la dispersion de la accion en 0.33 a 0.35 (maximo 0.15)
**aunque sigma bajo de 0.81 a 0.05**: la exploracion se apago y la politica
determinista sigue rotando. Esta herramienta mide de que depende esa accion. No
cambia ningun criterio ni elige un presupuesto, y corre solo sobre el fixture 4b.

Para cada politica entrenada se reemplaza, de a un grupo de columnas de la
observacion, el grupo por su media en el episodio:

- **lazo abierto:** se recalculan las acciones sobre las observaciones ya
  grabadas. Mide cuanto depende la accion de ese grupo.
- **lazo cerrado:** se vuelve a correr el episodio con la politica viendo el grupo
  fijo. Si la rotacion desaparece, el grupo era parte de un bucle.

``posicion`` incluye ``position_weight`` **y** ``cash_weight``: son
complementarios, y fijar solo uno dejaria la posicion visible por el otro.

**Regla de lectura, declarada antes de correr.** Un grupo es *suficiente* si, con
el fijo en lazo cerrado, la mediana de la dispersion de la accion entre las
politicas queda en o bajo ``ProtocolThresholds.level_4b_max_action_std`` (0.15).
Es el umbral que ya existe, no uno nuevo. Si ningun grupo lo es, la rotacion no se
explica por un solo grupo y se dice asi.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from agents.calibration import Unit, machine
from agents.policy import Policy
from agents.ppo import PPOConfig, train_ppo
from agents.protocol import STUDY_SEEDS, ProtocolThresholds, _sim_config
from agents.runner import build_env, run_policy, scaler_for
from data.fixtures import DEFAULT_N_BARS, Ceilings, level_4b_detectable_drift
from data.schema import FloatArray
from envs.observation import ObservationBuilder, ObservationSpec
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig, TradingEnv
from eval.distribution import summarize
from features.news import NEWS_FEATURE_NAMES
from sim.sizing import TargetWeightSizer

#: Disjuntos de ``PATH_SEEDS`` (701 a 761) y de ``CALIBRATION_PATH_SEEDS``.
DIAGNOSIS_PATH_SEEDS: tuple[int, ...] = (839, 853, 857, 859, 863)
#: 5 caminos x 2 semillas = 10 politicas, el minimo de ``summarize``.
DIAGNOSIS_AGENT_SEEDS: tuple[int, ...] = STUDY_SEEDS[:2]
DIAGNOSIS_TIMESTEPS = 120_000
DIAGNOSIS_POINT = "D"

Trainer = Callable[[Callable[[], TradingEnv], PPOConfig, int], Policy]


def diagnosis_units() -> tuple[Unit, ...]:
    return tuple(
        Unit(DIAGNOSIS_POINT, DIAGNOSIS_TIMESTEPS, camino, semilla)
        for camino in DIAGNOSIS_PATH_SEEDS
        for semilla in DIAGNOSIS_AGENT_SEEDS
    )


def feature_groups(names: tuple[str, ...]) -> dict[str, list[int]]:
    """Indices de las columnas de cada grupo, a partir de los nombres reales."""
    grupos: dict[str, list[int]] = {
        "retornos": [],
        "tecnicos": [],
        "posicion": [],
        "resultado_orden": [],
    }
    for i, nombre in enumerate(names):
        if nombre.startswith("log_return_"):
            grupos["retornos"].append(i)
        elif nombre in ("position_weight", "cash_weight"):
            grupos["posicion"].append(i)
        elif nombre in (
            "friction_gap",
            "last_order_blocked",
            "last_order_rejected",
            "last_fill_ratio",
        ):
            grupos["resultado_orden"].append(i)
        elif nombre in NEWS_FEATURE_NAMES:
            grupos.setdefault("noticias", []).append(i)
        else:
            grupos["tecnicos"].append(i)
    return grupos


class _Recorder:
    """Politica que guarda lo que ve, para recalcular acciones despues."""

    def __init__(self, inner: Policy) -> None:
        self._inner = inner
        self.observations: list[FloatArray] = []

    @property
    def name(self) -> str:
        return self._inner.name

    def reset(self, seed: int | None = None) -> None:
        self._inner.reset(seed)

    def act(self, observation: FloatArray) -> float:
        self.observations.append(np.array(observation, dtype=np.float64))
        return self._inner.act(observation)

    def describe(self) -> dict[str, object]:
        return self._inner.describe()


class _Fixed:
    """Politica que ve ciertas columnas fijas en un valor."""

    def __init__(self, inner: Policy, idx: list[int], values: FloatArray) -> None:
        self._inner = inner
        self._idx = idx
        self._values = values

    @property
    def name(self) -> str:
        return self._inner.name

    def reset(self, seed: int | None = None) -> None:
        self._inner.reset(seed)

    def act(self, observation: FloatArray) -> float:
        visto = np.array(observation, dtype=np.float64)
        visto[self._idx] = self._values
        return self._inner.act(visto)

    def describe(self) -> dict[str, object]:
        return self._inner.describe()


def diagnose_policy(
    policy: Policy,
    make_env: Callable[[], TradingEnv],
    groups: dict[str, list[int]],
    *,
    ceilings: Ceilings | None = None,
) -> dict[str, Any]:
    """Dispersion de la accion con cada grupo fijo, en lazo abierto y cerrado.

    Con ``ceilings`` agrega el ``capture`` del episodio base y, por grupo, el de
    lazo cerrado: es lo que pide el control positivo del ADR 0012 (5), donde
    fijar las noticias tiene que hundir el ``capture``.
    """

    def captura(salida: Any) -> float | None:
        if ceilings is None:
            return None
        return ceilings.capture(np.asarray(salida.result.equity, dtype=np.float64))

    grabadora = _Recorder(policy)
    base = run_policy(make_env(), grabadora)
    acciones = np.asarray(base.actions, dtype=np.float64)
    obs = np.asarray(grabadora.observations, dtype=np.float64)
    medias = obs.mean(axis=0)
    salida: dict[str, Any] = {
        "n_steps": len(acciones),
        "action_std": float(acciones.std()),
        "mean_action": float(acciones.mean()),
        "capture": captura(base),
        "groups": {},
    }
    for nombre, idx in groups.items():
        fijo = obs.copy()
        fijo[:, idx] = medias[idx]
        abierto = np.array([policy.act(fila) for fila in fijo], dtype=np.float64)
        cerrado = run_policy(make_env(), _Fixed(policy, idx, medias[idx]))
        salida["groups"][nombre] = {
            "open_loop_action_std": float(abierto.std()),
            "open_loop_mean_abs_change": float(np.abs(abierto - acciones).mean()),
            "closed_loop_action_std": float(np.std(cerrado.actions)),
            "closed_loop_capture": captura(cerrado),
        }
    return salida


def train_unit(
    unit: Unit,
    *,
    trainer: Trainer | None = None,
    ppo: PPOConfig | None = None,
    observation: ObservationSpec | None = None,
) -> dict[str, Any]:
    """Entrena una politica sobre un camino de 4b y la diagnostica.

    ``ppo`` y ``observation`` los pasa la confirmacion del ADR 0011 para
    re-diagnosticar con ``C*``; sin ellos es la configuracion de #57.
    """
    fixture = level_4b_detectable_drift(DEFAULT_N_BARS, seed=unit.path_seed)
    train, validacion, _ = fixture.split()
    spec = observation or ObservationSpec()
    escalador = scaler_for(train.series, spec)
    config_sim = _sim_config(fixture, 100_000.0, 0.0)
    env_config = EnvConfig(observation=spec, sizer=TargetWeightSizer(safety=0.98))
    config = ppo or PPOConfig(total_timesteps=unit.timesteps)

    def construir(f: Any) -> TradingEnv:
        return build_env(
            f.series,
            config_sim,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=config.reward_scale),
        )

    # perf_counter mide duracion; no consulta la hora de pared.
    inicio = time.perf_counter()
    entrenar = trainer or (
        lambda factory, cfg, seed: train_ppo(factory, cfg, seed=seed)
    )
    politica = entrenar(lambda: construir(train), config, unit.agent_seed)
    grupos = feature_groups(ObservationBuilder(spec).names)
    diagnostico = diagnose_policy(politica, lambda: construir(validacion), grupos)
    sigma = getattr(politica, "sigma", None)
    return {
        "unit": unit.key,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        "timesteps": unit.timesteps,
        "wall_seconds": time.perf_counter() - inicio,
        "policy_sigma": float(sigma()) if callable(sigma) else None,
        "machine": machine(),
        "diagnosis": diagnostico,
    }


def summarize_diagnosis(registros: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribucion sobre las politicas y la lectura segun la regla declarada."""
    umbral = ProtocolThresholds().level_4b_max_action_std
    ids = list(range(len(registros)))

    def dist(nombre: str, valores: list[float]) -> dict[str, object]:
        return summarize(nombre, ids, list(valores)).describe()

    base = [r["diagnosis"]["action_std"] for r in registros]
    grupos = registros[0]["diagnosis"]["groups"]
    por_grupo: dict[str, Any] = {}
    suficientes: list[str] = []
    for nombre in grupos:
        cerrado = [
            r["diagnosis"]["groups"][nombre]["closed_loop_action_std"]
            for r in registros
        ]
        por_grupo[nombre] = {
            "open_loop_action_std": dist(
                nombre,
                [
                    r["diagnosis"]["groups"][nombre]["open_loop_action_std"]
                    for r in registros
                ],
            ),
            "open_loop_mean_abs_change": dist(
                nombre,
                [
                    r["diagnosis"]["groups"][nombre]["open_loop_mean_abs_change"]
                    for r in registros
                ],
            ),
            "closed_loop_action_std": dist(nombre, cerrado),
        }
        if float(np.median(cerrado)) <= umbral:
            suficientes.append(nombre)
    lectura = (
        f"fijar {', '.join(suficientes)} baja la mediana de la dispersion a <= "
        f"{umbral:.2f}: ese grupo sostiene la rotacion"
        if suficientes
        else "ningun grupo, fijado solo, baja la mediana de la dispersion a <= "
        f"{umbral:.2f}: la rotacion no se explica por un solo grupo"
    )
    return {
        "n_policies": len(registros),
        "threshold": umbral,
        "baseline_action_std": dist("baseline", base),
        "groups": por_grupo,
        "sufficient_groups": suficientes,
        "reading": lectura,
    }
