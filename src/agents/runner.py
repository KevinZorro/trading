"""Corre una politica sobre el entorno y las estrategias contra el motor.

Las dos rutas terminan en un ``SimResult``, que es lo que hace comparables al
agente y a los baselines: la contabilidad es la misma, el ``eval.report`` que se
les aplica es el mismo, y ninguna de las dos puede inventarse un equity propio.

La unica asimetria que queda es de arranque, y esta igualada a proposito: tanto
el agente como los baselines empiezan a decidir en la barra ``warmup``. Ver
``agents.policy.WarmupDelay``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.baselines import BuyAndHold, MovingAverageCross, RandomAgent
from agents.policy import Policy, WarmupDelay
from data.schema import BarSeries
from envs.observation import ObservationSpec, fit_scaler_on_train
from envs.rewards import RewardFn
from envs.trading_env import EnvConfig, TradingEnv
from features.scaler import FeatureScaler
from sim.engine import SimConfig, SimResult, Simulator, Strategy
from sim.gate import OrderGate


class RunnerError(RuntimeError):
    """El episodio no se pudo correr."""


@dataclass(frozen=True)
class EpisodeOutcome:
    """Lo que deja un episodio, ademas del ``SimResult``.

    ``clipped_actions`` no es decorativo: un agente que recorta sistematicamente
    aprendio sobre un rango que el entorno no tiene, y ese numero es la unica
    forma de enterarse. ``steps`` permite verificar que el episodio recorrio la
    serie entera y no termino antes por ruina.
    """

    result: SimResult
    steps: int
    clipped_actions: int
    truncated: bool
    terminated: bool
    total_reward: float
    actions: tuple[float, ...] = field(repr=False, default=())

    @property
    def ruined(self) -> bool:
        return self.terminated

    def describe(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "clipped_actions": self.clipped_actions,
            "truncated": self.truncated,
            "terminated": self.terminated,
            "total_reward": self.total_reward,
            "mean_action": float(np.mean(self.actions)) if self.actions else None,
        }


def build_env(
    series: BarSeries,
    sim_config: SimConfig,
    *,
    scaler: FeatureScaler,
    env_config: EnvConfig | None = None,
    reward: RewardFn | None = None,
    gate_factory: Callable[[], OrderGate] | None = None,
) -> TradingEnv:
    """Entorno sobre una serie. El escalador entra ya ajustado, nunca se ajusta aca.

    Recibirlo en vez de ajustarlo es lo que hace imposible el anti-patron: si
    esta funcion lo ajustara, ajustaria con la serie que le pasan, y en
    evaluacion esa serie es la de test.
    """
    return TradingEnv(
        series,
        sim_config,
        scaler=scaler,
        reward=reward,
        env_config=env_config,
        gate_factory=gate_factory,
    )


def scaler_for(
    train: BarSeries, observation: ObservationSpec | None = None
) -> FeatureScaler:
    """Escalador ajustado con la porcion de train. Atajo con nombre explicito."""
    from envs.observation import ObservationBuilder

    return fit_scaler_on_train(train, ObservationBuilder(observation))


def run_policy(
    env: TradingEnv, policy: Policy, *, seed: int | None = None
) -> EpisodeOutcome:
    """Un episodio completo, de forma determinista.

    No hay exploracion: la politica se evalua tal como se usaria. Un agente que
    se evalua muestreando de su distribucion reporta el promedio de sus dudas,
    que no es lo que se pondria a operar.
    """
    observacion, _ = env.reset(seed=seed)
    policy.reset(seed)
    pasos = 0
    recompensa = 0.0
    acciones: list[float] = []
    while True:
        accion = policy.act(observacion)
        acciones.append(float(accion))
        observacion, r, terminado, truncado, info = env.step(np.array([accion]))
        recompensa += r
        pasos += 1
        if terminado or truncado:
            resultado = env.result
            if resultado is None:  # pragma: no cover - ruina en la ultima barra
                raise RunnerError("el episodio termino sin SimResult")
            return EpisodeOutcome(
                result=resultado,
                steps=pasos,
                clipped_actions=len(env.clip_events),
                truncated=truncado,
                terminated=terminado,
                total_reward=recompensa,
                actions=tuple(acciones),
            )
        del info


def run_strategy(
    series: BarSeries,
    sim_config: SimConfig,
    strategy: Strategy,
    *,
    warmup: int,
    seed: int | None = None,
    gate: OrderGate | None = None,
) -> SimResult:
    """Una estrategia contra el motor, arrancando en la misma barra que el agente."""
    return Simulator(series, sim_config).run(
        WarmupDelay(strategy, warmup), seed=seed, gate=gate
    )


def baseline_strategies(
    *, seed: int, ma_fast: int = 20, ma_slow: int = 50
) -> dict[str, Any]:
    """Los tres baselines obligatorios del proyecto.

    Se construyen aca y no en cada experimento para que ningun reporte pueda
    omitir uno sin que se note: el diccionario tiene siempre las tres claves.
    """
    return {
        "buy_and_hold": BuyAndHold(),
        "random": RandomAgent(seed=seed),
        "ma_cross": MovingAverageCross(fast=ma_fast, slow=ma_slow),
    }


def run_baselines(
    series: BarSeries,
    sim_config: SimConfig,
    *,
    warmup: int,
    seed: int,
    ma_fast: int = 20,
    ma_slow: int = 50,
) -> dict[str, SimResult]:
    """Corre los tres baselines sobre la misma serie y configuracion de costos."""
    salida: dict[str, SimResult] = {}
    for nombre, estrategia in baseline_strategies(
        seed=seed, ma_fast=ma_fast, ma_slow=ma_slow
    ).items():
        salida[nombre] = run_strategy(
            series, sim_config, estrategia, warmup=warmup, seed=seed
        )
    return salida
