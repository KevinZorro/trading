"""Agente A: PPO sobre el entorno, con politica MLP o recurrente.

Adaptador delgado sobre ``stable-baselines3``. Todo lo que decide el resultado
-la observacion, el sizing, los costos, la contabilidad- ya vive en ``envs/`` y
``sim/``; aca solo se elige el optimizador y se serializan sus hiperparametros.

**El import de ``stable_baselines3`` es diferido a proposito.** ``torch`` pesa
varios GB y solo hace falta para entrenar: el protocolo de validacion, el
reporte por semillas y el walk-forward se testean sin el, y el job principal de
CI sigue instalando solo el grupo por defecto. ``uv sync --group rl`` instala lo
que hace falta para entrenar.

Sobre la politica recurrente: el estado del mercado es parcialmente observable
-la observacion no dice en que regimen se esta- y una LSTM puede inferirlo de la
historia reciente. Es exactamente lo que el nivel 3 de los fixtures pone a
prueba, y por eso es una opcion y no el default: en los niveles 0, 1 y 2 el
estado **si** es completamente observable (el optimo depende solo de ``r_t``, que
esta en la observacion), asi que ahi la memoria no puede ayudar y solo agrega
parametros que sobreajustar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.schema import FloatArray
from envs.trading_env import TradingEnv
from sim.engine import LONG_ONLY_ACTION_RANGE


class PPOUnavailableError(ImportError):
    """No esta instalado el grupo ``rl``."""


def _require_sb3() -> tuple[Any, Any]:
    """Importa SB3 y da un error util si falta, en vez de un ImportError pelado."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise PPOUnavailableError(
            "stable-baselines3 no esta instalado. Es un grupo opcional porque "
            "torch pesa varios GB y solo hace falta para entrenar: "
            "`uv sync --group rl`."
        ) from exc
    recurrente: Any
    try:
        from sb3_contrib import RecurrentPPO

        recurrente = RecurrentPPO
    except ImportError:  # pragma: no cover - depende del entorno
        recurrente = None
    return PPO, recurrente


@dataclass(frozen=True)
class PPOConfig:
    """Hiperparametros. Se serializan junto a cada resultado, sin excepcion.

    Notas sobre los valores que no son el default de SB3:

    - ``gamma=0.99`` da un horizonte efectivo de ~100 barras. En los niveles 0,
      1, 3 y 4 el optimo es **miope** -las decisiones se desacoplan sin costos-
      asi que gamma casi no importa; en el nivel 2 si importa, porque la banda
      de histeresis existe precisamente por el acoplamiento que introduce el
      costo de cambiar de estado.
    - ``reward_scale=100`` multiplica el retorno por barra, que vive en el orden
      de 1e-2. Es una transformacion afin positiva: **no cambia la politica
      optima**, solo la escala de los gradientes. Queda serializado porque
      cambiarlo cambia el entrenamiento aunque no cambie el problema.
    - ``ent_coef=0.0`` es el default de PPO. Con una accion continua acotada, la
      entropia inicial ya es alta y forzarla mas retrasa la saturacion, que es
      justamente la forma del optimo en estos fixtures.
    """

    total_timesteps: int = 200_000
    n_steps: int = 2_048
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    recurrent: bool = False
    lstm_hidden_size: int = 64
    reward_scale: float = 100.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps debe ser positivo")
        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size={self.batch_size} supera n_steps={self.n_steps}: PPO "
                "no puede formar ni un minibatch completo"
            )
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                f"n_steps={self.n_steps} no es multiplo de batch_size="
                f"{self.batch_size}; SB3 deja un minibatch corto y el ultimo "
                "gradiente pesa distinto que el resto"
            )
        if self.reward_scale <= 0:
            raise ValueError("reward_scale debe ser positivo")

    @property
    def policy_name(self) -> str:
        return "MlpLstmPolicy" if self.recurrent else "MlpPolicy"

    def describe(self) -> dict[str, object]:
        salida: dict[str, object] = dict(vars(self))
        salida["net_arch"] = list(self.net_arch)
        salida["policy_name"] = self.policy_name
        return salida


class SB3Policy:
    """Envuelve un modelo entrenado y lo expone como :class:`~agents.policy.Policy`.

    ``deterministic=True`` siempre: la evaluacion tiene que medir la politica que
    se pondria a operar, no el promedio de sus dudas.

    Con politica recurrente hay que llevar el estado de la LSTM entre pasos y
    reiniciarlo al empezar el episodio. Si no se reinicia, el primer paso de un
    episodio arrastra la memoria del anterior y la evaluacion deja de ser
    reproducible: el resultado dependeria del orden en que se evaluaron los
    episodios.
    """

    def __init__(self, model: Any, *, recurrent: bool, label: str = "ppo") -> None:
        self._model = model
        self._recurrent = recurrent
        self._label = label
        self._state: Any = None
        self._episode_start = True

    @property
    def name(self) -> str:
        return self._label

    @property
    def model(self) -> Any:
        return self._model

    def reset(self, seed: int | None = None) -> None:
        self._state = None
        self._episode_start = True

    def act(self, observation: FloatArray) -> float:
        if self._recurrent:
            accion, self._state = self._model.predict(
                observation,
                state=self._state,
                episode_start=np.array([self._episode_start]),
                deterministic=True,
            )
        else:
            accion, _ = self._model.predict(observation, deterministic=True)
        self._episode_start = False
        return float(np.asarray(accion, dtype=np.float64).reshape(-1)[0])

    def describe(self) -> dict[str, object]:
        return {"policy": "sb3_ppo", "recurrent": self._recurrent, "label": self._label}

    def save(self, path: str) -> None:
        self._model.save(path)


def train_ppo(
    env_factory: Callable[[], TradingEnv],
    config: PPOConfig,
    *,
    seed: int,
    verbose: int = 0,
) -> SB3Policy:
    """Entrena y devuelve la politica lista para evaluar.

    ``env_factory`` y no un entorno: SB3 lo envuelve y lo consume, y reusar la
    misma instancia entre el entrenamiento y la evaluacion arrastraria el estado
    del episodio. Que sea una fabrica tambien deja claro que el entorno de
    entrenamiento es **el de train**: el de evaluacion se construye aparte, con
    la serie que corresponda y con el mismo escalador.

    La reproducibilidad es la que da SB3 con una semilla fija y las versiones
    pinneadas. No es bit-a-bit entre maquinas -torch no lo garantiza entre
    arquitecturas- y por eso todo se reporta como distribucion sobre semillas.
    """
    PPO, RecurrentPPO = _require_sb3()
    if config.recurrent:
        if RecurrentPPO is None:  # pragma: no cover - depende del entorno
            raise PPOUnavailableError(
                "la politica recurrente necesita sb3-contrib: `uv sync --group rl`"
            )
        constructor = RecurrentPPO
        extra: dict[str, Any] = {"lstm_hidden_size": config.lstm_hidden_size}
    else:
        constructor = PPO
        extra = {}

    modelo = constructor(
        config.policy_name,
        env_factory(),
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": list(config.net_arch), **extra},
        seed=seed,
        device=config.device,
        verbose=verbose,
    )
    modelo.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    return SB3Policy(
        modelo,
        recurrent=config.recurrent,
        label=f"ppo_lstm_seed{seed}" if config.recurrent else f"ppo_seed{seed}",
    )


def action_saturation(actions: tuple[float, ...], *, tol: float = 1e-9) -> float:
    """Fraccion de acciones pegadas a un extremo del rango.

    **No es lo mismo que el recorte del entorno.** SB3 recorta las acciones
    contra el espacio antes de entregarlas, asi que ``ClipEvent`` queda casi
    siempre vacio y no sirve para diagnosticar. Lo que si se puede medir es la
    saturacion: que fraccion de las decisiones quedo exactamente en 0 o en 1.

    Una saturacion alta **es lo esperado** en estos fixtures -el optimo es
    bang-bang- y por eso hay que reportarla como saturacion y no confundirla con
    un recorte, que seria un sintoma de que el agente aprendio sobre otro rango.
    """
    if not actions:
        return 0.0
    bajo, alto = LONG_ONLY_ACTION_RANGE
    arreglo = np.asarray(actions, dtype=np.float64)
    pegadas = np.isclose(arreglo, bajo, atol=tol) | np.isclose(arreglo, alto, atol=tol)
    return float(pegadas.mean())
