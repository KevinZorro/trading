"""Entorno Gymnasium sobre el simulador. Wrapper delgado: solo rutea.

Lo que este archivo **no** hace, porque ya existe en ``sim/`` y duplicarlo haria
que el backtest dejara de corresponder al simulador validado:

- no ejecuta ordenes ni lleva contabilidad: eso es ``Simulator``;
- no calcula costos: eso son los modelos de ``sim.costs``;
- no dimensiona posiciones: eso es ``sim.sizing.TargetWeightSizer``;
- no calcula indicadores: eso es ``features.technical``;
- no aplica limites de riesgo: eso es ``risk.RiskLayer``, y el entorno **no la
  puentea**, la pasa a ``drive`` para que se aplique en el mismo punto en que se
  aplicaria en vivo.

Lo que si hace: recortar la accion y **registrar** el recorte, llamar al sizer,
entregar la orden al generador del motor, ensamblar la observacion y calcular la
recompensa. Cinco cosas, ninguna de ellas logica de mercado.

Secuencia de un ``step``, que es la del motor y no otra::

    accion en t -> orden -> ejecucion al open de t+1 -> marca a close de t+1

La recompensa del paso refleja el equity entre el cierre de ``t`` y el de
``t+1``, o sea el efecto de la orden que la accion genero. No hay desfase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.schema import BarSeries, FloatArray
from envs.observation import ObservationBuilder, ObservationSpec
from envs.rewards import NetReturnReward, RewardFn
from features.scaler import FeatureScaler
from sim.engine import LONG_ONLY_ACTION_RANGE, Decision, SimConfig, SimResult, Simulator
from sim.gate import OrderGate
from sim.orders import MarketOrder, OrderStatus
from sim.sizing import TargetWeightSizer


@dataclass(frozen=True)
class EnvConfig:
    """Configuracion del entorno. Se serializa junto a cada resultado."""

    observation: ObservationSpec = field(default_factory=ObservationSpec)
    sizer: TargetWeightSizer = field(default_factory=TargetWeightSizer)
    # Cota de la observacion normalizada. No recorta nada por si sola; existe
    # para declarar el Box de Gymnasium, que exige limites finitos.
    observation_bound: float = 100.0

    def describe(self) -> dict[str, object]:
        return {
            "observation": self.observation.describe(),
            "sizer": self.sizer.describe(),
            "observation_bound": self.observation_bound,
        }


@dataclass(frozen=True)
class ClipEvent:
    """Una accion que llego fuera de rango y hubo que recortar.

    Queda registrada con el valor original. Un agente entrenado sobre un rango
    que el entorno recorta en silencio aprende sobre un mundo que no existe, y
    despues nadie puede explicar por que su politica no se reproduce.
    """

    step: int
    t: int
    raw_action: float
    clipped_action: float


class TradingEnv(gym.Env[FloatArray, FloatArray]):
    """Un simbolo, un regimen, long-only.

    La accion es la fraccion del capital asignada al activo, en
    ``LONG_ONLY_ACTION_RANGE``, que se importa del motor y no se redeclara.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        series: BarSeries,
        sim_config: SimConfig,
        *,
        scaler: FeatureScaler,
        reward: RewardFn | None = None,
        env_config: EnvConfig | None = None,
        gate_factory: Callable[[], OrderGate] | None = None,
    ) -> None:
        self._series = series
        self._sim_config = sim_config
        self._config = env_config or EnvConfig()
        self._builder = ObservationBuilder(self._config.observation)
        self._reward = reward or NetReturnReward()
        # Fabrica, no instancia: la capa de riesgo acumula estado por episodio
        # (perdida del dia, turnover, kill switch) y reusarla entre episodios
        # arrastraria el kill switch de uno al siguiente.
        self._gate_factory = gate_factory

        if tuple(scaler.names) != self._builder.names:
            raise ValueError(
                "el escalador no corresponde a esta observacion: se ajusto sobre "
                f"{len(scaler.names)} features y el layout declara "
                f"{len(self._builder.names)}"
            )
        self._scaler = scaler

        cota = self._config.observation_bound
        self.observation_space = spaces.Box(
            low=-cota, high=cota, shape=(len(self._builder),), dtype=np.float64
        )
        bajo, alto = LONG_ONLY_ACTION_RANGE
        self.action_space = spaces.Box(
            low=bajo, high=alto, shape=(1,), dtype=np.float64
        )

        self._warmup = self._config.observation.warmup
        if len(series) <= self._warmup:
            raise ValueError(
                f"la serie tiene {len(series)} barras y el calentamiento de los "
                f"indicadores necesita {self._warmup}. Sin barras despues del "
                "calentamiento no hay episodio."
            )

        # Estado de episodio.
        self._sim: Simulator | None = None
        self._gen: Any = None
        self._gate: OrderGate | None = None
        self._decision: Decision | None = None
        self._equity_prev = 0.0
        self._step = 0
        self._done = True
        self._result: SimResult | None = None
        self._clips: list[ClipEvent] = []
        self._last_obs: FloatArray = np.zeros(len(self._builder))

    # -- propiedades de inspeccion --------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._builder.names

    @property
    def warmup(self) -> int:
        return self._warmup

    @property
    def clip_events(self) -> list[ClipEvent]:
        """Acciones recortadas del episodio. Vacia es la unica lectura buena."""
        return list(self._clips)

    @property
    def result(self) -> SimResult | None:
        """``SimResult`` del episodio terminado, o ``None`` si sigue corriendo."""
        return self._result

    @property
    def gate(self) -> OrderGate | None:
        return self._gate

    def describe(self) -> dict[str, object]:
        return {
            "env": self._config.describe(),
            "sim": self._sim_config.describe(),
            "reward": self._reward.describe(),
            "scaler": self._scaler.to_dict(),
            "series": self._series.meta(),
        }

    # -- API de Gymnasium ------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Arranca un episodio y adelanta hasta el fin del calentamiento.

        Durante el calentamiento se envia ``None`` en cada barra, o sea "no
        operar". Esas barras existen en el ``SimResult`` sin fills, que es la
        verdad: el agente no decidio nada porque sus indicadores todavia no
        estaban definidos.
        """
        super().reset(seed=seed)
        self._sim = Simulator(self._series, self._sim_config)
        self._gate = self._gate_factory() if self._gate_factory is not None else None
        self._gen = self._sim.drive(seed=seed, gate=self._gate, name="gym_env")
        self._reward.reset()
        self._clips = []
        self._result = None
        self._step = 0
        self._done = False

        decision = next(self._gen)
        for _ in range(self._warmup):
            decision = self._gen.send(None)
        self._decision = decision
        self._equity_prev = decision.account.equity
        self._last_obs = self._observar(decision)
        return self._last_obs, self._info(
            decision, orden=None, expirada=False, recortada=False
        )

    def step(
        self, action: FloatArray | float
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._done or self._decision is None:
            raise RuntimeError("el episodio termino; llama a reset() antes de step()")

        decision = self._decision
        cruda = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        recortada, hubo_recorte = self._recortar(cruda, decision.view.t)

        # Sizing: no lo hace el entorno. La orden sale sin redondear para que un
        # delta demasiado chico se rechace en el venue y quede en el log, en vez
        # de desaparecer como si el agente no hubiera querido operar.
        orden = self._config.sizer.order_for(recortada, decision.view, decision.account)

        equity_prev = decision.account.equity
        try:
            siguiente = self._gen.send(orden)
        except StopIteration as fin:
            # Se acabaron las barras. La orden que acabamos de enviar quedo
            # encolada en la ultima y **expira sin ejecutarse**: es el invariante
            # de la Etapa 1, y va explicito en info en vez de en silencio.
            self._result = fin.value
            self._done = True
            self._decision = None
            info = self._info(
                decision, orden=orden, expirada=True, recortada=hubo_recorte
            )
            info["sim_result"] = self._result
            return self._last_obs, 0.0, False, True, info

        self._decision = siguiente
        self._step += 1
        recompensa = self._reward.compute(equity_prev, siguiente.account.equity)
        self._equity_prev = siguiente.account.equity
        self._last_obs = self._observar(siguiente)

        # Ruina: sin equity no hay nada que decidir. Es terminacion del proceso,
        # no truncamiento por agotar los datos.
        arruinado = siguiente.account.equity <= 0.0
        if arruinado:
            self._done = True
        return (
            self._last_obs,
            recompensa,
            arruinado,
            False,
            self._info(siguiente, orden=orden, expirada=False, recortada=hubo_recorte),
        )

    # -- internos --------------------------------------------------------

    def _recortar(self, accion: float, t: int) -> tuple[float, bool]:
        """Recorta al rango del motor y devuelve si hubo recorte.

        Devuelve la bandera en vez de dejar que quien llama la deduzca del
        contador de pasos: esa deduccion es fragil y el recorte tiene que quedar
        registrado sin depender de una coincidencia de indices.
        """
        bajo, alto = LONG_ONLY_ACTION_RANGE
        if not np.isfinite(accion):
            raise ValueError(f"accion no finita: {accion!r}")
        recortada = min(max(accion, bajo), alto)
        if recortada == accion:
            return recortada, False
        self._clips.append(
            ClipEvent(
                step=self._step,
                t=t,
                raw_action=accion,
                clipped_action=recortada,
            )
        )
        return recortada, True

    def _observar(self, decision: Decision) -> FloatArray:
        return self._scaler.transform(self._builder.build(decision))

    def _info(
        self,
        decision: Decision,
        *,
        orden: MarketOrder | None,
        expirada: bool,
        recortada: bool = False,
    ) -> dict[str, Any]:
        """Todo lo que el agente no ve en la observacion pero el analisis si.

        ``order_submitted`` distingue "no quiso" de "no pudo" desde el otro lado:
        junto con ``fill_status`` y ``gate_reason`` reconstruye por que la
        posicion no cambio.
        """
        fill = decision.last_fill
        veto = decision.last_gate_rejection
        return {
            "t": decision.view.t,
            "step": self._step,
            "timestamp": decision.view.timestamp,
            "equity": decision.account.equity,
            "equity_liquidation": decision.equity_liquidation,
            "position": decision.account.position,
            "cash": decision.account.cash,
            "order_submitted": orden is not None,
            "order_qty": orden.qty if orden is not None else 0.0,
            "order_expired": expirada,
            "fill_status": fill.status.value if fill is not None else None,
            "fill_reject_reason": (
                fill.reject_reason.value
                if fill is not None and fill.reject_reason is not None
                else None
            ),
            "fill_qty": fill.qty_filled if fill is not None else 0.0,
            "fill_partial": fill is not None and fill.status is OrderStatus.PARTIAL,
            "gate_reason": veto.reason if veto is not None else None,
            "gate_detail": veto.detail if veto is not None else "",
            "action_clipped": recortada,
        }

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
