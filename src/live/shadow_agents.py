"""Los Agentes A congelados dentro del proceso de la sombra (ADR 0009, PR 5c).

Cada ``daily`` de :func:`live.schedule.next_due` corre, **despues** de guardar el
snapshot de costos, un :class:`~live.agent_step.AgentStep` por semilla de
``STUDY_SEEDS``. Cada agente tiene su propio ``ShadowVenue`` y ``FillStore``
(``<data_dir>/agents/seed-<n>/``); comparten el libro y las barras de ese instante
(:class:`InstantMarket`), no el estado de la cuenta.

Los agentes **no pueden invalidar ni impedir el snapshot**, que es la medicion
pre-registrada (ADR 0007): se ejecutan despues de persistirlo y cualquier fallo de
un agente -pesos ausentes, manifest invalido, ``torch`` no instalado, barra vieja,
un bug- se registra y se aisla. Atrapar ``Exception`` aca es deliberado y distinto
de ``shadow.py`` (donde un bug tumba el proceso): una politica RL es codigo cuyo
fallo no debe costar datos de costos. El fallo queda en el log y en el heartbeat.

ASSUMPTION: ``RISK_LIMITS`` no esta en ningun ADR. La capa se aplica para ejercitar
el mismo camino que usara el dinero real; con un tope de 100 % del equity (el sizer
usa 98 %) no cambia las decisiones del agente salvo una caida diaria > 10 %.
No dispara kill switch: su estado no sobrevive a un reinicio (agent_step.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agents.protocol import STUDY_SEEDS
from agents.walkforward import WalkForwardConfig
from data.instruments import InstrumentSpec
from data.schema import BarSeries
from envs.observation import ObservationBuilder
from envs.trading_env import EnvConfig
from live.agent_step import AgentStep
from live.fillstore import FillStore, venue_header
from live.frozen import FrozenPolicy
from live.marketdata import BookSnapshot, MarketData
from live.venue import ShadowVenue
from risk.layer import RiskLayer, RiskLimits
from sim.clock import Clock
from sim.costs import SchemaCommission
from sim.sizing import TargetWeightSizer

AGENTS_DIR = "agents"
INITIAL_CASH = 100.0  # ADR 0007: ledger de 100 USDT por agente
RISK_LIMITS = RiskLimits(
    max_position_pct_equity=1.0, max_daily_loss_pct=0.10, trip_on_daily_loss=False
)

log = logging.getLogger(__name__)


class InstantMarket:
    """``MarketData`` que repite las mismas respuestas durante un instante.

    Los diez agentes ven el mismo libro y las mismas barras (ADR 0009, punto 2), y
    el libro es el del snapshot de costos cuando existe. ``reset`` abre el instante.
    """

    def __init__(self, inner: MarketData) -> None:
        self._inner = inner
        self._depth: dict[tuple[str, int], BookSnapshot] = {}
        self._klines: dict[tuple[str, str, int], BarSeries] = {}

    def reset(
        self, libro: BookSnapshot | None = None, *, symbol: str, limit: int
    ) -> None:
        self._depth.clear()
        self._klines.clear()
        if libro is not None:
            self._depth[(symbol, limit)] = libro

    def depth(self, symbol: str, limit: int) -> BookSnapshot:
        if (symbol, limit) not in self._depth:
            self._depth[(symbol, limit)] = self._inner.depth(symbol, limit)
        return self._depth[(symbol, limit)]

    def closed_klines(self, symbol: str, interval: str, n: int) -> BarSeries:
        clave = (symbol, interval, n)
        if clave not in self._klines:
            self._klines[clave] = self._inner.closed_klines(symbol, interval, n)
        return self._klines[clave]

    def instrument(self, symbol: str) -> InstrumentSpec:
        return self._inner.instrument(symbol)


@dataclass(frozen=True)
class AgentRun:
    seed: int
    status: Literal["ok", "error"]
    detail: str


def _spec_sha256(spec: InstrumentSpec) -> str:
    cuerpo = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str)
    return hashlib.sha256(cuerpo.encode()).hexdigest()


class ShadowAgents:
    def __init__(
        self,
        *,
        frozen_dir: Path,
        data_dir: Path,
        market: MarketData,
        clock: Clock,
        symbol: str,
        depth_limit: int,
        loader: Callable[[Path], Any] | None = None,
        on_progress: Callable[[], object] = lambda: None,
    ) -> None:
        self._frozen_dir = frozen_dir
        self._data_dir = data_dir
        self._market = InstantMarket(market)
        self._clock = clock
        self._symbol = symbol
        self._depth_limit = depth_limit
        self._loader = loader
        self._on_progress = on_progress
        self._steps: dict[int, AgentStep] = {}
        self._errors: dict[int, str] = {}

    def _build(self, seed: int) -> AgentStep:
        agente = (
            FrozenPolicy.load(self._frozen_dir, seed)
            if self._loader is None
            else FrozenPolicy.load(self._frozen_dir, seed, loader=self._loader)
        )
        spec = self._market.instrument(self._symbol)
        ahora = self._clock.now()
        almacen = FillStore(
            self._data_dir / AGENTS_DIR / f"seed-{seed}",
            venue_header(
                initial_cash=INITIAL_CASH,
                spec_sha256=_spec_sha256(spec),
                spec_received_at=ahora,
            ),
        )
        venue = ShadowVenue(
            market=self._market,
            instrument=spec,
            clock=self._clock,
            store=almacen,
            initial_cash=INITIAL_CASH,
            depth_limit=self._depth_limit,
        )
        return AgentStep(
            agent=agente,
            builder=ObservationBuilder(EnvConfig().observation),
            sizer=TargetWeightSizer(safety=WalkForwardConfig().safety),
            gate=RiskLayer(RISK_LIMITS, self._clock),
            venue=venue,
            store=almacen,
            market=self._market,
            symbol=self._symbol,
            commission=SchemaCommission(schema=spec.commission_schema),
            clock=self._clock,
            run_id=f"A{seed}",
            depth_limit=self._depth_limit,
        )

    def run(self, libro: BookSnapshot | None) -> list[AgentRun]:
        """Un paso por semilla, aislados entre si. Nunca lanza ``Exception``."""
        self._market.reset(libro, symbol=self._symbol, limit=self._depth_limit)
        corridas: list[AgentRun] = []
        for seed in STUDY_SEEDS:
            try:
                if seed not in self._steps:  # reintenta cada dia lo que no cargo
                    self._steps[seed] = self._build(seed)
                salida = self._steps[seed].run()
                self._errors.pop(seed, None)
                corridas.append(AgentRun(seed, "ok", f"w={salida.target_weight:.4f}"))
            except Exception as exc:
                detalle = f"{type(exc).__name__}: {exc}"
                self._errors[seed] = detalle
                corridas.append(AgentRun(seed, "error", detalle))
                log.exception("agente fallo", extra={"seed": seed})
            self._on_progress()
        return corridas

    def summary(self) -> dict[str, Any]:
        """Para el heartbeat: cuantos cargaron y que fallo en la ultima corrida."""
        return {
            "loaded": sorted(self._steps),
            "errors": {str(s): m for s, m in sorted(self._errors.items())},
        }
