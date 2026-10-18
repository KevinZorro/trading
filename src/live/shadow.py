"""Proceso de la sombra de costos (ADR 0007, secciones 3, 4 y 7; PR 4).

Cada ``Due`` de :func:`live.schedule.next_due` toma una foto del libro L2 y de las
barras diarias cerradas, y guarda por tamano y lado el costo real
(``walk_book``) junto a lo que habria cobrado el simulador (``model_cost``).

Solo lectura: ``MarketData`` no puede enviar ordenes. El tiempo entra por el
``Clock`` inyectado y la espera por ``sleep``; el unico ``SystemClock`` esta en
:func:`main`.

Un snapshot que falla **se guarda** con ``valid=False`` y el motivo, y el
proceso sigue: el hueco queda en los datos y el 25/30 del ADR se puede contar.
Solo se atrapan errores del venue y de entrada; un bug del propio proceso
tumba el proceso.

ASSUMPTION: el ADR no define ventana de validez para el snapshot de las
00:00:30, asi que no hay reintentos. Decision pendiente de Kevin (ver el PR).
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.walkforward import WalkForwardConfig
from live.config import ShadowConfig
from live.logs import configure_json_logging
from live.marketdata import BookSnapshot, MarketData, MarketDataError
from live.schedule import Due, next_due
from live.shadow_cost import ShadowCostError, Side, cost_ratios, model_cost, walk_book
from live.store import SnapshotStore, file_sha256
from sim.clock import Clock, SystemClock
from sim.costs import SlippageModel, SpreadModel

ADR_PATH = (
    Path(__file__).resolve().parents[2] / "docs/adr/0007-modo-sombra-y-testnet.md"
)
# Corwin-Schultz mira las dos ultimas barras cerradas (ADR 0007, seccion 3).
BARRAS_MODELO = 2
INTERVALO = "1d"
LADOS: tuple[Side, ...] = ("buy", "sell")

log = logging.getLogger(__name__)


def iso(ts: np.datetime64) -> str:
    return str(np.datetime_as_string(np.datetime64(ts, "ms"))) + "Z"


def dia(ts: np.datetime64) -> str:
    return str(np.datetime_as_string(np.datetime64(ts, "D")))


@dataclass(frozen=True)
class SnapshotRecord:
    kind: str
    scheduled: str
    received_at: str | None
    taken_at: str
    last_update_id: int | None
    mid: float | None
    bar_close: str | None
    valid: bool
    reasons: list[str]
    cells: list[dict[str, Any]] = field(default_factory=list)
    type: str = "snapshot"


class ShadowRunner:
    def __init__(
        self,
        config: ShadowConfig,
        market: MarketData,
        clock: Clock,
        store: SnapshotStore,
        *,
        spread: SpreadModel,
        slippage: SlippageModel,
        sleep: Callable[[float], object],
    ) -> None:
        self._config = config
        self._market = market
        self._clock = clock
        self._store = store
        self._spread = spread
        self._slippage = slippage
        self._sleep = sleep
        self._last_valid: str | None = None
        self._invalid_streak = 0

    def take_snapshot(self, due: Due) -> tuple[SnapshotRecord, BookSnapshot | None]:
        cfg = self._config
        razones: list[str] = []
        libro: BookSnapshot | None = None
        try:
            libro = self._market.depth(cfg.symbol, cfg.depth_limit)
        except MarketDataError as exc:
            razones.append(f"depth: {exc}")
        barras = None
        try:
            barras = self._market.closed_klines(cfg.symbol, INTERVALO, BARRAS_MODELO)
        except MarketDataError as exc:
            razones.append(f"klines: {exc}")

        cierre: str | None = None
        if libro is not None and barras is not None:
            cierre = iso(barras.timestamp[-1])
            # El timestamp de una barra es su cierre (CLAUDE.md): a las 00:00:30 del
            # dia D la ultima diaria cerrada cierra a las 00:00 de D. Si el venue
            # aun no la publico, la ultima es la anterior: no es lo que vio el
            # backtest, y se marca en vez de medir contra una barra vieja.
            esperado = np.datetime64(libro.received_at, "D").astype("datetime64[ns]")
            if barras.timestamp[-1] != esperado:
                razones.append(
                    f"barra_vieja: la ultima diaria cerrada cierra en {cierre} y se "
                    f"esperaba {iso(esperado)} (kline cerrada aun no publicada?)"
                )
                barras = None

        celdas: list[dict[str, Any]] = []
        if libro is not None:
            try:
                celdas = self._cells(libro, barras)
            except ShadowCostError as exc:
                razones.append(f"costo: {exc}")
                celdas = []
        valido = not razones and libro is not None
        registro = SnapshotRecord(
            kind=due.kind,
            scheduled=iso(due.at),
            received_at=iso(libro.received_at) if libro else None,
            taken_at=iso(self._clock.now()),
            last_update_id=libro.last_update_id if libro else None,
            mid=(libro.best_bid + libro.best_ask) / 2.0 if libro else None,
            bar_close=cierre,
            valid=valido,
            reasons=razones,
            cells=celdas,
        )
        return registro, libro

    def _cells(self, libro: BookSnapshot, barras: Any) -> list[dict[str, Any]]:
        celdas: list[dict[str, Any]] = []
        for tamano in self._config.order_sizes_usdt:
            modelo = None
            for lado in LADOS:
                real = walk_book(libro, lado, tamano)
                if barras is not None:
                    if modelo is None:
                        modelo = model_cost(
                            barras,
                            mid=real.mid,
                            notional=tamano,
                            as_of=libro.received_at,
                            spread=self._spread,
                            slippage=self._slippage,
                        )
                    cocientes: Any = asdict(cost_ratios(real, modelo))
                else:
                    cocientes = None
                celdas.append(
                    {
                        "size": tamano,
                        "side": lado,
                        "real": asdict(real),
                        "model": asdict(modelo) if modelo else None,
                        "ratios": cocientes,
                    }
                )
        return celdas

    def step(self, due: Due) -> SnapshotRecord:
        registro, libro = self.take_snapshot(due)
        d = dia(np.datetime64(registro.scheduled.rstrip("Z"), "ns"))
        self._store.append_snapshot(d, asdict(registro))
        if libro is not None:
            self._store.append_book(
                d,
                {
                    "scheduled": registro.scheduled,
                    "received_at": registro.received_at,
                    "last_update_id": libro.last_update_id,
                    "bids": libro.bids.tolist(),
                    "asks": libro.asks.tolist(),
                },
            )
        if registro.valid:
            self._last_valid = registro.scheduled
            self._invalid_streak = 0
        else:
            self._invalid_streak += 1
        self._heartbeat(registro.scheduled)
        log.log(
            logging.INFO if registro.valid else logging.WARNING,
            "snapshot",
            extra={
                "kind": due.kind,
                "valid": registro.valid,
                "reasons": registro.reasons,
            },
        )
        return registro

    def _heartbeat(self, last: str | None) -> None:
        self._store.write_heartbeat(
            {
                "at": iso(self._clock.now()),
                "last_scheduled": last,
                "last_valid_scheduled": self._last_valid,
                "consecutive_invalid": self._invalid_streak,
            }
        )

    def run(self, stop: Callable[[], bool]) -> None:
        cfg = self._config
        self._heartbeat(None)  # el watchdog ve al proceso vivo desde el arranque
        while not stop():
            ahora = self._clock.now()
            due = next_due(ahora, cfg.snapshot_minutes, cfg.daily_snapshot_utc)
            self._sleep(max(0.0, float((due.at - ahora) / np.timedelta64(1, "s"))))
            if stop():
                break
            self.step(due)


def build_runner(
    config: ShadowConfig,
    clock: Clock,
    *,
    sleep: Callable[[float], object],
    fetch: Callable[[str], bytes] | None = None,
    adr_path: Path = ADR_PATH,
    git_sha: str = "unknown",
) -> ShadowRunner:
    from live.marketdata import BinanceRestMarketData

    sim = WalkForwardConfig().sim_config("principal")
    cabecera: Mapping[str, Any] = {
        "type": "header",
        "adr": adr_path.name,
        "adr_sha256": file_sha256(adr_path),  # falla si no esta: sin ADR no hay datos
        "git_sha": git_sha,
        "config": config.describe(),
    }
    market = (
        BinanceRestMarketData(config.market_data_url, clock, fetch=fetch)
        if fetch
        else BinanceRestMarketData(config.market_data_url, clock)
    )
    return ShadowRunner(
        config,
        market,
        clock,
        SnapshotStore(config.data_dir, cabecera),
        spread=sim.spread,
        slippage=sim.slippage,
        sleep=sleep,
    )


def main(environ: Mapping[str, str]) -> int:  # pragma: no cover - proceso real
    configure_json_logging(sys.stdout)
    config = ShadowConfig.from_env(environ)
    parar = threading.Event()
    for senal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(senal, lambda *_: parar.set())
    runner = build_runner(
        config,
        SystemClock(),
        sleep=parar.wait,  # despierta al instante con SIGTERM
        git_sha=environ.get("SHADOW_GIT_SHA", "unknown"),
    )
    runner.run(parar.is_set)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import os

    raise SystemExit(main(os.environ))
