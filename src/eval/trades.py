"""Round-trips reconstruidos desde el log de fills.

El log del motor tiene **fills, no trades**. Un "win rate" necesita una unidad
de trade y hay que elegirla explicitamente; aqui se define como round-trip
*flat-to-flat*: desde que la posicion pasa de cero a positiva hasta que vuelve
a cero, con todas las compras y ventas intermedias adentro. Con
``allow_short=False`` y un solo simbolo la definicion no es ambigua.

Dos decisiones que cambian el numero que sale:

1. **La posicion todavia abierta al final no entra al win rate.** Se reporta
   aparte, en ``TradeStats.open_position``. Contarla mezclaria P&L realizado con
   no realizado, y hace que una estrategia que no cierra sus perdedoras se vea
   mejor de lo que es: exactamente el sesgo que un estudio honesto tiene que
   evitar.
2. **El P&L de un round-trip es efectivo contra efectivo**, comisiones
   incluidas. No hay reconstruccion de precio promedio ni de costo teorico: lo
   que salio del cash menos lo que entro. El spread y el slippage ya estan
   dentro de ``fill_price``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sim.orders import Fill

# Una posicion se considera cerrada por debajo de esta cantidad absoluta. No es
# cero exacto porque las cantidades pasan por ``round_qty`` y la suma de fills
# deja residuo de coma flotante (0.1 + 0.2 - 0.3 no es 0.0). El umbral esta muy
# por debajo del ``min_order_qty`` mas fino que maneja el proyecto (1e-8 en
# cripto), asi que no puede tragarse una posicion real.
POSITION_TOL = 1e-9


@dataclass(frozen=True)
class RoundTrip:
    """Un ciclo completo de cero a cero."""

    t_open: int
    t_close: int
    timestamp_open: np.datetime64
    timestamp_close: np.datetime64
    qty_bought: float
    qty_sold: float
    cost_basis: float
    proceeds: float
    commission: float
    n_fills: int

    @property
    def pnl(self) -> float:
        """Efectivo recibido menos efectivo desembolsado. Todo neto de costos."""
        return self.proceeds - self.cost_basis

    @property
    def return_pct(self) -> float:
        return self.pnl / self.cost_basis if self.cost_basis > 0 else 0.0

    @property
    def bars_held(self) -> int:
        return self.t_close - self.t_open

    @property
    def is_win(self) -> bool:
        return self.pnl > 0.0

    @property
    def is_loss(self) -> bool:
        return self.pnl < 0.0


@dataclass(frozen=True)
class OpenPosition:
    """Posicion viva al terminar la corrida. No entra a las estadisticas."""

    t_open: int
    timestamp_open: np.datetime64
    qty: float
    cost_basis: float
    commission: float
    n_fills: int


@dataclass(frozen=True)
class RoundTripLog:
    closed: list[RoundTrip] = field(default_factory=list)
    open_position: OpenPosition | None = None


class _Acumulador:
    """Estado de un round-trip en construccion."""

    __slots__ = (
        "commission",
        "cost_basis",
        "n_fills",
        "position",
        "proceeds",
        "qty_bought",
        "qty_sold",
        "t_open",
        "timestamp_open",
    )

    def __init__(self, t_open: int, timestamp_open: np.datetime64) -> None:
        self.t_open = t_open
        self.timestamp_open = timestamp_open
        self.position = 0.0
        self.qty_bought = 0.0
        self.qty_sold = 0.0
        self.cost_basis = 0.0
        self.proceeds = 0.0
        self.commission = 0.0
        self.n_fills = 0


def round_trips(fills: Sequence[Fill]) -> RoundTripLog:
    """Agrupa los fills ejecutados en round-trips flat-to-flat.

    Los rechazos y las ordenes expiradas se ignoran: no movieron efectivo. Se
    quedan en el log del motor, que es donde hay que mirarlos.
    """
    ejecutados = sorted(
        (f for f in fills if f.is_executed), key=lambda f: (f.t_fill, f.t_decision)
    )

    cerrados: list[RoundTrip] = []
    actual: _Acumulador | None = None

    for fill in ejecutados:
        if fill.qty_filled == 0.0:
            continue
        if actual is None:
            actual = _Acumulador(fill.t_fill, fill.timestamp_fill)

        actual.n_fills += 1
        actual.commission += fill.commission
        if fill.qty_filled > 0:
            actual.qty_bought += fill.qty_filled
            actual.cost_basis += fill.qty_filled * fill.fill_price + fill.commission
        else:
            vendido = -fill.qty_filled
            actual.qty_sold += vendido
            actual.proceeds += vendido * fill.fill_price - fill.commission
        actual.position += fill.qty_filled

        if abs(actual.position) <= POSITION_TOL:
            cerrados.append(
                RoundTrip(
                    t_open=actual.t_open,
                    t_close=fill.t_fill,
                    timestamp_open=actual.timestamp_open,
                    timestamp_close=fill.timestamp_fill,
                    qty_bought=actual.qty_bought,
                    qty_sold=actual.qty_sold,
                    cost_basis=actual.cost_basis,
                    proceeds=actual.proceeds,
                    commission=actual.commission,
                    n_fills=actual.n_fills,
                )
            )
            actual = None

    abierta = (
        OpenPosition(
            t_open=actual.t_open,
            timestamp_open=actual.timestamp_open,
            qty=actual.position,
            cost_basis=actual.cost_basis,
            commission=actual.commission,
            n_fills=actual.n_fills,
        )
        if actual is not None
        else None
    )
    return RoundTripLog(closed=cerrados, open_position=abierta)


@dataclass(frozen=True)
class TradeStats:
    """Estadisticas de los round-trips cerrados.

    ``win_rate`` y ``profit_factor`` son ``None`` sin trades cerrados. Un 0.0 se
    leeria como "perdio todos los trades", que es distinto de "no hizo ninguno":
    en un estudio donde la hipotesis nula es que el agente no opera, esa
    diferencia es el resultado.
    """

    n_round_trips: int
    n_wins: int
    n_losses: int
    n_scratches: int
    win_rate: float | None
    profit_factor: float | None
    gross_profit: float
    gross_loss: float
    net_pnl: float
    avg_win: float | None
    avg_loss: float | None
    avg_bars_held: float | None
    open_position: OpenPosition | None


def trade_stats(log: RoundTripLog) -> TradeStats:
    """Resume un ``RoundTripLog``. La posicion abierta se pasa sin computar."""
    cerrados = log.closed
    ganancias = [t.pnl for t in cerrados if t.is_win]
    perdidas = [t.pnl for t in cerrados if t.is_loss]
    # Un trade con P&L exactamente cero no es ni ganado ni perdido. Se cuenta
    # aparte para que n_wins + n_losses < n_round_trips no parezca un bug.
    scratches = len(cerrados) - len(ganancias) - len(perdidas)

    bruto_ganado = float(sum(ganancias))
    bruto_perdido = float(-sum(perdidas))

    if not cerrados:
        win_rate = None
        profit_factor = None
    else:
        win_rate = len(ganancias) / len(cerrados)
        if bruto_perdido > 0.0:
            profit_factor = bruto_ganado / bruto_perdido
        elif bruto_ganado > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = None

    return TradeStats(
        n_round_trips=len(cerrados),
        n_wins=len(ganancias),
        n_losses=len(perdidas),
        n_scratches=scratches,
        win_rate=win_rate,
        profit_factor=profit_factor,
        gross_profit=bruto_ganado,
        gross_loss=bruto_perdido,
        net_pnl=bruto_ganado - bruto_perdido,
        avg_win=(bruto_ganado / len(ganancias)) if ganancias else None,
        avg_loss=(bruto_perdido / len(perdidas)) if perdidas else None,
        avg_bars_held=(
            sum(t.bars_held for t in cerrados) / len(cerrados) if cerrados else None
        ),
        open_position=log.open_position,
    )
