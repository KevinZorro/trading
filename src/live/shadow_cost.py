"""Costo real de un libro L2 contra lo que cobraba el simulador (ADR 0007, seccion 3).

Funciones puras: sin red, sin reloj y sin I/O. El proceso que las llama (PR 4)
toma el snapshot y la hora; aca solo se calcula.

- ``walk_book``: lo que costaria ejecutar ``notional`` USDT contra el libro.
- ``model_cost``: lo que habria cobrado el simulador por la misma orden, con
  los modelos de ``sim.costs`` **inyectados**. Los parametros (``max_bps``, ``k``)
  no se copian aca: si C3 los cambia, la sombra sigue midiendo contra el mismo
  modelo.
- ``cost_ratios``: los dos cocientes pre-registrados, **sin comision**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from data.schema import BarSeries
from live.marketdata import BookSnapshot
from sim.costs import SlippageModel, SpreadContext, SpreadModel

Side = Literal["buy", "sell"]


class ShadowCostError(ValueError):
    """Entrada invalida: se rechaza con el motivo, nunca se corrige en silencio."""


@dataclass(frozen=True)
class BookWalk:
    """Costo "real": recorrer el libro consumiendo ``notional`` USDT.

    Si el libro no alcanza, ``exhausted`` es True, ``filled_notional < notional``
    y los costos son los de lo llenado.
    """

    side: Side
    notional: float
    mid: float
    best: float  # mejor precio del lado que se recorre
    vwap: float
    filled_notional: float
    levels_used: int
    exhausted: bool
    half_spread_rel: float
    depth_impact_rel: float
    real_rel: float


@dataclass(frozen=True)
class ModelCost:
    """Lo que cobraba el simulador de C3 por la misma orden, sin comision."""

    notional: float
    mid: float
    participation: float
    half_spread_rel: float
    slippage_rel: float
    model_rel: float


@dataclass(frozen=True)
class CostRatios:
    """``None`` donde el denominador es 0 o el libro se agoto; nunca inf ni 0."""

    ratio_total: float | None
    ratio_spread: float | None


def _check_notional(notional: float) -> None:
    if not math.isfinite(notional) or notional <= 0:
        raise ShadowCostError(f"notional debe ser finito y > 0, es {notional!r}")


def walk_book(book: BookSnapshot, side: Side, notional: float) -> BookWalk:
    """Consume ``notional`` USDT nivel por nivel; el ultimo nivel, parcial.

    Compra recorre los asks y venta los bids.

    ASSUMPTION: no se redondea a lote ni a tick. Mide el costo del nocional
    pedido. A 10 USDT el lote de 1e-5 BTC de BTCUSDT redondea ~1 % de la
    cantidad y no cambia el costo relativo de forma material.

    ASSUMPTION: si los niveles no alcanzan, el ADR no fija que se registra. Se
    devuelve el costo de lo llenado con ``exhausted=True`` y
    ``filled_notional < notional``; ``cost_ratios`` lo deja fuera.
    """
    if side not in ("buy", "sell"):
        raise ShadowCostError(f"side debe ser 'buy' o 'sell', es {side!r}")
    _check_notional(notional)

    niveles = book.asks if side == "buy" else book.bids
    mid = (book.best_bid + book.best_ask) / 2.0
    best = float(niveles[0, 0])

    restante = notional
    qty = 0.0
    qty_por_distancia = 0.0  # suma de cantidad * (precio - best), con signo
    usados = 0
    for precio, cantidad in niveles:
        usados += 1
        if restante <= precio * cantidad:
            tomada = restante / precio  # ultimo nivel, parcial (o justo)
            restante = 0.0
        else:
            tomada = float(cantidad)
            restante -= precio * cantidad
        qty += tomada
        qty_por_distancia += tomada * (float(precio) - best)
        if restante == 0.0:
            break

    # vwap = nocional llenado / cantidad llenada, escrito como
    # best + sum(q_i * (p_i - best)) / sum(q_i). Es la misma cantidad, pero un
    # llenado en el primer nivel da vwap == best exacto y no a un ulp de
    # distancia: (n / p) * p no siempre devuelve n.
    desvio = qty_por_distancia / qty
    half_spread_rel = abs(best - mid) / mid
    depth_impact_rel = abs(desvio) / mid
    return BookWalk(
        side=side,
        notional=notional,
        mid=mid,
        best=best,
        vwap=best + desvio,
        filled_notional=notional - restante,
        levels_used=usados,
        exhausted=restante > 0.0,
        half_spread_rel=half_spread_rel,
        depth_impact_rel=depth_impact_rel,
        real_rel=half_spread_rel + depth_impact_rel,
    )


def model_cost(
    bars: BarSeries,
    *,
    mid: float,
    notional: float,
    as_of: np.datetime64,
    spread: SpreadModel,
    slippage: SlippageModel,
) -> ModelCost:
    """Lo que habria cobrado el simulador por ``notional`` USDT a precio ``mid``.

    ``bars`` son las barras cerradas hasta ``as_of``. El ``timestamp`` de una
    barra es su cierre, asi que una barra con ``timestamp > as_of`` aun no
    existe: se lanza error y nunca se recorta. Tampoco se pasan menos de dos
    barras, que Corwin-Schultz necesita.

    El contexto de spread lleva todas las barras recibidas, como
    ``Simulator._spread_context``; Corwin-Schultz solo mira las dos ultimas.

    ASSUMPTION: el volumen del slippage es el de la ultima barra cerrada (el de
    la barra de ejecucion no existe en vivo); con volumen 0 la participacion es
    0, como en el motor.
    """
    _check_notional(notional)
    if not math.isfinite(mid) or mid <= 0:
        raise ShadowCostError(f"mid debe ser finito y > 0, es {mid!r}")
    corte = np.datetime64(as_of, "ns")
    futuras = int((bars.timestamp > corte).sum())
    if futuras:
        raise ShadowCostError(
            f"lookahead: {futuras} barra(s) con timestamp (cierre) posterior a "
            f"as_of={corte}; la ultima cierra en {bars.timestamp[-1]}"
        )
    if len(bars) < 2:
        raise ShadowCostError(
            f"se necesitan >= 2 barras cerradas para el spread, hay {len(bars)}"
        )

    ctx = SpreadContext(ref_price=mid, high=bars.high, low=bars.low, close=bars.close)
    half_spread_rel = spread.half_spread(ctx) / mid
    volumen = float(bars.volume[-1])
    participation = (notional / mid) / volumen if volumen > 0 else 0.0
    slippage_rel = slippage.impact(participation)
    if half_spread_rel < 0 or slippage_rel < 0:
        raise ShadowCostError(
            "spread y slippage deben ser no negativos: un costo nunca mejora el precio"
        )
    return ModelCost(
        notional=notional,
        mid=mid,
        participation=participation,
        half_spread_rel=half_spread_rel,
        slippage_rel=slippage_rel,
        model_rel=half_spread_rel + slippage_rel,
    )


def cost_ratios(real: BookWalk, model: ModelCost) -> CostRatios:
    """``real / modelo``, sin comision. Un libro agotado no entra a los cocientes."""
    if real.notional != model.notional or real.mid != model.mid:
        raise ShadowCostError(
            "real y modelo miden ordenes distintas: "
            f"notional {real.notional} vs {model.notional}, "
            f"mid {real.mid} vs {model.mid}"
        )
    if real.exhausted:
        return CostRatios(ratio_total=None, ratio_spread=None)
    return CostRatios(
        ratio_total=_cociente(real.real_rel, model.model_rel),
        ratio_spread=_cociente(real.half_spread_rel, model.half_spread_rel),
    )


def _cociente(numerador: float, denominador: float) -> float | None:
    return numerador / denominador if denominador > 0 else None
