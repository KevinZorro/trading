"""walk_book, model_cost y cost_ratios, con oraculos calculados a mano.

Los libros chicos se derivan de ``depth.json``: los precios son los de los tres
primeros niveles de la grabacion real y las cantidades se reemplazan para que
la cuenta se haga a mano. Los oraculos usan ``Fraction`` sobre los decimales
escritos aca, nunca la implementacion.
"""

from __future__ import annotations

import math
from fractions import Fraction as F
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agents.walkforward import WalkForwardConfig
from data.instruments import binance_spot_spec
from data.loaders import bars_from_frame
from data.schema import BarSeries
from live.marketdata import BookSnapshot
from live.shadow_cost import (
    BookWalk,
    ShadowCostError,
    cost_ratios,
    model_cost,
    walk_book,
)
from sim.costs import CorwinSchultzSpread, SpreadContext, SqrtSlippage

from .conftest import cliente, grabacion

SIM = WalkForwardConfig().sim_config("principal")
RECIBIDO = np.datetime64("2026-10-17T17:17:09", "ns")
MID = F("81261.015")  # (81261.01 + 81261.02) / 2


def libro_chico(cant_bids: list[str], cant_asks: list[str]) -> BookSnapshot:
    """Tres niveles con los precios reales del tope de ``depth.json``."""
    crudo = grabacion("depth.json")
    bids = [[p, q] for (p, _), q in zip(crudo["bids"][:3], cant_bids, strict=True)]
    asks = [[p, q] for (p, _), q in zip(crudo["asks"][:3], cant_asks, strict=True)]
    assert [p for p, _ in bids] == [
        "81261.01000000",
        "81261.00000000",
        "81260.99000000",
    ]
    assert [p for p, _ in asks] == [
        "81261.02000000",
        "81261.03000000",
        "81261.04000000",
    ]
    return BookSnapshot(
        received_at=RECIBIDO,
        last_update_id=1,
        bids=np.array(bids, dtype=float),
        asks=np.array(asks, dtype=float),
    )


# Asks: 81261.02 x 0.001 (81.26102 USDT), 81261.03 x 0.002, 81261.04 x 0.001.
# Bids: mismas cantidades. Las cuentas de abajo son sobre este libro.
LIBRO = libro_chico(["0.001", "0.002", "0.001"], ["0.001", "0.002", "0.001"])
TOTAL_ASKS = (
    F("81261.02") * F("0.001") + F("81261.03") * F("0.002") + F("81261.04") * F("0.001")
)  # 325.04412 USDT


class TestWalkBookCompra:
    def test_todo_cabe_en_el_primer_nivel(self) -> None:
        w = walk_book(LIBRO, "buy", 10.0)
        assert w.levels_used == 1
        assert not w.exhausted
        assert w.filled_notional == 10.0
        assert w.vwap == 81261.02
        assert w.depth_impact_rel == 0.0  # exacto, no aproximado
        assert w.half_spread_rel == pytest.approx(float(F("0.005") / MID), rel=1e-12)
        assert w.real_rel == w.half_spread_rel
        assert w.best == 81261.02
        assert w.mid == pytest.approx(81261.015, rel=1e-15)

    def test_ultimo_nivel_parcial(self) -> None:
        # Nivel 1 se consume entero: 0.001 x 81261.02 = 81.26102 USDT.
        # Quedan 100 - 81.26102 = 18.73898 USDT sobre 81261.03.
        q2 = F("18.73898") / F("81261.03")
        vwap = F(100) / (F("0.001") + q2)
        w = walk_book(LIBRO, "buy", 100.0)
        assert w.levels_used == 2
        assert not w.exhausted
        assert w.filled_notional == 100.0
        assert w.vwap == pytest.approx(float(vwap), rel=1e-13)
        assert w.depth_impact_rel == pytest.approx(
            float((vwap - F("81261.02")) / MID), rel=1e-8
        )
        assert w.depth_impact_rel > 0
        assert w.real_rel == w.half_spread_rel + w.depth_impact_rel

    def test_agotado(self) -> None:
        w = walk_book(LIBRO, "buy", 1000.0)
        assert w.exhausted
        assert w.levels_used == 3
        assert w.filled_notional == pytest.approx(float(TOTAL_ASKS), rel=1e-13)
        assert w.filled_notional < 1000.0
        # Costo de lo llenado: 0.004 BTC a vwap = 325.04412 / 0.004.
        vwap = TOTAL_ASKS / F("0.004")
        assert w.vwap == pytest.approx(float(vwap), rel=1e-13)
        assert w.depth_impact_rel == pytest.approx(
            float((vwap - F("81261.02")) / MID), rel=1e-8
        )

    def test_el_costo_crece_con_el_tamano(self) -> None:
        impactos = [walk_book(LIBRO, "buy", n).depth_impact_rel for n in (10, 100, 200)]
        assert impactos[0] == 0.0
        assert 0.0 < impactos[1] < impactos[2]


class TestWalkBookVenta:
    def test_ultimo_nivel_parcial(self) -> None:
        # Nivel 1: 0.001 x 81261.01 = 81.26101 USDT. Quedan 18.73899 sobre 81261.00.
        q2 = F("18.73899") / F("81261.00")
        vwap = F(100) / (F("0.001") + q2)
        w = walk_book(LIBRO, "sell", 100.0)
        assert w.levels_used == 2
        assert w.best == 81261.01
        assert w.vwap == pytest.approx(float(vwap), rel=1e-13)
        assert w.depth_impact_rel == pytest.approx(
            float((F("81261.01") - vwap) / MID), rel=1e-8
        )
        assert w.half_spread_rel == pytest.approx(float(F("0.005") / MID), rel=1e-12)

    @pytest.mark.parametrize("notional", [10.0, 100.0, 200.0])
    def test_libro_espejado_compra_y_venta_cuestan_lo_mismo(
        self, notional: float
    ) -> None:
        # Las distancias de precio al mejor nivel son exactamente simetricas
        # (0, 0.01, 0.02). La cantidad llenada depende del nocional / precio,
        # que difiere ~1e-7 entre lados: simetria a primer orden, no exacta.
        compra = walk_book(LIBRO, "buy", notional)
        venta = walk_book(LIBRO, "sell", notional)
        assert compra.levels_used == venta.levels_used
        assert compra.half_spread_rel == venta.half_spread_rel
        assert compra.depth_impact_rel == pytest.approx(
            venta.depth_impact_rel, rel=1e-5, abs=0.0
        )
        assert compra.real_rel == pytest.approx(venta.real_rel, rel=1e-5)


class TestWalkBookGrabacionReal:
    @pytest.mark.parametrize("side", ["buy", "sell"])
    def test_10_100_y_1000_no_se_agotan_y_el_impacto_no_decrece(
        self, side: Any
    ) -> None:
        libro = cliente().depth("BTCUSDT", 1000)
        assert len(libro.asks) == len(libro.bids) == 1000
        walks = [walk_book(libro, side, n) for n in (10.0, 100.0, 1000.0)]
        assert not any(w.exhausted for w in walks)
        assert all(w.filled_notional == w.notional for w in walks)
        impactos = [w.depth_impact_rel for w in walks]
        assert impactos == sorted(impactos)
        # El tope de esta foto tiene 2.1 BTC (bid) y 8.3 BTC (ask): 1000 USDT
        # caben en el primer nivel, asi que el impacto es 0 exacto.
        assert all(w.levels_used == 1 and w.depth_impact_rel == 0.0 for w in walks)


class TestValidacion:
    @pytest.mark.parametrize("notional", [0.0, -5.0, math.nan, math.inf, -math.inf])
    def test_notional_invalido(self, notional: float) -> None:
        with pytest.raises(ShadowCostError, match="notional debe ser finito y > 0"):
            walk_book(LIBRO, "buy", notional)

    def test_side_invalido(self) -> None:
        with pytest.raises(ShadowCostError, match="side debe ser 'buy' o 'sell'"):
            walk_book(LIBRO, "hold", 10.0)  # type: ignore[arg-type]


def barras(
    rangos: list[tuple[float, float]], volumen: float, inicio: str = "2026-10-15"
) -> BarSeries:
    """Barras diarias con high/low dados. El timestamp es el cierre (21:00Z)."""
    n = len(rangos)
    cierre = [(h + lo) / 2 for h, lo in rangos]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(f"{inicio}T21:00:00Z", periods=n, freq="1D"),
            "open": cierre,
            "high": [h for h, _ in rangos],
            "low": [lo for _, lo in rangos],
            "close": cierre,
            "volume": [volumen] * n,
            "symbol": "BTCUSDT",
        }
    )
    return bars_from_frame(frame, instrument=binance_spot_spec(), freq="1D")


AS_OF = np.datetime64("2026-10-16T21:00:30", "ns")  # 30 s tras el cierre de la ultima


class TestModelCost:
    def test_corwin_schultz_a_mano(self) -> None:
        # Dos barras iguales (h=101, l=100): beta = 2 L^2, gamma = L^2 con
        # L = ln(1.01). Entonces alpha = L (la cuenta se simplifica porque
        # (2 - sqrt2) / (3 - 2 sqrt2) - 1 / (sqrt2 - 1) = 1), e^alpha = 1.01 y
        # S = 2 (1.01 - 1) / (1.01 + 1) = 2/201. Medio spread relativo = 1/201.
        # Participacion: (1000 / 100) / 1000 = 0.01; sqrt = 0.1.
        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=1000.0),
            mid=100.0,
            notional=1000.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        assert m.participation == pytest.approx(0.01, rel=1e-12)
        assert m.half_spread_rel == pytest.approx(1 / 201, rel=1e-12)
        k = WalkForwardConfig().slippage_k
        assert m.slippage_rel == pytest.approx(k * 0.1, rel=1e-12)
        assert m.model_rel == m.half_spread_rel + m.slippage_rel

    def test_el_tope_en_bps_se_aplica(self) -> None:
        # h=110, l=100: S = 2/21 = 953 bps, por encima del tope de la config.
        m = model_cost(
            barras([(110.0, 100.0)] * 2, volumen=1000.0),
            mid=100.0,
            notional=1.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        assert m.half_spread_rel == pytest.approx(
            WalkForwardConfig().spread_max_bps / 2e4, rel=1e-12
        )

    def test_corwin_schultz_trunca_a_cero_ante_un_gap(self) -> None:
        # Barra 2 entera por encima de la 1: gamma domina y alpha < 0.
        m = model_cost(
            barras([(101.0, 100.0), (111.0, 110.0)], volumen=1000.0),
            mid=100.0,
            notional=1000.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        assert m.half_spread_rel == 0.0
        assert m.slippage_rel > 0.0
        real = walk_book(LIBRO, "buy", 10.0)
        asi = model_cost(
            barras([(101.0, 100.0), (111.0, 110.0)], volumen=1000.0),
            mid=real.mid,
            notional=10.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        r = cost_ratios(real, asi)
        assert r.ratio_spread is None  # el ADR: pasa en el 34 % de las barras de C3
        assert r.ratio_total is not None

    def test_usa_sim_costs_y_no_una_copia(self) -> None:
        series = barras([(105.0, 99.0), (108.0, 101.0), (107.0, 100.0)], volumen=500.0)
        mid, notional = 103.0, 250.0
        m = model_cost(
            series,
            mid=mid,
            notional=notional,
            as_of=AS_OF + np.timedelta64(1, "D"),  # 3 barras: la ultima cierra el 17
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        cs = CorwinSchultzSpread(max_bps=WalkForwardConfig().spread_max_bps)
        ctx = SpreadContext(
            ref_price=mid, high=series.high, low=series.low, close=series.close
        )
        sq = SqrtSlippage(k=WalkForwardConfig().slippage_k)
        assert m.half_spread_rel == cs.half_spread(ctx) / mid
        assert m.slippage_rel == sq.impact((notional / mid) / 500.0)

    def test_los_modelos_se_inyectan(self) -> None:
        class Spread:
            name = "fijo"

            def half_spread(self, ctx: SpreadContext) -> float:
                return 2.0  # monetario: 2.0 / mid

        class Slip:
            name = "fijo"

            def impact(self, participation: float) -> float:
                return 0.25

        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=1000.0),
            mid=50.0,
            notional=10.0,
            as_of=AS_OF,
            spread=Spread(),
            slippage=Slip(),
        )
        assert m.half_spread_rel == 2.0 / 50.0
        assert m.slippage_rel == 0.25

    def test_volumen_cero_da_participacion_cero(self) -> None:
        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=0.0),
            mid=100.0,
            notional=10.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        assert m.participation == 0.0
        assert m.slippage_rel == 0.0

    def test_mid_invalido(self) -> None:
        with pytest.raises(ShadowCostError, match="mid debe ser finito y > 0"):
            model_cost(
                barras([(101.0, 100.0)] * 2, volumen=1.0),
                mid=0.0,
                notional=10.0,
                as_of=AS_OF,
                spread=SIM.spread,
                slippage=SIM.slippage,
            )

    def test_notional_invalido(self) -> None:
        with pytest.raises(ShadowCostError, match="notional debe ser finito y > 0"):
            model_cost(
                barras([(101.0, 100.0)] * 2, volumen=1.0),
                mid=100.0,
                notional=-1.0,
                as_of=AS_OF,
                spread=SIM.spread,
                slippage=SIM.slippage,
            )

    def test_un_modelo_que_mejora_el_precio_se_rechaza(self) -> None:
        class Malo:
            name = "malo"

            def half_spread(self, ctx: SpreadContext) -> float:
                return -1.0

        with pytest.raises(ShadowCostError, match="no negativos"):
            model_cost(
                barras([(101.0, 100.0)] * 2, volumen=1.0),
                mid=100.0,
                notional=10.0,
                as_of=AS_OF,
                spread=Malo(),
                slippage=SIM.slippage,
            )


class TestCostRatios:
    def test_cocientes_a_mano(self) -> None:
        # Real a 10 USDT: sin impacto de profundidad, real_rel = 0.005 / mid.
        # Modelo: half = 1/201; participacion = (10 / mid) / 1000.
        real = walk_book(LIBRO, "buy", 10.0)
        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=1000.0),
            mid=real.mid,
            notional=10.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        slip = WalkForwardConfig().slippage_k * math.sqrt(10 / 81261.015 / 1000)
        real_rel = 0.005 / 81261.015
        r = cost_ratios(real, m)
        assert r.ratio_spread == pytest.approx(real_rel * 201, rel=1e-9)
        assert r.ratio_total == pytest.approx(real_rel / (1 / 201 + slip), rel=1e-9)

    def test_denominador_cero_es_none_no_inf(self) -> None:
        real = walk_book(LIBRO, "buy", 10.0)
        m = model_cost(
            barras([(100.0, 100.0)] * 2, volumen=1000.0),
            mid=real.mid,
            notional=10.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        assert m.half_spread_rel == 0.0  # alpha = 0 y el spread es 0
        r = cost_ratios(real, m)
        assert r.ratio_spread is None
        assert r.ratio_total is not None  # el slippage mantiene model_rel > 0

        sin_costo = model_cost(
            barras([(100.0, 100.0)] * 2, volumen=1000.0),
            mid=real.mid,
            notional=10.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=_SinSlippage(),
        )
        assert cost_ratios(real, sin_costo).ratio_total is None

    def test_libro_agotado_no_entra_a_los_cocientes(self) -> None:
        real = walk_book(LIBRO, "buy", 1000.0)
        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=1000.0),
            mid=real.mid,
            notional=1000.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        r = cost_ratios(real, m)
        assert (r.ratio_total, r.ratio_spread) == (None, None)

    def test_ordenes_distintas_se_rechazan(self) -> None:
        real: BookWalk = walk_book(LIBRO, "buy", 10.0)
        m = model_cost(
            barras([(101.0, 100.0)] * 2, volumen=1000.0),
            mid=real.mid,
            notional=100.0,
            as_of=AS_OF,
            spread=SIM.spread,
            slippage=SIM.slippage,
        )
        with pytest.raises(ShadowCostError, match="ordenes distintas"):
            cost_ratios(real, m)


class _SinSlippage:
    name = "ninguno"

    def impact(self, participation: float) -> float:
        return 0.0
