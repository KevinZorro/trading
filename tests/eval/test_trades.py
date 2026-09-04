"""Round-trips y estadisticas de trade, con fills construidos a mano.

El P&L esperado de cada caso se calcula aqui como efectivo contra efectivo:

    cost_basis = sum(qty * fill_price + comision)   sobre las compras
    proceeds   = sum(qty * fill_price - comision)   sobre las ventas
    pnl        = proceeds - cost_basis

Ninguna cifra sale de llamar a la implementacion.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.trades import (
    POSITION_TOL,
    RoundTripLog,
    round_trips,
    trade_stats,
)
from sim.orders import Fill, OrderStatus, RejectReason

TS = np.datetime64("2020-01-01T00:00:00", "ns")


def hacer_fill(
    *,
    t: int,
    qty: float,
    price: float,
    commission: float = 0.0,
    status: OrderStatus = OrderStatus.FILLED,
    reject_reason: RejectReason | None = None,
) -> Fill:
    """Fill minimo. Solo importan qty_filled, fill_price, commission y t_fill."""
    marca = TS + np.timedelta64(t, "D")
    return Fill(
        t_decision=max(t - 1, 0),
        t_fill=t,
        timestamp_decision=marca,
        timestamp_fill=marca,
        symbol="TEST",
        status=status,
        qty_requested=qty,
        qty_filled=qty if status in (OrderStatus.FILLED, OrderStatus.PARTIAL) else 0.0,
        decision_price=price,
        ref_price=price,
        fill_price=price,
        gap=0.0,
        spread_cost=0.0,
        slippage_cost=0.0,
        commission=commission,
        participation=0.0,
        reject_reason=reject_reason,
    )


class TestCeroTrades:
    def test_sin_fills(self) -> None:
        log = round_trips([])
        assert log.closed == []
        assert log.open_position is None

    def test_estadisticas_sin_trades_no_son_cero_sino_indefinidas(self) -> None:
        """ "No hizo ningun trade" no es lo mismo que "perdio todos"."""
        stats = trade_stats(RoundTripLog())
        assert stats.n_round_trips == 0
        assert stats.win_rate is None
        assert stats.profit_factor is None
        assert stats.avg_bars_held is None
        assert stats.net_pnl == 0.0

    def test_solo_rechazos_no_produce_trades(self) -> None:
        rechazo = hacer_fill(
            t=1,
            qty=10.0,
            price=10.0,
            status=OrderStatus.REJECTED,
            reject_reason=RejectReason.INSUFFICIENT_CASH,
        )
        expirada = hacer_fill(t=5, qty=10.0, price=10.0, status=OrderStatus.EXPIRED)
        log = round_trips([rechazo, expirada])
        assert log.closed == []
        assert log.open_position is None


class TestUnTrade:
    def test_ganador(self) -> None:
        # Compra: 10 * 10.00 + 1.00 = 101.00 desembolsados.
        # Venta:  10 * 12.00 - 1.00 = 119.00 recibidos.
        # P&L = 119.00 - 101.00 = 18.00
        log = round_trips(
            [
                hacer_fill(t=1, qty=10.0, price=10.0, commission=1.0),
                hacer_fill(t=4, qty=-10.0, price=12.0, commission=1.0),
            ]
        )
        assert len(log.closed) == 1
        trip = log.closed[0]
        assert trip.cost_basis == pytest.approx(101.0)
        assert trip.proceeds == pytest.approx(119.0)
        assert trip.pnl == pytest.approx(18.0)
        assert trip.return_pct == pytest.approx(18.0 / 101.0)
        assert trip.commission == pytest.approx(2.0)
        assert trip.bars_held == 3
        assert trip.n_fills == 2
        assert trip.is_win
        assert not trip.is_loss

    def test_perdedor(self) -> None:
        # Compra: 10 * 10.00 = 100.00. Venta: 10 * 9.00 = 90.00. P&L = -10.00
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=2, qty=-10.0, price=9.0),
            ]
        )
        trip = log.closed[0]
        assert trip.pnl == pytest.approx(-10.0)
        assert trip.is_loss

    def test_la_comision_puede_dar_vuelta_el_signo(self) -> None:
        """Precio favorable, resultado perdedor. Es el caso del capital bajo."""
        # Compra: 10 * 10.00 + 1.00 = 101.00. Venta: 10 * 10.05 - 1.00 = 99.50.
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0, commission=1.0),
                hacer_fill(t=1, qty=-10.0, price=10.05, commission=1.0),
            ]
        )
        trip = log.closed[0]
        assert trip.pnl == pytest.approx(-1.5)
        assert trip.is_loss

    def test_scratch_no_es_ni_ganado_ni_perdido(self) -> None:
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-10.0, price=10.0),
            ]
        )
        stats = trade_stats(log)
        assert stats.n_round_trips == 1
        assert stats.n_wins == 0
        assert stats.n_losses == 0
        assert stats.n_scratches == 1
        assert stats.win_rate == pytest.approx(0.0)
        assert stats.profit_factor is None


class TestFlatToFlat:
    def test_varias_compras_y_una_venta_son_un_solo_trade(self) -> None:
        # 5*10 + 5*12 = 110 desembolsados. 10*13 = 130 recibidos. P&L = 20.
        log = round_trips(
            [
                hacer_fill(t=0, qty=5.0, price=10.0),
                hacer_fill(t=1, qty=5.0, price=12.0),
                hacer_fill(t=5, qty=-10.0, price=13.0),
            ]
        )
        assert len(log.closed) == 1
        assert log.closed[0].pnl == pytest.approx(20.0)
        assert log.closed[0].n_fills == 3
        assert log.closed[0].bars_held == 5

    def test_venta_parcial_y_recompra_siguen_siendo_un_trade(self) -> None:
        """El ciclo va de cero a cero, no de compra a venta."""
        # Compras: 10*10 + 5*11 = 155. Ventas: 5*12 + 10*13 = 190. P&L = 35.
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-5.0, price=12.0),
                hacer_fill(t=2, qty=5.0, price=11.0),
                hacer_fill(t=3, qty=-10.0, price=13.0),
            ]
        )
        assert len(log.closed) == 1
        assert log.closed[0].pnl == pytest.approx(35.0)
        assert log.closed[0].qty_bought == pytest.approx(15.0)
        assert log.closed[0].qty_sold == pytest.approx(15.0)

    def test_dos_ciclos_son_dos_trades(self) -> None:
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-10.0, price=11.0),  # +10
                hacer_fill(t=2, qty=10.0, price=11.0),
                hacer_fill(t=3, qty=-10.0, price=10.0),  # -10
            ]
        )
        assert len(log.closed) == 2
        assert log.closed[0].pnl == pytest.approx(10.0)
        assert log.closed[1].pnl == pytest.approx(-10.0)

    def test_residuo_de_coma_flotante_cierra_el_trade(self) -> None:
        """0.1 + 0.2 - 0.3 no es cero en float64. El trade igual esta cerrado."""
        log = round_trips(
            [
                hacer_fill(t=0, qty=0.1, price=100.0),
                hacer_fill(t=1, qty=0.2, price=100.0),
                hacer_fill(t=2, qty=-0.3, price=110.0),
            ]
        )
        assert 0.0 < abs(0.1 + 0.2 - 0.3) < POSITION_TOL
        assert len(log.closed) == 1
        assert log.open_position is None


class TestPosicionAbierta:
    def test_no_se_cierra_ni_entra_al_win_rate(self) -> None:
        log = round_trips([hacer_fill(t=0, qty=10.0, price=10.0, commission=1.0)])
        assert log.closed == []
        assert log.open_position is not None
        assert log.open_position.qty == pytest.approx(10.0)
        assert log.open_position.cost_basis == pytest.approx(101.0)

        stats = trade_stats(log)
        assert stats.n_round_trips == 0
        assert stats.win_rate is None
        assert stats.open_position is log.open_position

    def test_una_perdedora_sin_cerrar_no_mejora_el_win_rate(self) -> None:
        """El sesgo que la decision de diseno evita, hecho test."""
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-10.0, price=11.0),  # ganadora, cerrada
                hacer_fill(t=2, qty=10.0, price=20.0),  # perdedora, sin cerrar
            ]
        )
        stats = trade_stats(log)
        assert stats.n_round_trips == 1
        assert stats.win_rate == pytest.approx(1.0)
        assert stats.open_position is not None


class TestEstadisticas:
    def _log(self) -> RoundTripLog:
        # +10, -4, +6  ->  bruto ganado 16, bruto perdido 4
        return round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-10.0, price=11.0),
                hacer_fill(t=2, qty=10.0, price=11.0),
                hacer_fill(t=3, qty=-10.0, price=10.6),
                hacer_fill(t=4, qty=10.0, price=10.0),
                hacer_fill(t=6, qty=-10.0, price=10.6),
            ]
        )

    def test_conteos_y_ratios(self) -> None:
        stats = trade_stats(self._log())
        assert stats.n_round_trips == 3
        assert stats.n_wins == 2
        assert stats.n_losses == 1
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.gross_profit == pytest.approx(16.0)
        assert stats.gross_loss == pytest.approx(4.0)
        assert stats.profit_factor == pytest.approx(4.0)
        assert stats.net_pnl == pytest.approx(12.0)
        assert stats.avg_win == pytest.approx(8.0)
        assert stats.avg_loss == pytest.approx(4.0)
        # Barras en posicion: 1, 1 y 2.
        assert stats.avg_bars_held == pytest.approx(4 / 3)

    def test_profit_factor_infinito_sin_perdedoras(self) -> None:
        log = round_trips(
            [
                hacer_fill(t=0, qty=10.0, price=10.0),
                hacer_fill(t=1, qty=-10.0, price=11.0),
            ]
        )
        assert trade_stats(log).profit_factor == float("inf")
