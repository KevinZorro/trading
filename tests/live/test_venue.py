"""ShadowVenue: acuse, ejecucion contra el libro, rechazos, persistencia y replay.

Los libros de los oraculos son chicos y armados a mano para que el resultado se
calcule con papel. El `InstrumentSpec` sale del exchangeInfo grabado
(5 USDT de nocional minimo, lote 1e-5, comision 10 bps), no de los defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from data.instruments import InstrumentSpec
from live.fillstore import FillStore, venue_header
from live.marketdata import BookSnapshot, MarketDataError
from live.store import StoreError
from live.venue import ShadowVenue
from sim.clock import SimulatedClock
from sim.orders import MarketOrder, OrderStatus, RejectReason
from sim.venue import ExecutionVenue

from .conftest import cliente, recibido

T0 = np.datetime64("2026-10-18T00:00:30", "ns")
SPEC_SHA = "a" * 64


def spec() -> InstrumentSpec:
    return cliente().instrument("BTCUSDT")


def libro(
    bids: list[list[float]], asks: list[list[float]], cuando: str = "00:00:31"
) -> BookSnapshot:
    return BookSnapshot(
        received_at=np.datetime64(f"2026-10-18T{cuando}", "ns"),
        last_update_id=1,
        bids=np.array(bids),
        asks=np.array(asks),
    )


# mid = 100.5, medio spread 0.5
LIBRO = libro([[100.0, 1.0], [99.0, 2.0]], [[101.0, 1.0], [103.0, 2.0]])


class Mercado:
    """Sirve los libros en orden; una excepcion en la cola se lanza al pedirla."""

    def __init__(self, *libros: BookSnapshot | Exception) -> None:
        self.cola = list(libros)

    def depth(self, symbol: str, limit: int) -> BookSnapshot:
        x = self.cola.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    def closed_klines(self, symbol: str, interval: str, n: int) -> object:
        raise AssertionError("el venue no pide klines")

    def instrument(self, symbol: str) -> InstrumentSpec:
        raise AssertionError("el spec entra por el constructor")


def armar(
    tmp: Path,
    *libros: BookSnapshot | Exception,
    cash: float = 1000.0,
    inicio: np.datetime64 = T0,
) -> tuple[ShadowVenue, Mercado, SimulatedClock]:
    mercado = Mercado(*libros)
    reloj = SimulatedClock(inicio)
    cabecera = venue_header(
        initial_cash=cash, spec_sha256=SPEC_SHA, spec_received_at=recibido("depth.json")
    )
    venue = ShadowVenue(
        market=mercado,  # type: ignore[arg-type]
        instrument=spec(),
        clock=reloj,
        store=FillStore(tmp, cabecera),
        initial_cash=cash,
    )
    return venue, mercado, reloj


def orden(qty: float, id_: str = "o1") -> MarketOrder:
    return MarketOrder(qty=qty, tag="t", client_order_id=id_)


def test_cumple_el_protocolo(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path)
    assert isinstance(venue, ExecutionVenue)


def test_submit_acusa_recepcion_y_no_ejecuta(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    ack = venue.submit_order(orden(2.0))
    assert ack.accepted
    assert not ack.is_duplicate
    assert venue.get_fills() == []
    assert venue.get_positions() == {}
    assert venue.reconcile().open_order_ids == ("o1",)


def test_sin_client_order_id_se_rechaza(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path)
    ack = venue.submit_order(MarketOrder(qty=1.0))
    assert not ack.accepted
    assert ack.reject_reason is RejectReason.MISSING_CLIENT_ORDER_ID
    assert venue.reconcile().open_order_ids == ()


def test_compra_recorre_el_libro_con_oraculo_a_mano(tmp_path: Path) -> None:
    venue, _, reloj = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    reloj.advance_to(np.datetime64("2026-10-18T00:00:31", "ns"))
    [f] = venue.settle()
    # 2 unidades: 1 a 101 y 1 a 103 -> vwap 102; mejor ask 101; mid 100.5
    assert f.status is OrderStatus.FILLED
    assert f.qty_filled == 2.0
    assert f.fill_price == pytest.approx(102.0)
    assert f.spread_cost == pytest.approx(2 * 0.5)  # |101 - 100.5| por unidad
    assert f.slippage_cost == pytest.approx(2 * 1.0)  # |102 - 101| por unidad
    assert f.commission == pytest.approx(2 * 102.0 * 0.001)
    assert f.gap == 0.0  # sin decision_price no se inventa un gap
    # el costo total frente al mid cuadra: qty * (vwap - mid)
    assert f.spread_cost + f.slippage_cost == pytest.approx(2 * (102.0 - 100.5))
    assert f.timestamp_fill == LIBRO.received_at  # hora del libro, no de la barra
    assert f.timestamp_decision == T0
    assert venue.get_positions() == {"BTCUSDT": 2.0}
    assert venue.reconcile().cash == pytest.approx(1000.0 - 204.0 - 0.204)
    assert venue.reconcile().open_order_ids == ()


def test_gap_usa_el_precio_de_decision(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    [f] = venue.settle(decision_price=100.0)
    assert f.gap == pytest.approx(2.0 * (100.5 - 100.0))
    assert f.decision_price == 100.0
    assert f.ref_price == 100.5


def test_venta_llena_contra_los_bids(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO, LIBRO)
    venue.submit_order(orden(2.0, "c"))
    venue.settle()
    venue.submit_order(orden(-1.0, "v"))
    [f] = venue.settle()
    assert f.fill_price == pytest.approx(100.0)  # mejor bid, 1 unidad
    assert f.spread_cost == pytest.approx(0.5)
    assert f.slippage_cost == pytest.approx(0.0)
    assert venue.get_positions() == {"BTCUSDT": 1.0}


def test_libro_agotado_llena_parcial_y_cancela_el_remanente(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(5.0))
    [f] = venue.settle()
    assert f.status is OrderStatus.PARTIAL
    assert f.qty_filled == 3.0
    assert f.qty_requested == 5.0
    assert f.fill_price == pytest.approx((101.0 + 2 * 103.0) / 3)
    assert f.participation == 1.0
    assert venue.reconcile().open_order_ids == ()  # el remanente no se arrastra


def test_rechazo_por_nocional_minimo_queda_en_el_log(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(0.01))  # 1 USDT < 5 USDT del exchangeInfo
    [f] = venue.settle()
    assert f.status is OrderStatus.REJECTED
    assert f.reject_reason is RejectReason.MIN_NOTIONAL
    assert venue.get_positions() == {}
    assert venue.get_fills() == [f]


def test_vender_sin_posicion_y_cash_insuficiente(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO, LIBRO, cash=100.0)
    venue.submit_order(orden(-1.0, "v"))
    venue.submit_order(orden(2.0, "c"))  # cuesta 204.2 con 100 de cash
    corto, caja = venue.settle()
    assert corto.reject_reason is RejectReason.SHORT_NOT_ALLOWED
    assert caja.reject_reason is RejectReason.INSUFFICIENT_CASH
    assert venue.reconcile().cash == 100.0


def test_reenviar_el_id_no_ejecuta_dos_veces(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO, LIBRO)
    venue.submit_order(orden(2.0))
    dup = venue.submit_order(orden(2.0))
    assert dup.accepted
    assert dup.is_duplicate
    assert len(venue.settle()) == 1
    assert venue.submit_order(orden(2.0)).is_duplicate  # tambien ya ejecutada
    assert venue.settle() == []
    assert venue.get_positions() == {"BTCUSDT": 2.0}


def test_cancelar_una_orden_viva_y_una_inexistente(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    assert venue.cancel_order("o1").cancelled
    assert venue.settle() == []
    assert not venue.cancel_order("o1").cancelled
    assert not venue.cancel_order("nada").cancelled


def test_get_fills_since_es_numero_de_secuencia(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO, LIBRO)
    venue.submit_order(orden(1.0, "a"))
    venue.submit_order(orden(1.0, "b"))
    venue.settle()
    assert [f.client_order_id for f in venue.get_fills(1)] == ["b"]
    with pytest.raises(ValueError, match="since"):
        venue.get_fills(-1)


def test_falla_de_red_deja_la_orden_viva_y_reintentar_es_seguro(
    tmp_path: Path,
) -> None:
    venue, _, _ = armar(tmp_path, MarketDataError("timeout"), LIBRO)
    venue.submit_order(orden(2.0))
    with pytest.raises(MarketDataError, match="timeout"):
        venue.settle()
    assert venue.reconcile().open_order_ids == ("o1",)
    assert venue.get_positions() == {}
    assert len(venue.settle()) == 1


def test_si_falla_la_escritura_no_cambia_el_estado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))

    def falla(*_: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(os, "fsync", falla)
    with pytest.raises(OSError, match="disco lleno"):
        venue.settle()
    assert venue.get_positions() == {}
    assert venue.get_fills() == []
    assert venue.reconcile().cash == 1000.0


def test_replay_recupera_cash_posicion_y_fills(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    venue.settle()
    venue.submit_order(orden(1.0, "vivo"))  # acusada, sin fill: el proceso cae
    nuevo, _, _ = armar(tmp_path, LIBRO)
    assert nuevo.get_positions() == venue.get_positions()
    assert nuevo.reconcile().cash == pytest.approx(venue.reconcile().cash)
    assert nuevo.get_fills() == venue.get_fills()
    assert nuevo.reconcile().open_order_ids == ("vivo",)
    assert nuevo.submit_order(orden(2.0)).is_duplicate  # idempotencia sobrevive
    [f] = nuevo.settle()
    assert f.client_order_id == "vivo"
    assert f.t_fill == 2  # la secuencia continua


def test_replay_descarta_la_linea_rota_por_una_caida(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    venue.settle()
    ruta = next(tmp_path.glob("venue-*.jsonl"))
    with open(ruta, "ab") as f:
        f.write(b'{"type": "ack", "client_ord')  # SIGKILL a mitad de linea
    nuevo, _, _ = armar(tmp_path)
    assert nuevo.get_positions() == {"BTCUSDT": 2.0}
    assert nuevo.submit_order(orden(1.0, "otro")).accepted  # el append separa la rota


def test_replay_con_otro_capital_inicial_es_un_error(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(2.0))
    with pytest.raises(StoreError, match="initial_cash"):
        armar(tmp_path, cash=500.0)


def test_cabecera_lleva_el_spec_de_adr_0008(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path)
    venue.submit_order(orden(2.0))
    cab = json.loads(next(tmp_path.glob("venue-*.jsonl")).read_text().splitlines()[0])
    assert cab["instrument_spec_sha256"] == SPEC_SHA
    assert cab["instrument_spec_received_at"].startswith("2026-10-17")


def test_con_el_libro_real_el_costo_cuadra_con_el_mid(tmp_path: Path) -> None:
    real = cliente().depth("BTCUSDT", 1000)
    venue, _, _ = armar(tmp_path, real, cash=100.0, inicio=real.received_at)
    venue.submit_order(orden(0.0005))  # ~40 USDT a ~81 000
    [f] = venue.settle()
    mid = (real.best_bid + real.best_ask) / 2
    assert f.status is OrderStatus.FILLED
    assert f.ref_price == mid
    assert f.spread_cost + f.slippage_cost == pytest.approx(
        f.qty_filled * (f.fill_price - mid)
    )
    assert f.fill_price >= real.best_ask
