"""Venue de testnet contra las respuestas reales grabadas en el 7a.

Ningun test inventa una respuesta de Binance: los casos que no tienen grabacion
(parcial, vencida, comision en otra moneda) se derivan modificando campos de una
real. Los rechazos -1013 y -2010 se mapean a un Fill REJECTED con las grabaciones
de #42; lo que ninguna grabacion cubre (-1022 al enviar, -1100) se propaga.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.instruments import binance_spot_spec
from live.fillstore import FillStore, venue_header
from live.testnet_http import (
    BinanceApiError,
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    HttpResponse,
    TransportError,
)
from live.testnet_venue import BinanceTestnetVenue, TestnetVenueError
from sim.clock import SimulatedClock
from sim.orders import MarketOrder, OrderStatus, RejectReason
from sim.venue import ExecutionVenue

DIR = Path(__file__).parent / "respuestas" / "testnet"
MANIFEST = json.loads((DIR / "manifest.json").read_text())
AHORA = np.datetime64("2026-10-19T03:11:02.000", "ns")  # 805 ms antes de la grabada


def grabacion(nombre: str) -> Any:
    cuerpo = (DIR / nombre).read_bytes()
    assert hashlib.sha256(cuerpo).hexdigest() == MANIFEST["files"][nombre]["sha256"]
    return json.loads(cuerpo)


def ok(datos: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(status, json.dumps(datos).encode())


def error(nombre: str) -> HttpResponse:
    return ok(grabacion(nombre), 400)


class Servidor:
    """Transporte doble: una cola de respuestas por ``(metodo, path)``."""

    def __init__(self, **rutas: list[HttpResponse | Exception]) -> None:
        self.rutas = {tuple(k.split("__", 1)): v for k, v in rutas.items()}
        self.pedidos: list[HttpRequest] = []

    def __call__(self, pedido: HttpRequest) -> HttpResponse:
        self.pedidos.append(pedido)
        metodo = pedido.method
        path = pedido.url.split("testnet.binance.vision")[1].split("?")[0]
        r = self.rutas[(metodo, path)].pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def llamadas(self, metodo: str, path: str) -> int:
        return sum(
            1
            for p in self.pedidos
            if p.method == metodo and p.url.split("?")[0].endswith(path)
        )


ORDER = "/api/v3/order"
POST = "POST__/api/v3/order"
GET = "GET__/api/v3/order"
CUENTA = "GET__/api/v3/account"
ABIERTAS = "GET__/api/v3/openOrders"
MIS_TRADES = "GET__/api/v3/myTrades"


def venue(tmp_path: Path, servidor: Servidor) -> BinanceTestnetVenue:
    clock = SimulatedClock(AHORA)
    cliente = BinanceTestnetClient(
        BinanceTestnetCredentials("K" * 64, "S" * 64), clock, transport=servidor
    )
    return BinanceTestnetVenue(
        client=cliente,
        instrument=binance_spot_spec(),
        base_asset="BTC",
        quote_asset="USDT",
        clock=clock,
        store=FillStore(
            tmp_path,
            venue_header(initial_cash=0.0, spec_sha256="x", spec_received_at=AHORA),
        ),
    )


def compra(id_: str = "rec7a-buy-1792379461", qty: float = 0.001) -> MarketOrder:
    return MarketOrder(qty, "agente", id_)


def modificada(nombre: str, **cambios: Any) -> dict[str, Any]:
    d = copy.deepcopy(grabacion(nombre))
    d.update(cambios)
    return d


def test_satisface_el_protocolo(tmp_path: Path) -> None:
    assert isinstance(venue(tmp_path, Servidor()), ExecutionVenue)


def test_compra_feliz_mapea_la_grabacion(tmp_path: Path) -> None:
    s = Servidor(**{POST: [ok(grabacion("order_buy.json"))]})
    v = venue(tmp_path, s)
    ack = v.submit_order(compra())
    assert ack.accepted
    assert not ack.is_duplicate
    (f,) = v.get_fills()
    # 82.64459 USDT / 0.001 BTC, a mano
    assert f.status is OrderStatus.FILLED
    assert f.qty_filled == 0.001
    assert f.fill_price == pytest.approx(82644.59)
    assert f.spread_cost == 0.0
    assert f.slippage_cost == 0.0
    assert f.gap == 0.0
    assert f.commission == 0.0
    assert f.timestamp_fill == np.datetime64("2026-10-19T03:11:02.805", "ns")
    assert f.client_order_id == "rec7a-buy-1792379461"
    assert f.tag == "agente"
    cuerpo = s.pedidos[0].body.decode()
    for esperado in (
        "side=BUY",
        "type=MARKET",
        "quantity=0.00100000",
        "newClientOrderId=rec7a-buy-1792379461",
        "newOrderRespType=FULL",
    ):
        assert esperado in cuerpo


def test_venta_lleva_signo_negativo(tmp_path: Path) -> None:
    v = venue(tmp_path, Servidor(**{POST: [ok(grabacion("order_sell.json"))]}))
    v.submit_order(compra("rec7a-sell-1792379461", -0.001))
    (f,) = v.get_fills()
    assert f.qty_filled == -0.001
    assert f.fill_price == pytest.approx(82644.58)


def test_id_repetido_no_toca_la_red(tmp_path: Path) -> None:
    s = Servidor(**{POST: [ok(grabacion("order_buy.json"))]})
    v = venue(tmp_path, s)
    v.submit_order(compra())
    otra = v.submit_order(compra())
    assert otra.accepted
    assert otra.is_duplicate
    assert s.llamadas("POST", ORDER) == 1
    assert len(v.get_fills()) == 1


def test_id_repetido_tras_reiniciar_no_toca_la_red(tmp_path: Path) -> None:
    s = Servidor(**{POST: [ok(grabacion("order_buy.json"))]})
    venue(tmp_path, s).submit_order(compra())
    reiniciado = venue(tmp_path, s)
    assert reiniciado.submit_order(compra()).is_duplicate
    assert s.llamadas("POST", ORDER) == 1
    assert len(reiniciado.get_fills()) == 1


def test_timeout_consulta_y_adopta_sin_reenviar(tmp_path: Path) -> None:
    s = Servidor(
        **{
            POST: [TransportError("t")],
            GET: [ok(grabacion("order_query.json"))],
            MIS_TRADES: [ok(grabacion("my_trades.json"))],
        }
    )
    v = venue(tmp_path, s)
    v.submit_order(compra())
    (f,) = v.get_fills()
    assert f.qty_filled == 0.001
    assert f.fill_price == pytest.approx(82644.59)
    assert s.llamadas("POST", ORDER) == 1
    assert "origClientOrderId=rec7a-buy-1792379461" in s.pedidos[1].url
    assert "orderId=4516876" in s.pedidos[2].url


def test_orden_recuperada_toma_la_comision_de_my_trades(tmp_path: Path) -> None:
    trades = grabacion("my_trades.json")
    trades[0]["commission"] = "0.00000100"  # BTC; la grabada vale 0 en testnet
    s = Servidor(
        **{
            POST: [TransportError("t")],
            GET: [ok(grabacion("order_query.json"))],
            MIS_TRADES: [ok(trades)],
        }
    )
    v = venue(tmp_path, s)
    v.submit_order(compra())
    (f,) = v.get_fills()
    assert f.commission == pytest.approx(0.000001 * 82644.59)  # BTC a USDT


def test_timeout_con_orden_inexistente_reenvia_una_vez(tmp_path: Path) -> None:
    s = Servidor(
        **{
            POST: [TransportError("t"), ok(grabacion("order_buy.json"))],
            GET: [error("err_consulta_inexistente.json")],
        }
    )
    v = venue(tmp_path, s)
    v.submit_order(compra())
    assert s.llamadas("POST", ORDER) == 2
    assert len(v.get_fills()) == 1


def test_timeout_y_consulta_caida_deja_la_intencion_pendiente(tmp_path: Path) -> None:
    s = Servidor(**{POST: [TransportError("t")], GET: [TransportError("t")]})
    v = venue(tmp_path, s)
    with pytest.raises(TransportError, match="t"):
        v.submit_order(compra())
    assert v.get_fills() == []
    assert v.submit_order(compra()).is_duplicate
    assert s.llamadas("POST", ORDER) == 1  # el duplicado no reenvio


def test_reconcile_adopta_la_orden_pendiente_tras_reiniciar(tmp_path: Path) -> None:
    caido = Servidor(**{POST: [TransportError("t")], GET: [TransportError("t")]})
    with pytest.raises(TransportError, match="t"):
        venue(tmp_path, caido).submit_order(compra())
    s = Servidor(
        **{
            GET: [ok(grabacion("order_query.json"))],
            MIS_TRADES: [ok(grabacion("my_trades.json"))],
            CUENTA: [ok(grabacion("account_despues.json"))],
            ABIERTAS: [ok(grabacion("open_orders.json"))],
        }
    )
    v = venue(tmp_path, s)
    estado = v.reconcile()
    assert len(v.get_fills()) == 1
    assert s.llamadas("POST", ORDER) == 0
    assert estado.position("BTCUSDT") == pytest.approx(1.001)


def test_reconcile_no_reenvia_una_orden_que_el_venue_no_conoce(tmp_path: Path) -> None:
    caido = Servidor(**{POST: [TransportError("t")], GET: [TransportError("t")]})
    with pytest.raises(TransportError, match="t"):
        venue(tmp_path, caido).submit_order(compra())
    s = Servidor(
        **{
            GET: [error("err_consulta_inexistente.json")],
            CUENTA: [ok(grabacion("account_antes.json"))],
            ABIERTAS: [ok(grabacion("open_orders.json"))],
        }
    )
    v = venue(tmp_path, s)
    v.reconcile()
    (f,) = v.get_fills()
    assert f.status is OrderStatus.REJECTED
    assert f.reject_reason is RejectReason.VENUE_NOT_RUNNING
    assert s.llamadas("POST", ORDER) == 0


def test_reconcile_toma_la_cuenta_del_venue(tmp_path: Path) -> None:
    s = Servidor(
        **{
            CUENTA: [ok(grabacion("account_antes.json"))],
            ABIERTAS: [ok(grabacion("open_orders.json"))],
        }
    )
    estado = venue(tmp_path, s).reconcile()
    # la cuenta de testnet arranca con 1 BTC y 10 000 USDT
    assert estado.cash == 10_000.0
    assert estado.position("BTCUSDT") == 1.0
    assert estado.open_order_ids == ()


def test_get_positions_sale_de_la_cuenta(tmp_path: Path) -> None:
    s = Servidor(**{CUENTA: [ok(grabacion("account_despues.json"))]})
    assert venue(tmp_path, s).get_positions() == {"BTCUSDT": pytest.approx(1.001)}


def test_ordenes_abiertas_se_reportan_por_id(tmp_path: Path) -> None:
    abierta = [{"clientOrderId": "abc-1"}]  # forma minima de la lista grabada
    s = Servidor(
        **{CUENTA: [ok(grabacion("account_antes.json"))], ABIERTAS: [ok(abierta)]}
    )
    assert venue(tmp_path, s).reconcile().open_order_ids == ("abc-1",)


@pytest.mark.parametrize("id_", ["", "con espacio", "x" * 37, "ñ"])
def test_id_invalido_se_rechaza_sin_red(tmp_path: Path, id_: str) -> None:
    s = Servidor()
    ack = venue(tmp_path, s).submit_order(compra(id_))
    assert not ack.accepted
    assert ack.reject_reason in (
        RejectReason.MISSING_CLIENT_ORDER_ID,
        RejectReason.INVALID_CLIENT_ORDER_ID,
    )
    assert s.pedidos == []


def test_id_con_solo_caracteres_prohibidos_es_invalido_no_faltante(
    tmp_path: Path,
) -> None:
    ack = venue(tmp_path, Servidor()).submit_order(compra("a b"))
    assert ack.reject_reason is RejectReason.INVALID_CLIENT_ORDER_ID


def test_cantidad_que_redondea_a_cero_se_registra_sin_enviar(tmp_path: Path) -> None:
    s = Servidor()
    v = venue(tmp_path, s)
    assert v.submit_order(compra(qty=1e-7)).accepted
    (f,) = v.get_fills()
    assert f.reject_reason is RejectReason.ZERO_AFTER_ROUNDING
    assert s.pedidos == []


def test_error_del_venue_se_propaga_y_no_se_reenvia(tmp_path: Path) -> None:
    s = Servidor(**{POST: [error("err_id_invalido.json")]})
    v = venue(tmp_path, s)
    with pytest.raises(BinanceApiError, match="-1100"):
        v.submit_order(compra())
    # reconcile no intenta resolver una orden que el venue rechazo
    s.rutas[("GET", "/api/v3/account")] = [ok(grabacion("account_antes.json"))]
    s.rutas[("GET", "/api/v3/openOrders")] = [ok([])]
    venue(tmp_path, s).reconcile()
    assert s.llamadas("GET", ORDER) == 0
    assert s.llamadas("POST", ORDER) == 1


def test_cancelar_inexistente_no_es_error(tmp_path: Path) -> None:
    s = Servidor(**{"DELETE__/api/v3/order": [error("err_cancelar_inexistente.json")]})
    ack = venue(tmp_path, s).cancel_order("nada-1")
    assert not ack.cancelled
    assert ack.client_order_id == "nada-1"


def test_cancelar_con_otro_error_se_propaga(tmp_path: Path) -> None:
    s = Servidor(**{"DELETE__/api/v3/order": [error("err_firma_invalida.json")]})
    with pytest.raises(BinanceApiError, match="-1022"):
        venue(tmp_path, s).cancel_order("nada-1")


def test_cancelar_exitoso(tmp_path: Path) -> None:
    # ASSUMPTION: respuesta de cancelacion exitosa sin grabacion; solo se mira el 200.
    s = Servidor(**{"DELETE__/api/v3/order": [ok({"status": "CANCELED"})]})
    assert venue(tmp_path, s).cancel_order("abc-1").cancelled


def test_fill_parcial_derivado_de_la_grabacion(tmp_path: Path) -> None:
    r = modificada(
        "order_buy.json",
        status="EXPIRED",
        executedQty="0.00050000",
        cummulativeQuoteQty="41.32229500",
    )
    v = venue(tmp_path, Servidor(**{POST: [ok(r)]}))
    v.submit_order(compra())
    (f,) = v.get_fills()
    assert f.status is OrderStatus.PARTIAL
    assert f.qty_filled == 0.0005
    assert f.fill_price == pytest.approx(82644.59)


def test_orden_vencida_sin_ejecucion_es_rechazo(tmp_path: Path) -> None:
    r = modificada(
        "order_buy.json", status="EXPIRED", executedQty="0.00000000", fills=[]
    )
    v = venue(tmp_path, Servidor(**{POST: [ok(r)]}))
    v.submit_order(compra())
    (f,) = v.get_fills()
    assert f.status is OrderStatus.REJECTED
    assert f.reject_reason is RejectReason.NO_VOLUME


def test_estado_desconocido_se_detiene(tmp_path: Path) -> None:
    r = modificada("order_buy.json", status="NEW", executedQty="0.00000000")
    v = venue(tmp_path, Servidor(**{POST: [ok(r)]}))
    with pytest.raises(TestnetVenueError, match="NEW"):
        v.submit_order(compra())


def test_comision_en_la_moneda_base_se_convierte_a_quote(tmp_path: Path) -> None:
    r = modificada("order_buy.json")
    r["fills"][0]["commission"] = "0.00000100"  # 1e-6 BTC
    v = venue(tmp_path, Servidor(**{POST: [ok(r)]}))
    v.submit_order(compra())
    assert v.get_fills()[0].commission == pytest.approx(1e-6 * 82644.59)


def test_comision_en_otra_moneda_se_detiene(tmp_path: Path) -> None:
    r = modificada("order_buy.json")
    r["fills"][0]["commissionAsset"] = "BNB"
    v = venue(tmp_path, Servidor(**{POST: [ok(r)]}))
    with pytest.raises(TestnetVenueError, match="BNB"):
        v.submit_order(compra())


def test_get_fills_desde_un_indice(tmp_path: Path) -> None:
    v = venue(
        tmp_path,
        Servidor(
            **{
                POST: [
                    ok(grabacion("order_buy.json")),
                    ok(grabacion("order_sell.json")),
                ]
            }
        ),
    )
    v.submit_order(compra())
    v.submit_order(compra("rec7a-sell-1792379461", -0.001))
    assert len(v.get_fills(1)) == 1
    assert v.get_fills(2) == []
    with pytest.raises(ValueError, match="negativo"):
        v.get_fills(-1)


@pytest.mark.parametrize(
    ("grabada", "qty", "motivo"),
    [
        ("err_filtro_lot_size.json", 0.001, RejectReason.MIN_QTY),
        ("err_filtro_notional.json", 0.001, RejectReason.MIN_NOTIONAL),
        ("err_saldo_compra.json", 0.001, RejectReason.INSUFFICIENT_CASH),
        ("err_saldo_venta.json", -0.001, RejectReason.INSUFFICIENT_POSITION),
    ],
)
def test_rechazo_grabado_es_un_fill_rejected_con_motivo(
    tmp_path: Path, grabada: str, qty: float, motivo: RejectReason
) -> None:
    s = Servidor(**{POST: [error(grabada)]})
    v = venue(tmp_path, s)
    assert v.submit_order(compra(qty=qty)).accepted  # el acuse es de admision
    (f,) = v.get_fills()
    assert f.status is OrderStatus.REJECTED
    assert f.reject_reason is motivo
    assert f.qty_filled == 0.0
    assert f.client_order_id == "rec7a-buy-1792379461"
    assert s.llamadas("POST", ORDER) == 1


def test_rechazo_grabado_no_se_reenvia_ni_se_consulta_tras_reiniciar(
    tmp_path: Path,
) -> None:
    s = Servidor(**{POST: [error("err_filtro_notional.json")]})
    v = venue(tmp_path, s)
    v.submit_order(compra())
    s.rutas[("GET", "/api/v3/account")] = [ok(grabacion("account_antes.json"))]
    s.rutas[("GET", "/api/v3/openOrders")] = [ok([])]
    v2 = venue(tmp_path, s)
    v2.reconcile()
    assert [f.reject_reason for f in v2.get_fills()] == [RejectReason.MIN_NOTIONAL]
    assert s.llamadas("POST", ORDER) == 1
    assert s.llamadas("GET", ORDER) == 0
    assert v2.submit_order(compra()).is_duplicate


def test_id_rechazado_por_el_venue_devuelve_el_mismo_acuse_sin_red(
    tmp_path: Path,
) -> None:
    s = Servidor(**{POST: [error("err_saldo_compra.json")]})
    v = venue(tmp_path, s)
    v.submit_order(compra())
    assert v.submit_order(compra()).is_duplicate
    assert s.llamadas("POST", ORDER) == 1


def test_rechazo_con_mensaje_no_grabado_se_propaga(tmp_path: Path) -> None:
    otro = modificada("err_filtro_notional.json", msg="Filter failure: PRICE_FILTER")
    s = Servidor(**{POST: [ok(otro, 400)]})
    with pytest.raises(BinanceApiError, match="-1013"):
        venue(tmp_path, s).submit_order(compra())


def test_firma_invalida_al_enviar_se_propaga_y_no_es_un_rechazo(
    tmp_path: Path,
) -> None:
    s = Servidor(**{POST: [error("err_firma_invalida.json")]})
    v = venue(tmp_path, s)
    with pytest.raises(BinanceApiError, match="-1022"):
        v.submit_order(compra())
    assert v.get_fills() == []
