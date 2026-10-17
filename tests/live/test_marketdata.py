"""MarketData REST contra respuestas reales grabadas de Binance.

Las grabaciones salen de ``scripts/grabar_respuestas_binance.py`` y se verifican
contra su manifest antes de usarse. Ningun test inventa una respuesta: los casos
de error se derivan modificando una grabacion real. Los oraculos se leen del
JSON crudo, no de la implementacion.
"""

from __future__ import annotations

import copy
import io
import json
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.instruments import binance_spot_spec
from live.config import MARKET_DATA_URL
from live.logs import configure_json_logging
from live.marketdata import (
    BinanceRestMarketData,
    BookSnapshot,
    InvalidBookError,
    MarketData,
    MarketDataError,
    assert_market_data_url,
    urllib_fetch,
)
from sim.clock import SystemClock

from .conftest import (
    ARCHIVO_POR_PATH,
    DIA_MS,
    MANIFEST,
    Servidor,
    cliente,
    grabacion,
    recibido,
)


def test_cumple_el_protocolo() -> None:
    md: MarketData = cliente()
    assert isinstance(md, BinanceRestMarketData)


# -- depth ----------------------------------------------------------------


class TestDepth:
    def test_libro_real(self) -> None:
        crudo = grabacion("depth.json")
        libro = cliente().depth("BTCUSDT", 1000)
        assert libro.last_update_id == crudo["lastUpdateId"]
        assert libro.received_at == recibido("depth.json")
        assert libro.bids.shape == (len(crudo["bids"]), 2)
        assert libro.asks.shape == (len(crudo["asks"]), 2)
        assert libro.best_bid == float(crudo["bids"][0][0])
        assert libro.best_ask == float(crudo["asks"][0][0])
        assert libro.best_bid < libro.best_ask
        assert not libro.bids.flags.writeable

    def test_pide_el_limite_y_el_simbolo(self) -> None:
        servidor = Servidor()
        cliente(servidor).depth("BTCUSDT", 1000)
        assert servidor.urls == [
            f"{MARKET_DATA_URL}/api/v3/depth?symbol=BTCUSDT&limit=1000"
        ]

    @staticmethod
    def _con_libro(mutar: Callable[[dict[str, Any]], None]) -> BinanceRestMarketData:
        libro = copy.deepcopy(grabacion("depth.json"))
        mutar(libro)
        return cliente(Servidor(depth=libro))

    def test_libro_cruzado(self) -> None:
        def cruzar(libro: dict[str, Any]) -> None:
            libro["bids"][0][0] = libro["asks"][0][0]

        with pytest.raises(InvalidBookError, match="libro cruzado"):
            self._con_libro(cruzar).depth("BTCUSDT", 1000)

    def test_bids_desordenados(self) -> None:
        def desordenar(libro: dict[str, Any]) -> None:
            libro["bids"][1], libro["bids"][2] = libro["bids"][2], libro["bids"][1]

        with pytest.raises(InvalidBookError, match="bids no estrictamente desc"):
            self._con_libro(desordenar).depth("BTCUSDT", 1000)

    def test_asks_con_nivel_repetido(self) -> None:
        def repetir(libro: dict[str, Any]) -> None:
            libro["asks"][3][0] = libro["asks"][2][0]

        with pytest.raises(InvalidBookError, match="asks no estrictamente asc"):
            self._con_libro(repetir).depth("BTCUSDT", 1000)

    def test_cantidad_cero(self) -> None:
        def vaciar(libro: dict[str, Any]) -> None:
            libro["asks"][5][1] = "0.00000000"

        with pytest.raises(InvalidBookError, match=r"asks nivel 5 .* <= 0"):
            self._con_libro(vaciar).depth("BTCUSDT", 1000)

    def test_lado_vacio(self) -> None:
        def vaciar(libro: dict[str, Any]) -> None:
            libro["bids"] = []

        with pytest.raises(InvalidBookError, match=r"bids debe ser \(n>=1, 2\)"):
            self._con_libro(vaciar).depth("BTCUSDT", 1000)

    def test_precio_no_finito(self) -> None:
        def romper(libro: dict[str, Any]) -> None:
            libro["bids"][0][0] = "nan"

        with pytest.raises(InvalidBookError, match="bids tiene valores no finitos"):
            self._con_libro(romper).depth("BTCUSDT", 1000)

    def test_respuesta_sin_campos(self) -> None:
        def romper(libro: dict[str, Any]) -> None:
            del libro["lastUpdateId"]

        with pytest.raises(MarketDataError, match="depth malformada"):
            self._con_libro(romper).depth("BTCUSDT", 1000)

    @pytest.mark.parametrize("limite", [0, 5001])
    def test_limite_fuera_de_rango(self, limite: int) -> None:
        with pytest.raises(MarketDataError, match=r"fuera de \[1, 5000\]"):
            cliente().depth("BTCUSDT", limite)

    def test_snapshot_es_inmutable(self) -> None:
        crudo = grabacion("depth.json")
        libro = BookSnapshot(
            received_at=recibido("depth.json"),
            last_update_id=1,
            bids=np.array(crudo["bids"][:3], dtype=np.float64),
            asks=np.array(crudo["asks"][:3], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="read-only"):
            libro.asks[0, 0] = 1.0


# -- klines ---------------------------------------------------------------


class TestClosedKlines:
    def test_pide_una_de_mas_por_la_vela_en_curso(self) -> None:
        servidor = Servidor()
        cliente(servidor).closed_klines("BTCUSDT", "1d", 5)
        assert servidor.urls[0].endswith("klines?symbol=BTCUSDT&interval=1d&limit=6")

    def test_menos_cerradas_que_las_pedidas(self) -> None:
        with pytest.raises(MarketDataError, match="se pidieron 30 klines cerradas"):
            cliente(ahora=recibido("klines_1d.json")).closed_klines("BTCUSDT", "1d", 30)

    def test_cierre_anticipado_se_conserva_y_se_avisa(self) -> None:
        klines = copy.deepcopy(grabacion("klines_1d.json"))
        klines[-3][6] -= 3_600_000  # el venue cerro una hora antes (caida)
        md = cliente(Servidor(klines_1d=klines), ahora=recibido("klines_1d.json"))
        salida = io.StringIO()
        configure_json_logging(salida, level=logging.WARNING)
        barras = md.closed_klines("BTCUSDT", "1d", 5)
        assert len(barras) == 5
        evento = json.loads(salida.getvalue())
        assert evento["msg"] == "klines cerradas antes de tiempo por el venue"
        cierre = pd.Timestamp(klines[-3][0] + DIA_MS, unit="ms", tz="UTC")
        assert evento["timestamps"] == [str(cierre)]

    def test_anomalias_estructurales_fallan(self) -> None:
        klines = copy.deepcopy(grabacion("klines_1d.json"))
        for k in klines[:10]:
            k[6] -= 3_600_000
        md = cliente(Servidor(klines_1d=klines), ahora=recibido("klines_1d.json"))
        with pytest.raises(MarketDataError, match="demasiadas"):
            md.closed_klines("BTCUSDT", "1d", 5)

    def test_respuesta_de_error_del_venue(self) -> None:
        error = {"code": -1121, "msg": "Invalid symbol."}
        with pytest.raises(MarketDataError, match="klines inesperada"):
            cliente(Servidor(klines_1d=error)).closed_klines("BTCUSDT", "1d", 5)

    def test_intervalo_no_soportado(self) -> None:
        with pytest.raises(MarketDataError, match="intervalo no soportado"):
            cliente().closed_klines("BTCUSDT", "4h", 5)

    @pytest.mark.parametrize("n", [0, 1000])
    def test_n_fuera_de_rango(self, n: int) -> None:
        with pytest.raises(MarketDataError, match=r"fuera de \[1, 999\]"):
            cliente().closed_klines("BTCUSDT", "1d", n)


# -- exchangeInfo ---------------------------------------------------------


class TestInstrument:
    def test_filtros_reales_contra_binance_spot_spec(self) -> None:
        real = cliente().instrument("BTCUSDT")
        supuesto = binance_spot_spec("BTCUSDT")
        assert real.tick_size == supuesto.tick_size == 0.01
        assert real.lot_size == supuesto.lot_size == 1e-5
        assert real.min_order_qty == supuesto.min_order_qty == 1e-5
        assert real.qty_precision == supuesto.qty_precision == 8
        assert real.commission_schema == supuesto.commission_schema
        assert (real.venue, real.asset_class) == (supuesto.venue, supuesto.asset_class)
        # Divergencia medida: el NOTIONAL vigente es 5 USDT, el supuesto de C3 es 10.
        assert real.min_notional == 5.0
        assert supuesto.min_notional == 10.0

    @staticmethod
    def _con_info(mutar: Callable[[dict[str, Any]], None]) -> BinanceRestMarketData:
        info = copy.deepcopy(grabacion("exchange_info.json"))
        mutar(info["symbols"][0])
        return cliente(Servidor(exchange_info=info))

    def test_min_notional_heredado(self) -> None:
        def heredar(simbolo: dict[str, Any]) -> None:
            for f in simbolo["filters"]:
                if f["filterType"] == "NOTIONAL":
                    f["filterType"] = "MIN_NOTIONAL"

        assert self._con_info(heredar).instrument("BTCUSDT").min_notional == 5.0

    def test_sin_nocional_minimo(self) -> None:
        def quitar(simbolo: dict[str, Any]) -> None:
            simbolo["filters"] = [
                f for f in simbolo["filters"] if f["filterType"] != "NOTIONAL"
            ]

        with pytest.raises(MarketDataError, match="exactamente uno de NOTIONAL"):
            self._con_info(quitar).instrument("BTCUSDT")

    def test_dos_nocionales_es_ambiguo(self) -> None:
        def duplicar(simbolo: dict[str, Any]) -> None:
            simbolo["filters"].append(
                {"filterType": "MIN_NOTIONAL", "minNotional": "10.00000000"}
            )

        with pytest.raises(MarketDataError, match="hay 2"):
            self._con_info(duplicar).instrument("BTCUSDT")

    def test_simbolo_que_no_opera(self) -> None:
        def detener(simbolo: dict[str, Any]) -> None:
            simbolo["status"] = "BREAK"

        with pytest.raises(MarketDataError, match="status=BREAK"):
            self._con_info(detener).instrument("BTCUSDT")

    def test_simbolo_ausente(self) -> None:
        with pytest.raises(MarketDataError, match="0 entradas para ETHUSDT"):
            cliente().instrument("ETHUSDT")

    def test_filtro_faltante(self) -> None:
        def quitar(simbolo: dict[str, Any]) -> None:
            simbolo["filters"] = [
                f for f in simbolo["filters"] if f["filterType"] != "PRICE_FILTER"
            ]

        with pytest.raises(MarketDataError, match="malformado.*PRICE_FILTER"):
            self._con_info(quitar).instrument("BTCUSDT")


# -- lista blanca y transporte --------------------------------------------


class TestListaBlanca:
    @pytest.mark.parametrize("nombre", sorted(ARCHIVO_POR_PATH.values()))
    def test_las_urls_grabadas_pasan(self, nombre: str) -> None:
        url = MANIFEST["files"][nombre]["url"]
        assert assert_market_data_url(url, MARKET_DATA_URL) == url

    @pytest.mark.parametrize(
        ("reemplazo", "motivo"),
        [
            (("/api/v3/depth", "/api/v3/order"), "fuera de la lista blanca"),
            (("/api/v3/depth", "/api/v3/account"), "fuera de la lista blanca"),
            (("/api/v3/depth", "/api/v3/depth/"), "fuera de la lista blanca"),
            (("https://", "http://"), "solo por https"),
            (("data-api", "user:pw@data-api"), "credenciales embebidas"),
            (("binance.vision", "binance.vision.evil.com"), "distinto del config"),
            (("binance.vision", "binance.vision:8443"), "distinto del config"),
        ],
    )
    def test_url_grabada_modificada_se_rechaza(
        self, reemplazo: tuple[str, str], motivo: str
    ) -> None:
        url = MANIFEST["files"]["depth.json"]["url"].replace(*reemplazo)
        with pytest.raises(MarketDataError, match=motivo):
            assert_market_data_url(url, MARKET_DATA_URL)

    @pytest.mark.parametrize(
        "base", ["http://data-api.binance.vision", f"{MARKET_DATA_URL}/api/v3/order"]
    )
    def test_base_url_invalida(self, base: str) -> None:
        with pytest.raises(MarketDataError, match="base_url debe ser https://host"):
            BinanceRestMarketData(base, SystemClock())

    def test_error_de_red_conserva_la_causa(self) -> None:
        def caido(url: str) -> bytes:
            raise ConnectionResetError("reset por el par")

        md = BinanceRestMarketData(MARKET_DATA_URL, SystemClock(), fetch=caido)
        with pytest.raises(MarketDataError, match="GET /api/v3/depth fallo") as info:
            md.depth("BTCUSDT", 10)
        assert isinstance(info.value.__cause__, ConnectionResetError)

    def test_cuerpo_que_no_es_json(self) -> None:
        md = BinanceRestMarketData(
            MARKET_DATA_URL, SystemClock(), fetch=lambda url: b"<html>502</html>"
        )
        with pytest.raises(MarketDataError, match="GET /api/v3/exchangeInfo fallo"):
            md.instrument("BTCUSDT")


@pytest.mark.network
def test_red_real_formato_vigente() -> None:
    md = BinanceRestMarketData(MARKET_DATA_URL, SystemClock(), fetch=urllib_fetch)
    libro = md.depth("BTCUSDT", 100)
    assert libro.best_bid < libro.best_ask
    barras = md.closed_klines("BTCUSDT", "1d", 2)
    assert (barras.timestamp <= SystemClock().now()).all()
