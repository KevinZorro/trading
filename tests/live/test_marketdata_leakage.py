"""Anti-leakage de ``closed_klines``: una vela abierta nunca llega como cerrada.

Una kline en curso tiene high, low, close y volumen provisorios. Si llegara a
la sombra como cerrada, el spread de Corwin-Schultz y el volumen del slippage
se calcularian con datos que el backtest no habria tenido en ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live.config import MARKET_DATA_URL
from live.marketdata import BinanceRestMarketData
from sim.clock import SimulatedClock

from .conftest import DIA_MS, Servidor, cliente, grabacion, recibido

pytestmark = pytest.mark.leakage


def test_descarta_la_vela_en_curso() -> None:
    crudo = grabacion("klines_1d.json")
    corte = recibido("klines_1d.json")
    # La grabacion se hizo a media jornada: la ultima kline esta abierta.
    assert crudo[-1][6] > corte.astype("datetime64[ms]").astype(np.int64)
    barras = cliente(ahora=corte).closed_klines("BTCUSDT", "1d", 5)
    assert len(barras) == 5
    cerradas = crudo[-6:-1]
    esperados = pd.to_datetime([k[0] + DIA_MS for k in cerradas], unit="ms")
    np.testing.assert_array_equal(barras.timestamp, esperados.to_numpy())
    np.testing.assert_array_equal(barras.close, [float(k[4]) for k in cerradas])
    np.testing.assert_array_equal(barras.volume, [float(k[5]) for k in cerradas])
    assert (barras.timestamp <= corte).all()
    assert barras.freq == "1D"
    assert barras.instrument.min_notional == 5.0


def test_la_hora_de_corte_se_toma_antes_de_pedir() -> None:
    crudo = grabacion("klines_1d.json")
    cierre = np.datetime64(crudo[-2][0] + DIA_MS, "ms").astype("datetime64[ns]")
    servidor = Servidor()
    reloj = SimulatedClock(cierre - np.timedelta64(100, "ms"))
    # La barra cierra mientras la respuesta viaja: el venue la mando abierta.
    servidor.al_pedir = lambda: reloj.advance_to(cierre + np.timedelta64(1, "s"))
    md = BinanceRestMarketData(MARKET_DATA_URL, reloj, fetch=servidor)
    barras = md.closed_klines("BTCUSDT", "1d", 3)
    assert barras.timestamp[-1] == cierre - np.timedelta64(1, "D")
