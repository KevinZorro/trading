"""``with_news``: de la tabla de N4 a las columnas exogenas de la serie (N5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from agents.diagnosis import feature_groups
from data.errors import DataValidationError
from data.instruments import CommissionSchema, InstrumentSpec
from data.schema import BarSeries
from envs.observation import ObservationBuilder, ObservationSpec
from features.news import (
    NEWS_EXOG,
    NEWS_FEATURE_NAMES,
    PLACEBO_LAG,
    build_news_features,
    with_news,
)

from ..sim.conftest import make_series
from .news_helpers import grilla

SPEC = InstrumentSpec(
    symbol="FRAC",
    venue="TEST",
    tick_size=0.01,
    lot_size=1e-8,
    allow_fractional=True,
    qty_precision=8,
    min_order_qty=0.0,
    min_notional=0.0,
    commission_schema=CommissionSchema(kind="fixed", value=0.0),
    asset_class="crypto",
)


def barras(inicio: str, n: int) -> BarSeries:
    """Barras diarias que cierran a las 00:00 UTC desde ``inicio``."""
    s = make_series(SPEC, np.linspace(100.0, 110.0, n))
    cierres = pd.date_range(inicio, periods=n, freq="D").to_numpy("datetime64[ns]")
    return replace(s, timestamp=cierres)


def tabla():
    frame = grilla(400, "2019-12-01")
    # Un dia de atencion alta: distingue las barras entre si.
    pico = (frame["label"] >= "2020-01-10") & (frame["label"] < "2020-01-11")
    frame.loc[pico, "rows_theme"] = 8
    return build_news_features(frame)


def test_columnas_de_b_en_el_orden_del_adr() -> None:
    s = with_news(barras("2020-01-09", 5), tabla())
    assert s.exog_names == NEWS_FEATURE_NAMES
    assert s.exog is not None
    assert s.exog.shape == (5, 7)
    assert [c.scaled for c in NEWS_EXOG] == [True] * 5 + [False] * 2
    # La atencion no es constante: el pico esta en la ventana.
    assert np.ptp(s.exog[:, 0]) > 0


def test_placebo_es_el_bloque_de_364_dias_antes() -> None:
    t = tabla()
    b = with_news(barras("2020-01-09", 5), t)
    # Fechas desplazadas 364 dias: el placebo de la serie tardia es la B temprana.
    tarde = barras(str(np.datetime64("2020-01-09") + PLACEBO_LAG), 5)
    p = with_news(tarde, t, PLACEBO_LAG)
    assert b.exog is not None
    assert p.exog is not None
    assert_array_equal(p.exog, b.exog)  # NaN == NaN en assert_array_equal
    assert_array_equal(p.close, tarde.close)  # el precio es el de la serie tardia


def test_barra_sin_fila_en_la_tabla_falla_en_vez_de_rellenar() -> None:
    with pytest.raises(DataValidationError, match="no hay features de noticias"):
        with_news(barras("2030-01-01", 3), tabla())


def test_el_lag_no_puede_mirar_el_futuro() -> None:
    with pytest.raises(DataValidationError, match="negativo"):
        with_news(barras("2020-01-09", 3), tabla(), -PLACEBO_LAG)


def test_el_grupo_noticias_del_diagnostico() -> None:
    nombres = ObservationBuilder(ObservationSpec(exog=NEWS_EXOG)).names
    grupos = feature_groups(nombres)
    assert grupos["noticias"] == list(range(len(nombres) - 7, len(nombres)))
    assert "noticias" not in feature_groups(ObservationBuilder().names)
