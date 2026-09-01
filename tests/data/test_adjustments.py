"""Retornos y eventos corporativos.

El punto del modulo es que la serie usable como feature (ajuste hacia adelante)
sea point-in-time y la que reescribe el pasado no se pueda usar por accidente.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.adjustments import (
    backward_adjusted_close,
    total_return_index,
    total_return_log_returns,
)
from data.errors import LookaheadError
from data.instruments import InstrumentSpec
from data.loaders import bars_from_frame
from data.schema import BarSeries

from .conftest import make_frame


def _series(equity: InstrumentSpec, **overrides) -> BarSeries:
    frame = make_frame(n=6)
    frame["split_factor"] = 1.0
    frame["cash_dividend"] = 0.0
    for column, (index, value) in overrides.items():
        frame.loc[index, column] = value
    return bars_from_frame(frame, instrument=equity, freq="1D")


def test_sin_eventos_el_tri_es_el_cociente_de_cierres(
    equity: InstrumentSpec,
) -> None:
    series = _series(equity)
    tri = total_return_index(series)
    esperado = series.close / series.close[0]
    np.testing.assert_allclose(tri, esperado)


def test_dividendo_aumenta_el_retorno_total(equity: InstrumentSpec) -> None:
    limpio = total_return_index(_series(equity))[-1]
    con_dividendo = total_return_index(_series(equity, cash_dividend=(3, 2.0)))[-1]
    assert con_dividendo > limpio


def test_split_no_produce_retorno_espurio(equity: InstrumentSpec) -> None:
    """Un split 2:1 halvea el precio; el retorno total no debe moverse."""
    frame = make_frame(n=6)
    frame["split_factor"] = 1.0
    frame["cash_dividend"] = 0.0
    # A partir de la barra 3 los precios cotizan a la mitad (post-split).
    frame.loc[3:, ["open", "high", "low", "close"]] /= 2.0
    frame.loc[3, "split_factor"] = 2.0
    series = bars_from_frame(frame, instrument=equity, freq="1D")

    tri = total_return_index(series)
    retornos = total_return_log_returns(series)
    # El retorno de la barra del split es del mismo orden que los vecinos,
    # no un -69% artificial.
    assert abs(retornos[3]) < 0.05
    assert tri[-1] == pytest.approx(105.0 / 100.0, rel=1e-12)


def test_tri_es_point_in_time(equity: InstrumentSpec) -> None:
    """El TRI hasta t no cambia si aparecen eventos posteriores a t.

    Esta es la propiedad que permite usarlo como feature. La serie ajustada
    hacia atras no la tiene, y por eso no se usa.
    """
    base = _series(equity)
    con_evento_futuro = _series(equity, cash_dividend=(5, 3.0))
    np.testing.assert_allclose(
        total_return_index(base)[:5], total_return_index(con_evento_futuro)[:5]
    )


def test_ajuste_hacia_atras_no_es_point_in_time(equity: InstrumentSpec) -> None:
    base = backward_adjusted_close(_series(equity), allow_lookahead=True)
    futuro = backward_adjusted_close(
        _series(equity, cash_dividend=(5, 3.0)), allow_lookahead=True
    )
    # Un evento en la barra 5 cambia los precios de las barras 0..4: eso es
    # exactamente el lookahead que el flag obliga a reconocer.
    assert not np.allclose(base[:5], futuro[:5])


def test_ajuste_hacia_atras_exige_flag_explicito(equity: InstrumentSpec) -> None:
    with pytest.raises(LookaheadError, match="point-in-time"):
        backward_adjusted_close(_series(equity))


def test_primer_retorno_es_cero(equity: InstrumentSpec) -> None:
    assert total_return_log_returns(_series(equity))[0] == 0.0
