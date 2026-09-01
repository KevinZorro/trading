"""Generador sintetico.

Si el generador produjera barras incoherentes, los tests del simulador pasarian
sobre datos que ningun mercado produce. Aqui se comprueba que lo que genera
sobrevive a la misma validacion que los datos reales.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.calendars import AlwaysOpen
from data.instruments import binance_spot_spec
from data.loaders import bars_to_frame
from data.synthetic import RISK_REGIMES, HestonParams, generate_gbm_sv
from data.validation import validate_bars


def test_pasa_la_validacion_completa() -> None:
    series = generate_gbm_sv(500, seed=0)
    validate_bars(
        bars_to_frame(series), instrument=series.instrument, enforce_tick_size=True
    )


def test_es_determinista_por_seed() -> None:
    a = generate_gbm_sv(200, seed=42)
    b = generate_gbm_sv(200, seed=42)
    c = generate_gbm_sv(200, seed=43)
    np.testing.assert_array_equal(a.close, b.close)
    np.testing.assert_array_equal(a.volume, b.volume)
    assert not np.array_equal(a.close, c.close)


def test_ohlc_coherente() -> None:
    series = generate_gbm_sv(1000, seed=7)
    assert np.all(series.high >= series.low)
    assert np.all(series.high >= np.maximum(series.open, series.close))
    assert np.all(series.low <= np.minimum(series.open, series.close))
    assert np.all(series.close > 0)
    assert np.all(series.volume > 0)


def test_arrays_son_de_solo_lectura() -> None:
    """Una estrategia no debe poder corromper la serie bajo el motor."""
    series = generate_gbm_sv(50, seed=1)
    with pytest.raises(ValueError):
        series.close[0] = 1.0


@pytest.mark.parametrize("regime", sorted(RISK_REGIMES))
def test_regimenes_ordenados_por_volatilidad(regime: str) -> None:
    series = generate_gbm_sv(2000, seed=11, params=regime)
    vol = np.std(np.diff(np.log(series.close))) * np.sqrt(252)
    esperada = np.sqrt(RISK_REGIMES[regime].theta)
    # Banda amplia: es una realizacion, no la media del proceso.
    assert 0.4 * esperada < vol < 2.5 * esperada


def test_los_tres_regimenes_estan_separados() -> None:
    vols = []
    for regime in ("low", "medium", "high"):
        series = generate_gbm_sv(2000, seed=3, params=regime)
        vols.append(np.std(np.diff(np.log(series.close))))
    assert vols[0] < vols[1] < vols[2]


def test_volumen_correlaciona_con_volatilidad() -> None:
    series = generate_gbm_sv(2000, seed=5, params="medium")
    rango = (series.high - series.low) / series.close
    corr = np.corrcoef(rango, series.volume)[0, 1]
    assert corr > 0.1


def test_cripto_intradia_contra_calendario_24_7() -> None:
    spec = binance_spot_spec("BTCUSDT")
    series = generate_gbm_sv(
        24 * 30,
        seed=2,
        instrument=spec,
        params="high",
        s0=50_000.0,
        freq="1h",
        bars_per_year=24 * 365,
        start="2021-01-01T00:00:00Z",
    )
    validate_bars(
        bars_to_frame(series),
        instrument=spec,
        calendar=AlwaysOpen(),
        freq="1h",
    )


def test_sin_eventos_corporativos() -> None:
    series = generate_gbm_sv(100, seed=9)
    assert not series.has_corporate_actions()


def test_parametros_invalidos() -> None:
    with pytest.raises(ValueError):
        generate_gbm_sv(1, seed=0)
    with pytest.raises(ValueError, match="regimen desconocido"):
        generate_gbm_sv(10, seed=0, params="extremo")
    with pytest.raises(ValueError, match="rho"):
        HestonParams(rho=1.5)


class TestGapOvernight:
    """Sin gaps inyectados el camino es continuo y el efecto de latencia
    (valuacion entre close[t] y open[t+1]) queda sin testear."""

    def test_por_defecto_no_hay_gap(self) -> None:
        series = generate_gbm_sv(200, seed=4)
        np.testing.assert_allclose(series.open[1:], series.close[:-1], rtol=1e-4)

    def test_con_gap_las_aperturas_se_despegan_de_los_cierres(self) -> None:
        series = generate_gbm_sv(500, seed=4, overnight_gap_frac=0.5)
        gaps = np.log(series.open[1:] / series.close[:-1])
        assert np.std(gaps) > 0
        assert np.mean(np.abs(gaps)) > 1e-4

    def test_el_regimen_alto_gapea_mas(self) -> None:
        def gap_std(regime: str) -> float:
            series = generate_gbm_sv(
                1000, seed=6, params=regime, overnight_gap_frac=0.5
            )
            return float(np.std(np.log(series.open[1:] / series.close[:-1])))

        assert gap_std("low") < gap_std("medium") < gap_std("high")

    def test_las_barras_con_gap_siguen_siendo_validas(self) -> None:
        series = generate_gbm_sv(500, seed=8, params="high", overnight_gap_frac=0.8)
        validate_bars(bars_to_frame(series), instrument=series.instrument)

    def test_gap_negativo_rechazado(self) -> None:
        with pytest.raises(ValueError, match="overnight_gap_frac"):
            generate_gbm_sv(10, seed=0, overnight_gap_frac=-0.1)
