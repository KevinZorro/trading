"""Carga desde disco y construccion de BarSeries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.calendars import WeekdayCalendar
from data.errors import AdjustedPriceError, SchemaError
from data.instruments import InstrumentSpec
from data.loaders import bars_from_frame, bars_to_frame, load_bars
from data.schema import BarSeries

from .conftest import make_frame


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_round_trip_disco(
    tmp_path, equity: InstrumentSpec, suffix: str
) -> None:
    frame = make_frame(n=12)
    path = tmp_path / f"bars{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    series = load_bars(
        path, instrument=equity, freq="1D", calendar=WeekdayCalendar()
    )
    assert len(series) == 12
    np.testing.assert_allclose(series.close, frame["close"].to_numpy())
    assert series.symbol == "TEST"
    assert str(path) in series.source


def test_extension_no_soportada(tmp_path, equity: InstrumentSpec) -> None:
    path = tmp_path / "bars.txt"
    path.write_text("nada")
    with pytest.raises(SchemaError, match="extension no soportada"):
        load_bars(path, instrument=equity, freq="1D")


def test_fichero_inexistente(tmp_path, equity: InstrumentSpec) -> None:
    with pytest.raises(FileNotFoundError):
        load_bars(tmp_path / "no_existe.csv", instrument=equity, freq="1D")


def test_column_map(equity: InstrumentSpec) -> None:
    frame = make_frame(n=5).rename(
        columns={"timestamp": "Date", "close": "Close", "volume": "Volume"}
    )
    series = bars_from_frame(
        frame,
        instrument=equity,
        freq="1D",
        column_map={"Date": "timestamp", "Close": "close", "Volume": "volume"},
    )
    assert len(series) == 5


def test_csv_de_yahoo_es_rechazado(tmp_path, equity: InstrumentSpec) -> None:
    """Caso real: el CSV trae Adj Close y alguien lo carga sin mirar."""
    frame = make_frame(n=5)
    frame["Adj Close"] = frame["close"] * 0.87
    path = tmp_path / "yahoo.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(AdjustedPriceError):
        load_bars(path, instrument=equity, freq="1D")


def test_eventos_por_defecto_cuando_faltan(equity: InstrumentSpec) -> None:
    series = bars_from_frame(make_frame(n=5), instrument=equity, freq="1D")
    assert np.all(series.split_factor == 1.0)
    assert np.all(series.cash_dividend == 0.0)
    assert not series.has_corporate_actions()


def test_timestamp_naive_en_csv_falla(tmp_path, equity: InstrumentSpec) -> None:
    """Nunca se localiza a UTC en silencio: un naive es ambiguo."""
    frame = make_frame(n=5)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    path = tmp_path / "naive.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="tz-naive"):
        load_bars(path, instrument=equity, freq="1D")


class TestBarSeries:
    def test_slice(self, equity: InstrumentSpec) -> None:
        series = bars_from_frame(make_frame(n=10), instrument=equity, freq="1D")
        sub = series.slice(2, 5)
        assert len(sub) == 3
        np.testing.assert_allclose(sub.close, series.close[2:5])
        assert sub.instrument is series.instrument

    @pytest.mark.parametrize("bounds", [(-1, 3), (0, 99), (5, 5)])
    def test_slice_invalido(self, equity: InstrumentSpec, bounds) -> None:
        series = bars_from_frame(make_frame(n=10), instrument=equity, freq="1D")
        with pytest.raises(SchemaError):
            series.slice(*bounds)

    def test_campo_desconocido(self, equity: InstrumentSpec) -> None:
        series = bars_from_frame(make_frame(n=5), instrument=equity, freq="1D")
        with pytest.raises(SchemaError, match="campo desconocido"):
            series.field("vwap")

    def test_meta_serializable(self, equity: InstrumentSpec) -> None:
        series = bars_from_frame(make_frame(n=5), instrument=equity, freq="1D")
        meta = series.meta()
        assert meta["symbol"] == "TEST"
        assert meta["n_bars"] == 5
        assert meta["freq"] == "1D"

    def test_arrays_congelados(self, equity: InstrumentSpec) -> None:
        series = bars_from_frame(make_frame(n=5), instrument=equity, freq="1D")
        for field in ("open", "high", "low", "close", "volume"):
            assert not series.field(field).flags.writeable

    def test_serie_vacia(self, equity: InstrumentSpec) -> None:
        with pytest.raises(SchemaError, match="vacia"):
            BarSeries(
                instrument=equity,
                freq="1D",
                timestamp=np.array([], dtype="datetime64[ns]"),
                open=np.array([]),
                high=np.array([]),
                low=np.array([]),
                close=np.array([]),
                volume=np.array([]),
                split_factor=np.array([]),
                cash_dividend=np.array([]),
            )

    def test_longitudes_inconsistentes(self, equity: InstrumentSpec) -> None:
        with pytest.raises(SchemaError, match="longitud inconsistente"):
            BarSeries(
                instrument=equity,
                freq="1D",
                timestamp=pd.DatetimeIndex(
                    pd.date_range("2020-01-01", periods=3)
                ).to_numpy("datetime64[ns]"),
                open=np.ones(3),
                high=np.ones(3),
                low=np.ones(3),
                close=np.ones(2),
                volume=np.ones(3),
                split_factor=np.ones(3),
                cash_dividend=np.zeros(3),
            )


def test_bars_to_frame_es_inverso_de_bars_from_frame(
    equity: InstrumentSpec,
) -> None:
    original = make_frame(n=8)
    series = bars_from_frame(original, instrument=equity, freq="1D")
    vuelta = bars_to_frame(series)
    pd.testing.assert_series_equal(
        vuelta["close"], original["close"], check_names=False
    )
    pd.testing.assert_series_equal(
        vuelta["timestamp"], original["timestamp"], check_names=False
    )
