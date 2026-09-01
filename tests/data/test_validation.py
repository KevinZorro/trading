"""La validacion debe fallar ruidosamente. Cada test rompe una invariante."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.calendars import AlwaysOpen, ExplicitCalendar, WeekdayCalendar
from data.errors import (
    AdjustedPriceError,
    CalendarGapError,
    DataValidationError,
    SchemaError,
)
from data.instruments import InstrumentSpec, us_equity_spec
from data.validation import validate_bars

from .conftest import SESSION_TIME, make_frame


def test_frame_valido_pasa(equity: InstrumentSpec) -> None:
    validate_bars(make_frame(), instrument=equity)


def test_frame_valido_pasa_contra_calendario(equity: InstrumentSpec) -> None:
    validate_bars(
        make_frame(),
        instrument=equity,
        calendar=WeekdayCalendar(),
        freq="1D",
    )


class TestEsquema:
    @pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
    def test_columna_faltante(self, equity: InstrumentSpec, column: str) -> None:
        frame = make_frame().drop(columns=[column])
        with pytest.raises(SchemaError, match="columnas requeridas ausentes"):
            validate_bars(frame, instrument=equity)

    def test_columna_no_numerica(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame["close"] = frame["close"].astype(str)
        with pytest.raises(SchemaError, match="debe ser numerica"):
            validate_bars(frame, instrument=equity)

    def test_precios_ajustados_rechazados(self, equity: InstrumentSpec) -> None:
        """El caso Yahoo: adj_close en la ruta de ejecucion es lookahead."""
        frame = make_frame()
        frame["Adj Close"] = frame["close"] * 0.9
        with pytest.raises(AdjustedPriceError, match="precio ajustado"):
            validate_bars(frame, instrument=equity)


class TestTimestamps:
    def test_tz_naive_rechazado(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
        with pytest.raises(SchemaError, match="tz-naive"):
            validate_bars(frame, instrument=equity)

    def test_zona_horaria_distinta_de_utc_rechazada(
        self, equity: InstrumentSpec
    ) -> None:
        frame = make_frame()
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("America/New_York")
        with pytest.raises(SchemaError, match="UTC"):
            validate_bars(frame, instrument=equity)

    def test_duplicados_rechazados(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]
        with pytest.raises(DataValidationError, match="duplicados"):
            validate_bars(frame, instrument=equity)

    def test_no_monotonico_rechazado(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame = frame.iloc[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]].reset_index(drop=True)
        with pytest.raises(DataValidationError, match="crecientes"):
            validate_bars(frame, instrument=equity)


class TestPrecios:
    @pytest.mark.parametrize("column", ["open", "high", "low", "close"])
    def test_precio_no_positivo(self, equity: InstrumentSpec, column: str) -> None:
        frame = make_frame()
        frame.loc[4, column] = 0.0
        with pytest.raises(DataValidationError):
            validate_bars(frame, instrument=equity)

    def test_high_menor_que_low(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[5, "high"] = frame.loc[5, "low"] - 1.0
        with pytest.raises(DataValidationError, match="high < low"):
            validate_bars(frame, instrument=equity)

    def test_close_fuera_de_rango(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[6, "close"] = frame.loc[6, "high"] + 1.0
        with pytest.raises(DataValidationError, match=r"close fuera del rango"):
            validate_bars(frame, instrument=equity)

    def test_open_fuera_de_rango(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[6, "open"] = frame.loc[6, "low"] - 1.0
        with pytest.raises(DataValidationError, match=r"open fuera del rango"):
            validate_bars(frame, instrument=equity)

    def test_volumen_negativo(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[2, "volume"] = -1.0
        with pytest.raises(DataValidationError, match="volumen negativo"):
            validate_bars(frame, instrument=equity)

    def test_nan_en_precio(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[7, "close"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_bars(frame, instrument=equity)


class TestSimbolo:
    def test_varios_simbolos_rechazados(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[3, "symbol"] = "OTRO"
        with pytest.raises(DataValidationError, match="unico simbolo"):
            validate_bars(frame, instrument=equity)

    def test_simbolo_distinto_del_instrumento(self, equity: InstrumentSpec) -> None:
        frame = make_frame(symbol="OTRO")
        with pytest.raises(DataValidationError, match="!= instrumento"):
            validate_bars(frame, instrument=equity)


class TestCalendario:
    def test_sesion_faltante_es_hueco(self, equity: InstrumentSpec) -> None:
        frame = make_frame(n=10).drop(index=4).reset_index(drop=True)
        with pytest.raises(CalendarGapError, match="barras faltantes"):
            validate_bars(
                frame, instrument=equity, calendar=WeekdayCalendar(), freq="1D"
            )

    def test_feriado_declarado_no_es_hueco(self, equity: InstrumentSpec) -> None:
        """Un feriado no es un hueco; la diferencia solo la sabe el calendario."""
        frame = make_frame(n=10)
        holiday = frame.loc[4, "timestamp"].date()
        frame = frame.drop(index=4).reset_index(drop=True)
        validate_bars(
            frame,
            instrument=equity,
            calendar=WeekdayCalendar(holidays=frozenset({holiday})),
            freq="1D",
        )

    def test_barra_en_fin_de_semana_rechazada(self, equity: InstrumentSpec) -> None:
        frame = make_frame(n=6)
        weekend = pd.Timestamp(f"2020-01-04T{SESSION_TIME}Z")
        extra = frame.iloc[[0]].copy()
        extra["timestamp"] = weekend
        frame = (
            pd.concat([frame, extra])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        with pytest.raises(CalendarGapError, match="fuera del calendario"):
            validate_bars(
                frame, instrument=equity, calendar=WeekdayCalendar(), freq="1D"
            )

    def test_cripto_24_7_sin_huecos(self) -> None:
        from data.instruments import binance_spot_spec

        spec = binance_spot_spec("BTCUSDT")
        frame = make_frame(n=48, symbol="BTCUSDT", freq="1h", business_days=False)
        validate_bars(frame, instrument=spec, calendar=AlwaysOpen(), freq="1h")

    def test_hueco_intradia_en_cripto(self) -> None:
        from data.instruments import binance_spot_spec

        spec = binance_spot_spec("BTCUSDT")
        frame = make_frame(n=48, symbol="BTCUSDT", freq="1h", business_days=False)
        frame = frame.drop(index=20).reset_index(drop=True)
        with pytest.raises(CalendarGapError):
            validate_bars(frame, instrument=spec, calendar=AlwaysOpen(), freq="1h")

    def test_calendario_explicito(self, equity: InstrumentSpec) -> None:
        frame = make_frame(n=8)
        sessions = pd.DatetimeIndex(frame["timestamp"])
        validate_bars(
            frame,
            instrument=equity,
            calendar=ExplicitCalendar(sessions=sessions),
            freq="1D",
        )


class TestTickSize:
    def test_desactivado_por_defecto(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame.loc[3, "close"] = 103.00456  # dentro de [low, high], fuera del tick
        validate_bars(frame, instrument=equity)

    def test_activado_detecta_precio_fuera_de_tick(
        self, equity: InstrumentSpec
    ) -> None:
        frame = make_frame()
        frame.loc[3, "close"] = 103.00456
        with pytest.raises(DataValidationError, match="tick_size"):
            validate_bars(frame, instrument=equity, enforce_tick_size=True)


class TestEventosCorporativos:
    def test_split_no_positivo(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame["split_factor"] = 1.0
        frame.loc[2, "split_factor"] = 0.0
        with pytest.raises(DataValidationError, match="split_factor"):
            validate_bars(frame, instrument=equity)

    def test_dividendo_negativo(self, equity: InstrumentSpec) -> None:
        frame = make_frame()
        frame["cash_dividend"] = 0.0
        frame.loc[2, "cash_dividend"] = -0.5
        with pytest.raises(DataValidationError, match="cash_dividend"):
            validate_bars(frame, instrument=equity)
