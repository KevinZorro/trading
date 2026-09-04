"""Calendarios de mercado.

La deteccion de huecos se hace contra un calendario **explicito e inyectable**,
no contra una heuristica de "diferencia modal entre timestamps". Un feriado no
es un hueco; una sesion faltante si lo es, y la diferencia solo la sabe el
calendario.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

UTC = "UTC"


@runtime_checkable
class Calendar(Protocol):
    """Contrato minimo: dado un rango observado, que barras deberian existir."""

    name: str

    def expected_index(
        self, start: pd.Timestamp, end: pd.Timestamp, freq: str
    ) -> pd.DatetimeIndex:
        """Indice UTC de las barras esperadas en ``[start, end]`` inclusive."""
        ...


@dataclass(frozen=True)
class AlwaysOpen:
    """Mercado 24/7 (cripto). Rejilla regular a la frecuencia declarada."""

    name: str = "always_open"

    def expected_index(
        self, start: pd.Timestamp, end: pd.Timestamp, freq: str
    ) -> pd.DatetimeIndex:
        return pd.date_range(start=start, end=end, freq=freq, tz=UTC)


@dataclass(frozen=True)
class WeekdayCalendar:
    """Sesiones diarias de lunes a viernes, con feriados explicitos.

    ASSUMPTION: pensado para barras diarias. Todas las barras deben compartir la
    misma hora del dia (la hora de cierre de la sesion expresada en UTC); si no,
    el dataset mezcla convenciones y se rechaza. Para intradia de renta variable
    usar :class:`ExplicitCalendar` con las sesiones reales del venue.
    """

    holidays: frozenset[date] = frozenset()
    name: str = "weekday"

    def expected_index(
        self, start: pd.Timestamp, end: pd.Timestamp, freq: str
    ) -> pd.DatetimeIndex:
        if start.time() != end.time():
            raise ValueError(
                "WeekdayCalendar exige una unica hora de sesion; "
                f"se observaron {start.time()} y {end.time()}"
            )
        session_time = start.time()
        days = pd.bdate_range(start=start.normalize(), end=end.normalize(), tz=UTC)
        if self.holidays:
            es_feriado = pd.Series(days.date, index=days).isin(self.holidays)
            days = days[~es_feriado.to_numpy()]
        return pd.DatetimeIndex(
            [
                pd.Timestamp.combine(d.date(), session_time).tz_localize(UTC)
                for d in days
            ]
        )


@dataclass(frozen=True)
class ExplicitCalendar:
    """Calendario dado por la lista exacta de timestamps de sesion.

    Es la opcion honesta cuando el calendario del venue es complejo (medias
    sesiones, cambios de horario, historia de feriados): se carga de una fuente
    externa en vez de reconstruirse con reglas aproximadas.
    """

    sessions: pd.DatetimeIndex
    name: str = "explicit"

    def __post_init__(self) -> None:
        idx = pd.DatetimeIndex(self.sessions)
        if idx.tz is None:
            raise ValueError("las sesiones del calendario deben ser tz-aware")
        object.__setattr__(self, "sessions", idx.tz_convert(UTC).sort_values())

    def expected_index(
        self, start: pd.Timestamp, end: pd.Timestamp, freq: str
    ) -> pd.DatetimeIndex:
        mask = (self.sessions >= start) & (self.sessions <= end)
        return self.sessions[mask]
