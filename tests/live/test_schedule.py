"""next_due con oraculos escritos a mano (ADR 0007, seccion 4)."""

from __future__ import annotations

import numpy as np
import pytest

from live.schedule import Due, next_due, parse_daily


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


@pytest.mark.parametrize(
    ("ahora", "esperado", "kind"),
    [
        ("2026-10-17T17:17:09", "2026-10-17T17:30:00", "periodic"),
        ("2026-10-17T17:30:00", "2026-10-17T17:45:00", "periodic"),  # estrictamente
        ("2026-10-17T23:50:00", "2026-10-18T00:00:00", "periodic"),
        ("2026-10-18T00:00:00", "2026-10-18T00:00:30", "daily"),
        ("2026-10-18T00:00:30", "2026-10-18T00:15:00", "periodic"),
        ("2026-10-18T00:00:29.999", "2026-10-18T00:00:30", "daily"),
    ],
)
def test_proximo_instante(ahora: str, esperado: str, kind: str) -> None:
    assert next_due(t(ahora), 15, "00:00:30") == Due(t(esperado), kind)  # type: ignore[arg-type]


def test_paso_de_60_minutos() -> None:
    assert next_due(t("2026-10-17T17:17:09"), 60, "00:00:30").at == t(
        "2026-10-17T18:00:00"
    )


def test_diaria_en_otra_hora() -> None:
    due = next_due(t("2026-10-17T13:00:00"), 60, "13:00:30")
    assert due == Due(t("2026-10-17T13:00:30"), "daily")


@pytest.mark.parametrize("malo", ["0:0", "25:00:00", "aa:bb:cc", "00:60:00"])
def test_hora_diaria_invalida(malo: str) -> None:
    with pytest.raises(ValueError, match="daily_snapshot_utc"):
        parse_daily(malo)
