"""Anti-leakage del conteo por barra: cada lote cae en la primera barra que lo ve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.news_aggregate import bar_healthy_counts

pytestmark = pytest.mark.leakage

LAT60 = np.timedelta64(60, "m")


def un_lote(label: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame({"label": pd.to_datetime([label])})
    return frame, np.array([True])


@pytest.mark.parametrize(
    ("label", "barra"),
    [
        # 23:00 + 60 min = 00:00 = cierre de la barra del dia 1: <= close, entra.
        ("2020-01-01T23:00:00", "2020-01-02"),
        # 23:15 + 60 min = 00:15 del dia 2: despues del cierre, va a la barra siguiente.
        ("2020-01-01T23:15:00", "2020-01-03"),
        # 00:00 + 60 min = 01:00 del dia 2: barra que cierra el dia 3.
        ("2020-01-02T00:00:00", "2020-01-03"),
        # 22:45 + 60 = 23:45 < 00:00 y > cierre previo: barra del dia 1.
        ("2020-01-01T22:45:00", "2020-01-02"),
    ],
)
def test_el_lote_cae_en_la_barra_que_ya_podia_verlo(label: str, barra: str) -> None:
    frame, sano = un_lote(label)
    cuentas = bar_healthy_counts(frame, sano, LAT60)
    assert list(cuentas.index) == [pd.Timestamp(barra)]


def test_ningun_lote_cuenta_en_dos_barras() -> None:
    labels = pd.date_range("2020-01-01", periods=96 * 3, freq="15min")
    frame = pd.DataFrame({"label": labels})
    cuentas = bar_healthy_counts(frame, np.ones(len(labels), dtype=bool), LAT60)
    assert int(cuentas.sum()) == len(labels)
    assert cuentas.iloc[1] == 96  # una barra interior ve exactamente 96 lotes
