"""Anti-leakage de la regla de alineacion: ``label + latencia <= close_time``."""

from __future__ import annotations

import numpy as np
import pytest

from data.errors import DataValidationError
from data.news import DEFAULT_LATENCY, NewsBatch, NewsDataset

pytestmark = pytest.mark.leakage

MD5 = "0" * 32
LAT = np.timedelta64(30, "m")
NS = np.timedelta64(1, "ns")


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


def lote(label: str) -> NewsBatch:
    return NewsBatch(t(label), "ok", 10, 0, (), MD5, None, None)


def dataset() -> NewsDataset:
    return NewsDataset(
        tuple(
            lote(f"2026-10-20T0{h}:{m}")
            for h in (1, 2)
            for m in ("00", "15", "30", "45")
        )
    )


def rotulos(batches: tuple[NewsBatch, ...]) -> list[str]:
    return [str(b.label)[11:16] for b in batches]


def test_lote_exactamente_en_close_menos_latencia_entra() -> None:
    # close 02:15, latencia 30: el lote 01:45 queda disponible justo a las 02:15.
    usable = dataset().usable(t("2026-10-20T02:15:00"), LAT)
    assert rotulos(usable)[-1] == "01:45"


def test_un_nanosegundo_antes_el_lote_no_entra() -> None:
    usable = dataset().usable(t("2026-10-20T02:15:00") - NS, LAT)
    assert rotulos(usable)[-1] == "01:30"


def test_agregar_lotes_posteriores_no_cambia_lo_visible_en_t() -> None:
    corto = dataset()
    largo = NewsDataset(
        (*corto.batches, lote("2026-10-20T03:00"), lote("2026-10-20T03:15"))
    )
    for cierre in ("2026-10-20T01:30", "2026-10-20T02:00", "2026-10-20T02:44"):
        assert corto.usable(t(cierre), LAT) == largo.usable(t(cierre), LAT)


def test_cierre_anterior_a_todo_no_ve_nada() -> None:
    assert dataset().usable(t("2026-10-20T01:29:59"), LAT) == ()


def test_mas_latencia_nunca_muestra_mas() -> None:
    cierre = t("2026-10-20T02:20")
    visto = [
        len(dataset().usable(cierre, np.timedelta64(m, "m"))) for m in (0, 15, 30, 60)
    ]
    assert visto == sorted(visto, reverse=True)
    assert visto == [6, 5, 4, 2]


def test_latencia_negativa_se_rechaza() -> None:
    with pytest.raises(DataValidationError, match="no puede ser negativa"):
        dataset().usable(t("2026-10-20T02:15"), np.timedelta64(-1, "m"))
    with pytest.raises(DataValidationError, match="no puede ser negativa"):
        dataset().batches[0].available_at(np.timedelta64(-1, "s"))


def test_latencia_por_defecto_es_la_de_la_etapa_4() -> None:
    cierre = t("2026-10-20T02:15:00")
    assert dataset().usable(cierre) == dataset().usable(cierre, DEFAULT_LATENCY)
