"""Anti-leakage del historico de GDELT: lo escrito respeta la regla de la Etapa 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.gdelt_history import (
    HistoryStore,
    dedupe_first_seen,
    download,
)
from data.news import NewsEvent, load_news
from tests.data.test_gdelt_history import L1900, ZIP_2017, Servidor, entrada, ok

pytestmark = pytest.mark.leakage

MIN = np.timedelta64(1, "m")


def test_lote_historico_solo_es_usable_desde_rotulo_mas_latencia(
    tmp_path: Path,
) -> None:
    e = entrada(L1900, ZIP_2017)
    download(
        [e],
        L1900,
        L1900 + 15 * MIN,
        HistoryStore(tmp_path, {}),
        Servidor({e.url: [ok(ZIP_2017)]}),
        workers=1,
        sleep=lambda s: None,
    )
    ds = load_news(tmp_path)
    lat = np.timedelta64(30, "m")
    cierre = L1900 + lat
    assert ds.usable(cierre - np.timedelta64(1, "ms"), lat) == ()  # un ms antes: no
    assert len(ds.usable(cierre, lat)) == 1  # justo en rotulo + latencia: si


def test_el_historico_no_declara_hora_de_recepcion_ni_pub_time(tmp_path: Path) -> None:
    e = entrada(L1900, ZIP_2017)
    download(
        [e],
        L1900,
        L1900 + 15 * MIN,
        HistoryStore(tmp_path, {}),
        Servidor({e.url: [ok(ZIP_2017)]}),
        workers=1,
        sleep=lambda s: None,
    )
    (lote,) = load_news(tmp_path).batches
    assert lote.received_at is None
    texto = (tmp_path / "news-2017-09-01.jsonl").read_text()
    assert "pub_time" not in texto  # la regla vive solo en NewsDataset.usable


def test_dedupe_no_usa_copias_posteriores_para_decidir_el_pasado() -> None:
    """Agregar eventos futuros no cambia lo que se conserva hasta t."""

    def ev(i: str, h: str) -> NewsEvent:
        return NewsEvent(i, np.datetime64(h, "ns"), "d", "u" + i, 0.0, "Mismo titulo")

    base = [ev("a", "2022-03-10T00:00"), ev("b", "2022-03-10T01:00")]
    ext = [*base, ev("c", "2022-03-10T02:00"), ev("d", "2022-03-12T00:00")]
    hasta_t = [e.record_id for e in dedupe_first_seen(base)]
    assert [e.record_id for e in dedupe_first_seen(ext)][: len(hasta_t)] == hasta_t
    assert hasta_t == ["a"]  # la primera, no la ultima
