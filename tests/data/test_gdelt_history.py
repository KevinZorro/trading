"""Descarga historica de GDELT: servidor doble y archivos reales recortados."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.errors import DataValidationError
from data.gdelt_history import (
    HistoryStore,
    HttpResult,
    MasterEntry,
    dedupe_first_seen,
    download,
    expected_labels,
    fetch_batch,
    iso,
    parse_master,
    title_key,
)
from data.news import NewsEvent, load_news
from live.news import Fetched, NewsConfig, build_recorder
from sim.clock import SimulatedClock

AQUI = Path(__file__).parent / "gdelt_history"
ZIP_2017 = AQUI / "20170901190000.gkg.csv.zip"
ZIP_2022 = AQUI / "20220310000000.gkg.csv.zip"
MASTER = AQUI / "masterfilelist_2017-09-01.txt"
L1900 = np.datetime64("2017-09-01T19:00:00", "ns")
L1915 = np.datetime64("2017-09-01T19:15:00", "ns")  # hueco real de GDELT
L1930 = np.datetime64("2017-09-01T19:30:00", "ns")
MIN = np.timedelta64(1, "m")


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes(), usedforsecurity=False).hexdigest()


def entrada(label: np.datetime64, ruta: Path) -> MasterEntry:
    return MasterEntry(label, f"https://x/{iso(label)}", md5(ruta), ruta.stat().st_size)


class Servidor:
    """``respuestas[url]`` es una lista de HttpResult que se sirven en orden."""

    def __init__(self, respuestas: dict[str, list[HttpResult]]) -> None:
        self.respuestas = respuestas
        self.pedidos: list[str] = []

    def __call__(self, url: str) -> HttpResult:
        self.pedidos.append(url)
        cola = self.respuestas[url]
        return cola.pop(0) if len(cola) > 1 else cola[0]


def ok(ruta: Path) -> HttpResult:
    return HttpResult(200, ruta.read_bytes(), "Fri, 01 Sep 2017 19:02:00 GMT")


# --- masterfilelist ---------------------------------------------------------


def test_parse_master_real_solo_gkg_y_https() -> None:
    e = parse_master(
        MASTER.read_text().splitlines(), t("2017-09-01T18:00"), t("2017-09-02")
    )
    assert [x.label for x in e] == [
        t("2017-09-01T18:00"),
        t("2017-09-01T18:15"),
        t("2017-09-01T18:30"),
        t("2017-09-01T18:45"),
        L1900,
        L1930,  # el 19:15 no esta listado
        t("2017-09-01T19:45"),
    ]
    assert e[4] == MasterEntry(
        L1900,
        "https://data.gdeltproject.org/gdeltv2/20170901190000.gkg.csv.zip",
        "adbab1f0c143c042d24a190a5b635fe9",
        15413861,
    )


def test_parse_master_rango_semiabierto() -> None:
    e = parse_master(MASTER.read_text().splitlines(), L1900, L1930)
    assert [x.label for x in e] == [L1900]


@pytest.mark.parametrize(
    "linea",
    [
        "12 abc http://data.gdeltproject.org/gdeltv2/20170901190000.gkg.csv.zip",
        "x "
        + "a" * 32
        + " http://data.gdeltproject.org/gdeltv2/20170901190000.gkg.csv.zip",
        "12 "
        + "a" * 32
        + " http://data.gdeltproject.org/gdeltv2/20170901190700.gkg.csv.zip",
    ],
)
def test_parse_master_linea_corrupta_falla(linea: str) -> None:
    with pytest.raises(DataValidationError, match=r"masterfilelist|rejilla"):
        parse_master([linea], t("2017-01-01"), t("2018-01-01"))


def test_expected_labels_rejilla_de_15_min() -> None:
    assert expected_labels(t("2017-09-01T18:00"), t("2017-09-01T19:00")) == [
        t("2017-09-01T18:00"),
        t("2017-09-01T18:15"),
        t("2017-09-01T18:30"),
        t("2017-09-01T18:45"),
    ]
    # un inicio fuera de rejilla arranca en el primer rotulo >= start
    assert expected_labels(t("2017-09-01T18:01"), t("2017-09-01T18:31")) == [
        t("2017-09-01T18:15"),
        t("2017-09-01T18:30"),
    ]


# --- un lote ----------------------------------------------------------------


def test_fetch_batch_registro_ok_con_oraculo_a_mano() -> None:
    e = entrada(L1900, ZIP_2017)
    r = fetch_batch(
        e, Servidor({e.url: [ok(ZIP_2017)]}), retries=3, sleep=lambda s: None
    )
    assert r is not None
    assert (r["status"], r["label"], r["md5"]) == (
        "ok",
        "2017-09-01T19:00:00.000Z",
        e.md5,
    )
    # recortado a 8 filas: 6 del tema (5 de Hearst + seekingalpha) y 2 sin tema
    assert (r["rows_total"], r["rows_malformed"], r["rows_theme"]) == (8, 0, 6)
    assert r["received_at"] is None
    assert r["last_modified"].startswith("Fri, 01 Sep 2017")
    hearst = next(a for a in r["articles"] if a["domain"] == "newstimes.com")
    assert hearst["tone"][0] == 1.49253731343284
    assert hearst["title"] is None


def test_fetch_batch_mismo_esquema_que_el_grabador_en_vivo(tmp_path: Path) -> None:
    zip_vivo = Path(__file__).parents[1] / "live/gdelt/20261020011500.gkg.csv.zip"
    label = t("2026-10-20T01:15")
    cfg = NewsConfig(data_dir=tmp_path / "vivo", give_up_minutes=15)

    def copiar(url: str, dest: Path) -> Fetched:
        shutil.copy(zip_vivo, dest)
        return Fetched(200, "Tue, 20 Oct 2026 01:03:41 GMT")

    reloj = SimulatedClock(t("2026-10-20T01:20"))
    vivo = build_recorder(cfg, reloj, sleep=lambda s: None, fetch=copiar).collect(label)
    e = entrada(label, zip_vivo)
    hist = fetch_batch(
        e, Servidor({e.url: [ok(zip_vivo)]}), retries=0, sleep=lambda s: None
    )
    assert hist is not None
    for campo in (
        "type",
        "schema",
        "label",
        "status",
        "md5",
        "bytes",
        "rows_total",
        "rows_malformed",
        "rows_theme",
        "articles",
    ):
        assert hist[campo] == vivo[campo], campo
    # el historico no inventa una hora de recepcion
    assert hist["received_at"] is None


def test_fetch_batch_404_es_hueco_explicito() -> None:
    e = entrada(L1900, ZIP_2017)
    r = fetch_batch(
        e, Servidor({e.url: [HttpResult(404)]}), retries=3, sleep=lambda s: None
    )
    assert r is not None
    assert (r["status"], r["last_error"]) == ("missing", "http 404")
    assert "articles" not in r


def test_fetch_batch_reintenta_con_backoff_y_se_recupera() -> None:
    e = entrada(L1900, ZIP_2017)
    srv = Servidor({e.url: [HttpResult(503), HttpResult(200, b"basura"), ok(ZIP_2017)]})
    esperas: list[float] = []
    r = fetch_batch(e, srv, retries=3, sleep=esperas.append)
    assert r is not None
    assert r["status"] == "ok"
    assert len(srv.pedidos) == 3
    assert esperas == [2.0, 4.0]  # 503, md5 malo, ok


def test_fetch_batch_md5_distinto_agota_reintentos_y_no_escribe() -> None:
    e = entrada(L1900, ZIP_2017)
    malo = MasterEntry(e.label, e.url, "0" * 32, e.size)
    srv = Servidor({e.url: [ok(ZIP_2017)]})
    esperas: list[float] = []
    assert fetch_batch(malo, srv, retries=3, sleep=esperas.append) is None
    assert len(srv.pedidos) == 4
    assert esperas == [2.0, 4.0, 8.0]


def test_fetch_batch_zip_roto_con_md5_correcto_no_escribe() -> None:
    cuerpo = b"PK\x03\x04 no es un zip"
    e = MasterEntry(L1900, "https://x/a", hashlib.md5(cuerpo).hexdigest(), len(cuerpo))
    srv = Servidor({e.url: [HttpResult(200, cuerpo)]})
    assert fetch_batch(e, srv, retries=1, sleep=lambda s: None) is None


def test_fetch_batch_zip_sin_gkg_falla_sin_escribir(tmp_path: Path) -> None:
    ruta = tmp_path / "x.zip"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("otro.txt", "hola")
    e = entrada(L1900, ruta)
    assert (
        fetch_batch(e, Servidor({e.url: [ok(ruta)]}), retries=0, sleep=lambda s: None)
        is None
    )


# --- descarga completa, reanudable ------------------------------------------


def _store(d: Path) -> HistoryStore:
    return HistoryStore(d, {"theme": "ECON_BITCOIN"})


def test_download_escribe_huecos_y_carga_con_load_news(tmp_path: Path) -> None:
    e = entrada(L1900, ZIP_2017)
    resumen = download(
        [e],
        L1900,
        t("2017-09-01T19:30"),
        _store(tmp_path),
        Servidor({e.url: [ok(ZIP_2017)]}),
        workers=2,
        sleep=lambda s: None,
    )
    assert (resumen.written_ok, resumen.written_missing, resumen.failed) == (1, 1, ())
    ds = load_news(tmp_path)
    assert [(b.label, b.status) for b in ds.batches] == [
        (L1900, "ok"),
        (L1915, "missing"),
    ]
    assert len(ds.batches[0].events) == 6
    assert ds.torn_lines == 0
    assert (
        (tmp_path / "news-2017-09-01.jsonl")
        .read_text()
        .splitlines()[0]
        .startswith('{"origin": "history"')
    )


def test_download_reanuda_y_no_repite_pedidos(tmp_path: Path) -> None:
    e = entrada(L1900, ZIP_2017)
    srv = Servidor({e.url: [ok(ZIP_2017)]})
    kw: dict[str, Any] = {"workers": 1, "sleep": lambda s: None}
    download([e], L1900, L1930, _store(tmp_path), srv, **kw)
    n = len(srv.pedidos)
    otra = download([e], L1900, L1930, _store(tmp_path), srv, **kw)
    assert len(srv.pedidos) == n
    assert (otra.written_ok, otra.written_missing, otra.skipped_done) == (0, 0, 2)
    assert len((tmp_path / "news-2017-09-01.jsonl").read_text().splitlines()) == 3


def test_download_fallo_transitorio_no_queda_escrito_y_se_reintenta(
    tmp_path: Path,
) -> None:
    e = entrada(L1900, ZIP_2017)
    caido = Servidor({e.url: [HttpResult(500)]})
    r1 = download(
        [e],
        L1900,
        L1915,
        _store(tmp_path),
        caido,
        retries=1,
        workers=1,
        sleep=lambda s: None,
    )
    assert r1.failed == ("2017-09-01T19:00:00.000Z",)
    assert r1.written_ok == 0
    assert not list(tmp_path.glob("news-*.jsonl"))  # nada que tape el lote
    sano = Servidor({e.url: [ok(ZIP_2017)]})
    r2 = download(
        [e], L1900, L1915, _store(tmp_path), sano, workers=1, sleep=lambda s: None
    )
    assert (r2.written_ok, r2.failed) == (1, ())


def test_download_tras_linea_truncada_por_caida(tmp_path: Path) -> None:
    e = entrada(L1900, ZIP_2017)
    srv = Servidor({e.url: [ok(ZIP_2017)]})
    download([e], L1900, L1915, _store(tmp_path), srv, workers=1, sleep=lambda s: None)
    ruta = tmp_path / "news-2017-09-01.jsonl"
    ruta.write_bytes(ruta.read_bytes() + b'{"type": "batch", "label": "2017-09-01T19:1')
    assert _store(tmp_path).done() == {"2017-09-01T19:00:00.000Z"}
    download([e], L1900, L1930, _store(tmp_path), srv, workers=1, sleep=lambda s: None)
    ds = load_news(tmp_path)
    assert ds.torn_lines == 1
    assert [b.label for b in ds.batches] == [L1900, L1915]


def test_download_reparte_por_dia_utc_del_rotulo(tmp_path: Path) -> None:
    resumen = download(
        [],
        t("2017-09-01T23:30"),
        t("2017-09-02T00:30"),
        _store(tmp_path),
        Servidor({}),
        workers=1,
        sleep=lambda s: None,
    )
    assert resumen.written_missing == 4
    assert sorted(p.name for p in tmp_path.glob("news-*.jsonl")) == [
        "news-2017-09-01.jsonl",
        "news-2017-09-02.jsonl",
    ]
    r = json.loads((tmp_path / "news-2017-09-02.jsonl").read_text().splitlines()[1])
    assert r["label"] == "2017-09-02T00:00:00.000Z"
    assert r["last_error"].startswith("ausente")


# --- deduplicacion de sindicacion --------------------------------------------


def ev(i: str, label: str, titulo: str | None) -> NewsEvent:
    return NewsEvent(i, t(label), "d.com", "https://d.com/" + i, 0.0, titulo)


BIDEN = (
    "Biden Signs Cryptocurrency Oversight Order as Its Use Explodes"  # real, 17 copias
)


def test_title_key_normaliza_caso_y_puntuacion() -> None:
    assert title_key(
        "Biden Pushes for Coordinated Approach to Crypto Oversight"
    ) == title_key("Biden pushes for coordinated approach to crypto oversight!")
    assert title_key("  A--B  ") == "a b"


def test_dedupe_se_queda_con_la_primera_aparicion() -> None:
    eventos = [
        ev("a", "2022-03-10T00:00", BIDEN),
        ev("b", "2022-03-10T00:00", BIDEN),
        ev("c", "2022-03-10T00:15", BIDEN.lower()),
        ev("d", "2022-03-10T00:15", "Otro titulo"),
    ]
    assert [e.record_id for e in dedupe_first_seen(eventos)] == ["a", "d"]


def test_dedupe_ventana_de_24h_es_semiabierta_y_desde_la_ultima_conservada() -> None:
    eventos = [
        ev("a", "2022-03-10T00:00", BIDEN),
        ev("b", "2022-03-10T23:45", BIDEN),  # 23:45 < 24 h: descartada
        ev("c", "2022-03-11T00:00", BIDEN),  # exactamente 24 h: conservada
        ev("d", "2022-03-11T23:45", BIDEN),  # 23:45 desde "c", no desde "b": descartada
    ]
    assert [e.record_id for e in dedupe_first_seen(eventos)] == ["a", "c"]


def test_dedupe_sin_titulo_no_hay_clave_y_se_conserva() -> None:
    eventos = [ev("a", "2017-09-01T19:00", None), ev("b", "2017-09-01T19:00", None)]
    assert [e.record_id for e in dedupe_first_seen(eventos)] == ["a", "b"]


def test_dedupe_exige_orden_por_rotulo() -> None:
    eventos = [ev("a", "2022-03-10T00:15", BIDEN), ev("b", "2022-03-10T00:00", BIDEN)]
    with pytest.raises(DataValidationError, match="no decrecientes"):
        list(dedupe_first_seen(eventos))
