"""Grabador de noticias: reloj simulado y servidor doble, sin red."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from live.config import ConfigError
from live.health import read_heartbeat
from live.news import (
    Fetched,
    NewsConfig,
    NewsRecorder,
    NewsStore,
    build_recorder,
    evaluate,
)
from sim.clock import SimulatedClock

LOTE = Path(__file__).parent / "gdelt/20261020011500.gkg.csv.zip"
L0115 = np.datetime64("2026-10-20T01:15:00", "ns")
MIN = np.timedelta64(1, "m")
LAST_MODIFIED = "Tue, 20 Oct 2026 01:03:41 GMT"


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


class Servidor:
    """GDELT de juguete: cada rotulo aparece a una hora y sirve el lote real."""

    def __init__(
        self, reloj: SimulatedClock, disponible: dict[str, np.datetime64]
    ) -> None:
        self.reloj = reloj
        self.disponible = disponible  # AAAAMMDDHHMMSS -> hora en que aparece
        self.pedidos: list[tuple[np.datetime64, str]] = []
        self.forzar: list[Fetched | bytes] = []  # respuestas a inyectar, en orden

    def __call__(self, url: str, dest: Path) -> Fetched:
        rotulo = url.rsplit("/", 1)[1].split(".")[0]
        self.pedidos.append((self.reloj.now(), rotulo))
        if self.forzar:
            forzada = self.forzar.pop(0)
            if isinstance(forzada, bytes):
                dest.write_bytes(forzada)
                return Fetched(200)
            return forzada
        if rotulo not in self.disponible or self.reloj.now() < self.disponible[rotulo]:
            return Fetched(404)
        shutil.copy(LOTE, dest)
        return Fetched(200, LAST_MODIFIED)


def montar(
    tmp_path: Path,
    ahora: str,
    disponible: dict[str, str],
    **opciones: object,
) -> tuple[NewsRecorder, SimulatedClock, Servidor]:
    reloj = SimulatedClock(t(ahora))
    servidor = Servidor(reloj, {k: t(v) for k, v in disponible.items()})
    config = NewsConfig(tmp_path, **opciones)  # type: ignore[arg-type]
    recorder = build_recorder(
        config,
        reloj,
        sleep=lambda s: reloj.advance_to(
            reloj.now() + np.timedelta64(round(s * 1000), "ms")
        ),
        fetch=servidor,
    )
    return recorder, reloj, servidor


def registros(tmp_path: Path, dia: str = "2026-10-20") -> list[dict[str, object]]:
    ruta = tmp_path / f"news-{dia}.jsonl"
    lineas = ruta.read_text().splitlines() if ruta.exists() else []
    return [json.loads(x) for x in lineas]


def test_lote_disponible_antes_del_rotulo_se_registra_con_su_latencia(
    tmp_path: Path,
) -> None:
    # Arranca a rotulo - 15 min; el archivo aparece a las 01:03:41 (dato medido).
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:00:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    r = rec.collect(L0115)
    assert r["status"] == "ok"
    # 404 a las 01:00, 01:01, 01:02 y 01:03; el 01:04 ya existe: 5 intentos.
    assert r["attempts"] == 5
    assert r["last_404_at"] == "2026-10-20T01:03:00.000Z"
    assert r["first_attempt_at"] == "2026-10-20T01:00:00.000Z"
    assert r["received_at"] == "2026-10-20T01:04:00.000Z"
    assert r["last_modified"] == LAST_MODIFIED
    assert r["label"] == "2026-10-20T01:15:00.000Z"


def test_registro_trae_conteos_md5_y_articulos_del_lote(tmp_path: Path) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    r = rec.collect(L0115)
    crudo = LOTE.read_bytes()
    assert r["md5"] == hashlib.md5(crudo, usedforsecurity=False).hexdigest()
    assert r["bytes"] == len(crudo)
    assert (r["rows_total"], r["rows_theme"], r["rows_malformed"]) == (5, 3, 0)
    assert r["last_404_at"] is None  # nunca vimos el 404: solo hay cota superior
    arts = r["articles"]
    assert isinstance(arts, list)
    assert [a["domain"] for a in arts] == [
        "econotimes.com",
        "zerohedge.com",
        "cnet.com",
    ]
    assert (
        arts[0]["title"]
        == "Siebert Analyst Sets Bitcoin Target at $175,000 Within 12 Months"
    )
    assert len(arts[0]["tone"]) == 7


def test_archivo_diario_lleva_cabecera_primero_y_un_registro_por_lote(
    tmp_path: Path,
) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    rec.collect(L0115)
    cab, lote = registros(tmp_path)
    assert cab["type"] == "header"
    assert cab["source"] == "gdelt-gkg-2.0"
    assert lote["type"] == "batch"
    assert not (tmp_path / "descarga.tmp").exists()  # la descarga temporal no queda


def test_lote_que_nunca_aparece_queda_como_hueco_explicito(tmp_path: Path) -> None:
    rec, _, srv = montar(tmp_path, "2026-10-20T01:30:00", {}, give_up_minutes=10)
    r = rec.collect(t("2026-10-20T01:30:00"))
    assert r["status"] == "missing"
    assert r["attempts"] == 11  # 01:30 a 01:40, uno por minuto
    assert r["last_404_at"] == "2026-10-20T01:40:00.000Z"
    assert "url" not in r
    assert len(srv.pedidos) == 11


def test_hueco_de_un_rotulo_viejo_no_se_espera(tmp_path: Path) -> None:
    # Recuperacion tras una caida: el 01:30 ya paso hace horas y da 404.
    rec, _, srv = montar(tmp_path, "2026-10-20T05:00:00", {})
    r = rec.collect(t("2026-10-20T01:30:00"))
    assert (r["status"], r["attempts"]) == ("missing", 1)
    assert len(srv.pedidos) == 1


def test_error_http_en_rotulo_viejo_se_reintenta_cinco_minutos(tmp_path: Path) -> None:
    rec, _, srv = montar(tmp_path, "2026-10-20T05:00:00", {})
    srv.forzar = [Fetched(500)] * 50
    r = rec.collect(t("2026-10-20T01:30:00"))
    assert r["status"] == "missing"
    assert r["attempts"] == 6  # 05:00 a 05:05
    assert r["last_error"] == "http 500"


def test_zip_corrupto_se_reintenta_y_luego_se_registra_el_bueno(tmp_path: Path) -> None:
    rec, _, srv = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    srv.forzar = [b"esto no es un zip"]
    r = rec.collect(L0115)
    assert (r["status"], r["attempts"]) == ("ok", 2)
    assert r["rows_theme"] == 3


def test_primer_rotulo_sin_historia_es_el_del_instante(tmp_path: Path) -> None:
    rec, _, _ = montar(tmp_path, "2026-10-20T01:40:43", {})
    assert rec.first_label() == t("2026-10-20T01:30:00")


def test_primer_rotulo_retoma_despues_del_ultimo_grabado(tmp_path: Path) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    rec.collect(L0115)
    otro, _, _ = montar(
        tmp_path, "2026-10-20T02:00:00", {}
    )  # reinicio, mismo directorio
    assert otro.first_label() == t("2026-10-20T01:30:00")


def test_recuperacion_tras_caida_larga_tiene_tope(tmp_path: Path) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    rec.collect(L0115)
    otro, _, _ = montar(tmp_path, "2026-10-21T00:00:00", {}, max_catchup_batches=4)
    # 4 lotes antes del 00:00 del dia 21 = 23:00; mas nuevo que el 01:30 del 20.
    assert otro.first_label() == t("2026-10-20T23:00:00")


def test_ultima_linea_rota_por_una_caida_no_bloquea_el_reinicio(tmp_path: Path) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:20:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    rec.collect(L0115)
    with open(tmp_path / "news-2026-10-20.jsonl", "ab") as f:
        f.write(b'{"type": "batch", "label": "2026-10-20T01:3')  # sin salto de linea
    assert NewsStore(tmp_path, {}).last_label() == L0115
    otro, _, _ = montar(
        tmp_path, "2026-10-20T01:50:00", {"20261020013000": "2026-10-20T01:40:00"}
    )
    otro.collect(t("2026-10-20T01:30:00"))  # el append separa la linea rota
    ultimo = (tmp_path / "news-2026-10-20.jsonl").read_text().splitlines()[-1]
    assert json.loads(ultimo)["label"] == "2026-10-20T01:30:00.000Z"


def test_run_graba_lotes_consecutivos_y_refresca_el_heartbeat(tmp_path: Path) -> None:
    disponible = {
        "20261020004500": "2026-10-20T00:34:00",
        "20261020010000": "2026-10-20T00:49:00",
    }
    rec, _, srv = montar(tmp_path, "2026-10-20T00:50:00", disponible)
    rec.run(lambda: len(registros(tmp_path)) >= 3)  # cabecera + 2 lotes
    lotes = [r for r in registros(tmp_path) if r["type"] == "batch"]
    assert [r["label"] for r in lotes] == [
        "2026-10-20T00:45:00.000Z",
        "2026-10-20T01:00:00.000Z",
    ]
    # El 01:00 se espera hasta rotulo - 15 min = 00:45 y recien entonces se pide.
    assert min(h for h, r in srv.pedidos if r == "20261020010000") >= t(
        "2026-10-20T00:45:00"
    )
    hb = read_heartbeat(tmp_path)
    assert hb is not None
    assert hb["last_label"] == "2026-10-20T01:00:00.000Z"
    assert hb["consecutive_missing"] == 0


def test_heartbeat_cuenta_huecos_seguidos(tmp_path: Path) -> None:
    rec, _, _ = montar(tmp_path, "2026-10-20T05:00:00", {})
    rec.collect(t("2026-10-20T01:30:00"))
    rec.collect(t("2026-10-20T01:45:00"))
    hb = read_heartbeat(tmp_path)
    assert hb is not None
    assert (hb["last_status"], hb["consecutive_missing"]) == ("missing", 2)


def test_salud_segun_la_edad_del_heartbeat() -> None:
    ahora = t("2026-10-20T02:00:00")
    assert evaluate({"at": "2026-10-20T01:30:00.000Z"}, ahora, 45).ok
    atrasado = evaluate({"at": "2026-10-20T01:00:00.000Z"}, ahora, 45)
    assert not atrasado.ok
    assert atrasado.reasons == ("heartbeat atrasado mas de 45 min",)
    assert not evaluate(None, ahora, 45).ok
    assert not evaluate({}, ahora, 45).ok


def test_config_por_entorno(tmp_path: Path) -> None:
    c = NewsConfig.from_env({"NEWS_DATA_DIR": str(tmp_path), "NEWS_POLL_SECONDS": "30"})
    assert (c.poll_seconds, c.theme, c.give_up_minutes) == (30.0, "ECON_BITCOIN", 120)
    assert c.describe()["data_dir"] == str(tmp_path)


@pytest.mark.parametrize(
    ("entorno", "mensaje"),
    [
        ({}, "falta NEWS_DATA_DIR"),
        (
            {"NEWS_DATA_DIR": "/x", "NEWS_POLL_SECONDS": "rapido"},
            "valor numerico invalido",
        ),
        (
            {"NEWS_DATA_DIR": "/x", "NEWS_BASE_URL": "http://data.gdeltproject.org"},
            "debe ser https",
        ),
        ({"NEWS_DATA_DIR": "/x", "NEWS_POLL_SECONDS": "0"}, "deben ser positivos"),
        (
            {"NEWS_DATA_DIR": "/x", "NEWS_HEARTBEAT_MAX_AGE_MINUTES": "15"},
            "debe superar",
        ),
    ],
)
def test_config_invalida_falla_con_motivo(
    entorno: dict[str, str], mensaje: str
) -> None:
    with pytest.raises(ConfigError, match=mensaje):
        NewsConfig.from_env(entorno)
