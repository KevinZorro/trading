"""Klines de Binance: paginacion, parseo, dataset versionado y reporte.

Las klines de estos tests se construyen a mano en el formato documentado de
``GET /api/v3/klines`` (doce campos, precios como strings, tiempos en ms). No
son una respuesta grabada: los valores se eligen para que los oraculos se
puedan verificar de cabeza. La fidelidad al formato real la cubre el test
marcado ``network``, que el CI no corre.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data import AlwaysOpen, binance_spot_spec
from data.binance import (
    CSV_COLUMNS,
    MAX_LIMIT,
    build_dataset,
    fetch_klines,
    klines_to_frame,
    write_klines_csv,
)
from data.errors import SchemaError
from data.loaders import load_versioned_bars
from data.manifest import (
    ManifestError,
    manifest_path_for,
    read_manifest,
    sha256_of,
    verify_manifest,
)
from data.validation import validation_report

HORA_MS = 3_600_000
DIA_MS = 24 * HORA_MS
T0 = pd.Timestamp("2022-01-01", tz="UTC")
T0_MS = int(T0.value // 1_000_000)


def kline(
    open_ms: int, paso_ms: int, precio: float = 100.0, *, close_ms: int | None = None
) -> list[Any]:
    """Una kline en el formato exacto del venue. ``close_ms`` fuerza un cierre
    anomalo; por defecto es el teorico, 1 ms antes de la siguiente."""
    return [
        open_ms,
        f"{precio:.2f}",
        f"{precio + 2:.2f}",
        f"{precio - 1:.2f}",
        f"{precio + 1:.2f}",
        "10.50000000",
        open_ms + paso_ms - 1 if close_ms is None else close_ms,
        "1050.00000000",
        42,
        "5.25000000",
        "525.00000000",
        "0",
    ]


class VenueFalso:
    """Sirve ``filas`` respetando ``startTime``, ``endTime`` y ``limit``."""

    def __init__(self, filas: list[list[Any]], *, max_pagina: int = MAX_LIMIT) -> None:
        self.filas = filas
        self.max_pagina = max_pagina
        self.llamadas: list[dict[str, str | int]] = []

    def __call__(self, url: str, params: dict[str, str | int]) -> Any:
        self.llamadas.append(dict(params))
        desde, hasta = int(params["startTime"]), int(params["endTime"])
        limite = min(int(params["limit"]), self.max_pagina)
        dentro = [f for f in self.filas if desde <= int(f[0]) <= hasta]
        return dentro[:limite]


def ahora(ms: int) -> np.datetime64:
    return np.datetime64(ms, "ms").astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Paginacion
# ---------------------------------------------------------------------------


def test_pagina_hasta_traer_todo_sin_repetir() -> None:
    """2500 horas en paginas de 1000: tres llamadas, cada fila una vez."""
    filas = [kline(T0_MS + i * HORA_MS, HORA_MS) for i in range(2500)]
    venue = VenueFalso(filas)
    traidas = fetch_klines(
        venue,
        symbol="BTCUSDT",
        interval="1h",
        start=T0,
        end=T0 + pd.Timedelta(hours=2500),
    )
    assert len(traidas) == 2500
    assert [int(f[0]) for f in traidas] == [int(f[0]) for f in filas]
    # Cada pagina arranca 1 ms despues del open_time de la ultima fila traida.
    # La cuarta vuelve vacia y corta: no se asume que una pagina corta sea la
    # ultima, porque eso seria asumir el tamano de pagina del venue.
    assert [c["startTime"] for c in venue.llamadas] == [
        T0_MS,
        T0_MS + 999 * HORA_MS + 1,
        T0_MS + 1999 * HORA_MS + 1,
        T0_MS + 2499 * HORA_MS + 1,
    ]


def test_no_asume_el_tamano_de_pagina_del_venue() -> None:
    """Si el venue baja su maximo a 300, se sigue paginando igual."""
    filas = [kline(T0_MS + i * HORA_MS, HORA_MS) for i in range(700)]
    traidas = fetch_klines(
        VenueFalso(filas, max_pagina=300),
        symbol="BTCUSDT",
        interval="1h",
        start=T0,
        end=T0 + pd.Timedelta(hours=700),
    )
    assert len(traidas) == 700


def test_paginacion_que_no_avanza_es_un_error() -> None:
    fila = kline(T0_MS, HORA_MS)

    def venue_roto(url: str, params: dict[str, str | int]) -> Any:
        return [fila]

    with pytest.raises(SchemaError, match="la paginacion no avanza"):
        fetch_klines(
            venue_roto,
            symbol="BTCUSDT",
            interval="1h",
            start=T0,
            end=T0 + pd.Timedelta(hours=5),
        )


def test_respuesta_que_no_es_lista_es_un_error() -> None:
    def venue_error(url: str, params: dict[str, str | int]) -> Any:
        return {"code": -1121, "msg": "Invalid symbol."}

    with pytest.raises(SchemaError, match="respuesta inesperada"):
        fetch_klines(
            venue_error,
            symbol="XXX",
            interval="1d",
            start=T0,
            end=T0 + pd.Timedelta(days=2),
        )


def test_intervalo_no_soportado() -> None:
    with pytest.raises(SchemaError, match="intervalo no soportado"):
        fetch_klines(VenueFalso([]), symbol="BTCUSDT", interval="5m", start=T0, end=T0)


# ---------------------------------------------------------------------------
# Parseo
# ---------------------------------------------------------------------------


def test_el_timestamp_es_el_cierre_de_la_barra() -> None:
    """La kline que abre el 2022-01-01 se fecha el 2022-01-02: todo lo que la
    fila contiene se conoce recien a esa hora."""
    frame = klines_to_frame(
        [kline(T0_MS, DIA_MS)],
        symbol="BTCUSDT",
        interval="1d",
        now=ahora(T0_MS + 2 * DIA_MS),
    )
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2022-01-02", tz="UTC")
    assert frame["open_time"].iloc[0] == pd.Timestamp("2022-01-01", tz="UTC")
    assert tuple(frame.columns) == CSV_COLUMNS
    fila = frame.iloc[0]
    assert (fila["open"], fila["high"], fila["low"], fila["close"]) == (
        100.0,
        102.0,
        99.0,
        101.0,
    )
    assert fila["volume"] == 10.5
    assert fila["n_trades"] == 42
    assert fila["symbol"] == "BTCUSDT"


def test_descarta_la_barra_en_curso() -> None:
    """Con ``now`` a mitad del tercer dia, solo quedan los dos cerrados."""
    filas = [kline(T0_MS + i * DIA_MS, DIA_MS) for i in range(3)]
    frame = klines_to_frame(
        filas,
        symbol="BTCUSDT",
        interval="1d",
        now=ahora(T0_MS + 2 * DIA_MS + DIA_MS // 2),
    )
    assert len(frame) == 2


def test_la_barra_que_cierra_justo_ahora_todavia_no_cuenta() -> None:
    """close_time = open + 1d - 1ms. Con now en ese mismo milisegundo la barra
    aun no cerro; un milisegundo despues, si."""
    fila = [kline(T0_MS, DIA_MS)]
    cierre = T0_MS + DIA_MS - 1
    with pytest.raises(SchemaError, match="ninguna kline cerrada"):
        klines_to_frame(fila, symbol="BTCUSDT", interval="1d", now=ahora(cierre))
    frame = klines_to_frame(
        fila, symbol="BTCUSDT", interval="1d", now=ahora(cierre + 1)
    )
    assert len(frame) == 1


def test_detecta_timestamps_en_microsegundos() -> None:
    """Los volcados de data.binance.vision pasaron a microsegundos en 2025.
    Se detecta por la magnitud del epoch, aunque sea una sola fila."""
    fila = kline(T0_MS * 1000, DIA_MS * 1000)
    with pytest.raises(SchemaError, match="unidad de tiempo"):
        klines_to_frame(
            [fila], symbol="BTCUSDT", interval="1d", now=ahora(T0_MS + 5 * DIA_MS)
        )


def test_un_close_time_en_microsegundos_tambien_se_detecta() -> None:
    fila = kline(T0_MS, DIA_MS, close_ms=(T0_MS + DIA_MS - 1) * 1000)
    with pytest.raises(SchemaError, match="close_time fuera del rango"):
        klines_to_frame(
            [fila], symbol="BTCUSDT", interval="1d", now=ahora(T0_MS + 5 * DIA_MS)
        )


# ---------------------------------------------------------------------------
# Anomalias del venue: se conservan, se marcan, no se reparan
# ---------------------------------------------------------------------------


def _horas(n: int, anomalas: set[int], *, adelanto_ms: int = -1_905_211) -> list[Any]:
    """``n`` klines horarias; las de ``anomalas`` cierran ``adelanto_ms`` antes
    del teorico (el desvio del 8-feb-2018 en la barra horaria)."""
    return [
        kline(
            T0_MS + i * HORA_MS,
            HORA_MS,
            close_ms=T0_MS + (i + 1) * HORA_MS - 1 + adelanto_ms
            if i in anomalas
            else None,
        )
        for i in range(n)
    ]


def _convertir(filas: list[Any], n: int) -> Any:
    return klines_to_frame(
        filas, symbol="BTCUSDT", interval="1h", now=ahora(T0_MS + (n + 1) * HORA_MS)
    )


def test_la_kline_cerrada_antes_conserva_el_cierre_teorico() -> None:
    """Caso real del 8-feb-2018: la barra horaria de las 00:00 cerro a las
    00:28:14.788. Lleva timestamp 01:00 -sigue en la rejilla y es point-in-time,
    porque a las 01:00 la fila ya se conoce entera- y queda marcada."""
    frame = _convertir(_horas(3, {1}), 3)
    assert len(frame) == 3
    assert frame["timestamp"].iloc[1] == T0 + pd.Timedelta(hours=2)
    assert frame["close_time_desvio_ms"].tolist() == [0, -1_905_211, 0]


def test_la_kline_vacia_de_una_caida_se_conserva() -> None:
    """Caso real del 6-sep-2017 16:00: close_time == open_time, volumen cero,
    OHLC plano. No se descarta ni se rellena: se marca."""
    vacia = kline(T0_MS, HORA_MS, close_ms=T0_MS)
    vacia[5] = "0.00000000"
    vacia[8] = 0
    frame = _convertir([vacia, kline(T0_MS + HORA_MS, HORA_MS)], 2)
    assert frame["close_time_desvio_ms"].iloc[0] == -(HORA_MS - 1)
    assert frame["volume"].iloc[0] == 0.0
    assert frame["timestamp"].iloc[0] == T0 + pd.Timedelta(hours=1)


def test_si_el_venue_cerrara_despues_manda_el_cierre_real() -> None:
    """No aparece en BTCUSDT, pero la regla tiene que ser correcta en las dos
    direcciones: con el cierre teorico la fila contendria operaciones
    posteriores a su propio timestamp. Sale de la rejilla y el calendario lo
    senala."""
    tarde = T0_MS + HORA_MS - 1 + 600_000  # diez minutos despues del teorico
    filas = [kline(T0_MS, HORA_MS, close_ms=tarde), kline(T0_MS + 2 * HORA_MS, HORA_MS)]
    frame = klines_to_frame(
        filas, symbol="BTCUSDT", interval="1h", now=ahora(T0_MS + 4 * HORA_MS)
    )
    assert frame["timestamp"].iloc[0] == pd.Timestamp(tarde + 1, unit="ms", tz="UTC")
    assert frame["close_time_desvio_ms"].iloc[0] == 600_000
    frame["split_factor"], frame["cash_dividend"] = 1.0, 0.0
    reporte = validation_report(
        frame, instrument=binance_spot_spec(), calendar=AlwaysOpen(), freq="1h"
    )
    assert dict(reporte.checks)["calendario"] is not None


def test_cinco_anomalias_en_mil_filas_pasan() -> None:
    """0.5 %: por encima del conteo pero por debajo de la fraccion."""
    frame = _convertir(_horas(1000, {10, 200, 400, 600, 800}), 1000)
    assert len(frame) == 1000
    assert int((frame["close_time_desvio_ms"] != 0).sum()) == 5


def test_veinte_anomalias_en_mil_filas_fallan() -> None:
    """2 %: por encima del conteo y de la fraccion. Ya no son caidas aisladas."""
    with pytest.raises(SchemaError, match="20 de 1000 klines"):
        _convertir(_horas(1000, set(range(0, 1000, 50))), 1000)


def test_una_descarga_corta_con_una_caida_pasa() -> None:
    """Una anomalia en diez filas es el 10 %, pero es una sola: una descarga
    corta que contiene una caida no falla por accidente."""
    frame = _convertir(_horas(10, {4}), 10)
    assert int((frame["close_time_desvio_ms"] != 0).sum()) == 1


@pytest.mark.parametrize(
    ("n", "anomalas", "falla"),
    [
        (400, 3, False),  # 0.75 % y 3 filas: ninguna condicion
        (400, 4, False),  # 1.00 % exacto: no supera la fraccion
        (400, 5, True),  # 1.25 % y 5 filas: las dos
        (4, 3, False),  # 75 % pero 3 filas: no supera el conteo
        (4, 4, True),  # 100 %: un intervalo equivocado en una descarga minima
    ],
)
def test_frontera_del_umbral_estructural(n: int, anomalas: int, falla: bool) -> None:
    filas = _horas(n, set(range(anomalas)))
    if falla:
        with pytest.raises(SchemaError, match="demasiadas para ser caidas aisladas"):
            _convertir(filas, n)
    else:
        assert len(_convertir(filas, n)) == n


def test_un_intervalo_equivocado_falla() -> None:
    """Pedir 1h y recibir klines diarias desvia el 100 % de las filas."""
    filas = [kline(T0_MS + i * DIA_MS, DIA_MS) for i in range(30)]
    with pytest.raises(SchemaError, match="30 de 30 klines"):
        klines_to_frame(
            filas, symbol="BTCUSDT", interval="1h", now=ahora(T0_MS + 40 * DIA_MS)
        )


def test_rechaza_ancho_distinto_y_lista_vacia() -> None:
    with pytest.raises(SchemaError, match="ancho"):
        klines_to_frame(
            [kline(T0_MS, DIA_MS)[:11]],
            symbol="BTCUSDT",
            interval="1d",
            now=ahora(T0_MS),
        )
    with pytest.raises(SchemaError, match="no hay klines"):
        klines_to_frame([], symbol="BTCUSDT", interval="1d", now=ahora(T0_MS))


# ---------------------------------------------------------------------------
# CSV determinista y manifest
# ---------------------------------------------------------------------------


def _frame(n: int = 5, paso_ms: int = DIA_MS, interval: str = "1d") -> pd.DataFrame:
    filas = [kline(T0_MS + i * paso_ms, paso_ms, 100.0 + i) for i in range(n)]
    return klines_to_frame(
        filas, symbol="BTCUSDT", interval=interval, now=ahora(T0_MS + (n + 1) * paso_ms)
    )


@pytest.mark.parametrize("nombre", ["x.csv", "x.csv.gz"])
def test_el_csv_es_determinista(tmp_path: Path, nombre: str) -> None:
    """Mismo frame, mismos bytes. Con gzip exige mtime=0: gzip graba la hora."""
    a, b = tmp_path / "a" / nombre, tmp_path / "b" / nombre
    a.parent.mkdir()
    b.parent.mkdir()
    write_klines_csv(_frame(), a)
    write_klines_csv(_frame(), b)
    assert sha256_of(a) == sha256_of(b)


def test_el_gz_es_gzip_de_verdad(tmp_path: Path) -> None:
    ruta = tmp_path / "x.csv.gz"
    write_klines_csv(_frame(), ruta)
    with gzip.open(ruta, "rt") as handle:
        assert handle.readline().strip() == ",".join(CSV_COLUMNS)


def test_sha256_contra_valor_conocido(tmp_path: Path) -> None:
    """SHA256 de "abc", del vector de prueba de FIPS 180-2."""
    ruta = tmp_path / "abc.txt"
    ruta.write_bytes(b"abc")
    assert sha256_of(ruta) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def _dataset(tmp_path: Path, n: int = 5) -> Path:
    filas = [kline(T0_MS + i * DIA_MS, DIA_MS, 100.0 + i) for i in range(n + 1)]
    salida = tmp_path / "BTCUSDT-1d.csv"
    build_dataset(
        VenueFalso(filas),
        symbol="BTCUSDT",
        interval="1d",
        start=T0,
        now=ahora(T0_MS + n * DIA_MS + DIA_MS // 2),  # la ultima barra sigue abierta
        out_path=salida,
        script="scripts/download_binance.py --interval 1d",
    )
    return manifest_path_for(salida)


def test_el_manifest_describe_el_archivo(tmp_path: Path) -> None:
    manifest = read_manifest(_dataset(tmp_path, n=5))
    assert manifest.file == "BTCUSDT-1d.csv"
    assert manifest.rows == 5
    assert manifest.first_timestamp == "2022-01-02T00:00:00+00:00"
    assert manifest.last_timestamp == "2022-01-06T00:00:00+00:00"
    assert manifest.sha256 == sha256_of(tmp_path / "BTCUSDT-1d.csv")
    assert manifest.downloaded_at == "2022-01-06T12:00:00+00:00"
    assert "data-api.binance.vision" in manifest.source
    assert manifest.script.startswith("scripts/download_binance.py")


def test_un_dataset_editado_no_se_carga(tmp_path: Path) -> None:
    manifest_path = _dataset(tmp_path)
    datos = tmp_path / "BTCUSDT-1d.csv"
    datos.write_text(datos.read_text().replace("104.0", "104.5"))
    with pytest.raises(ManifestError, match="no coincide"):
        verify_manifest(manifest_path)
    with pytest.raises(ManifestError, match="no coincide"):
        load_versioned_bars(manifest_path, instrument=binance_spot_spec(), freq="1D")


def test_un_manifest_sin_archivo_falla(tmp_path: Path) -> None:
    manifest_path = _dataset(tmp_path)
    (tmp_path / "BTCUSDT-1d.csv").unlink()
    with pytest.raises(ManifestError, match="inexistente"):
        verify_manifest(manifest_path)


def test_el_dataset_versionado_llega_a_una_barseries(tmp_path: Path) -> None:
    serie = load_versioned_bars(
        _dataset(tmp_path, n=5),
        instrument=binance_spot_spec(),
        freq="1D",
        calendar=AlwaysOpen(),
    )
    assert len(serie) == 5
    np.testing.assert_array_equal(serie.close, [101.0, 102.0, 103.0, 104.0, 105.0])


def test_el_gz_versionado_tambien_carga(tmp_path: Path) -> None:
    filas = [kline(T0_MS + i * HORA_MS, HORA_MS) for i in range(4)]
    salida = tmp_path / "BTCUSDT-1h.csv.gz"
    build_dataset(
        VenueFalso(filas),
        symbol="BTCUSDT",
        interval="1h",
        start=T0,
        now=ahora(T0_MS + 4 * HORA_MS),
        out_path=salida,
        script="s",
    )
    serie = load_versioned_bars(
        manifest_path_for(salida), instrument=binance_spot_spec(), freq="1h"
    )
    assert len(serie) == 4


# ---------------------------------------------------------------------------
# Reporte de validacion
# ---------------------------------------------------------------------------


def test_el_reporte_resume_los_huecos() -> None:
    """10 horas con las horas 3, 4, 5 y 8 ausentes: dos huecos, cuatro barras,
    el mas largo de tres."""
    frame = _frame(n=10, paso_ms=HORA_MS, interval="1h")
    frame = frame.drop(index=[3, 4, 5, 8]).reset_index(drop=True)
    reporte = validation_report(
        frame, instrument=binance_spot_spec(), calendar=AlwaysOpen(), freq="1h"
    )
    assert reporte.n_bars == 6
    assert reporte.gaps == 2
    assert reporte.missing_bars == 4
    assert reporte.longest_gap is not None
    assert reporte.longest_gap.startswith("3 barras desde 2022-01-01 04:00:00+00:00")
    assert not reporte.passed
    assert dict(reporte.checks)["calendario"] is not None
    assert "huecos de calendario: 2 (4 barras faltantes)" in reporte.render()


def test_el_reporte_sigue_despues_del_primer_fallo() -> None:
    """``validate_bars`` para en el primero; el reporte tiene que listarlos todos."""
    frame = _frame(n=4)
    frame.loc[1, "high"] = 50.0  # high < low
    frame.loc[2, "symbol"] = "ETHUSDT"
    reporte = validation_report(
        frame, instrument=binance_spot_spec(), calendar=AlwaysOpen(), freq="1D"
    )
    fallas = {nombre for nombre, error in reporte.checks if error is not None}
    assert fallas == {"simbolo", "precios coherentes"}
    assert reporte.gaps == 0
    assert reporte.longest_gap is None
    assert "[FALLA] simbolo" in reporte.render()


def test_un_dataset_sano_pasa_todo() -> None:
    reporte = validation_report(
        _frame(n=6), instrument=binance_spot_spec(), calendar=AlwaysOpen(), freq="1D"
    )
    assert reporte.passed
    assert reporte.missing_bars == 0


# ---------------------------------------------------------------------------
# Red real: fuera del CI
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_el_formato_real_del_venue_no_cambio() -> None:
    """Los tres primeros dias de BTCUSDT, contra la API publica de verdad."""
    import json
    import urllib.parse
    import urllib.request

    def get_json(url: str, params: dict[str, str | int]) -> Any:
        with urllib.request.urlopen(
            f"{url}?{urllib.parse.urlencode(params)}", timeout=30
        ) as respuesta:
            return json.load(respuesta)

    inicio = pd.Timestamp("2017-08-17", tz="UTC")
    filas = fetch_klines(
        get_json,
        symbol="BTCUSDT",
        interval="1d",
        start=inicio,
        end=inicio + pd.Timedelta(days=3),
    )
    frame = klines_to_frame(filas, symbol="BTCUSDT", interval="1d", now=ahora(T0_MS))
    assert len(frame) == 3
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2017-08-18", tz="UTC")
