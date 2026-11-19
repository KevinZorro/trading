"""Evaluacion de la prediccion pre-registrada sobre el spread (ADR 0007, seccion 6).

Lee los ``snapshots-*.jsonl`` que escribe ``live.shadow`` y aplica la tabla de
veredicto del ADR tal cual: P1 (mediana de ``ratio_spread`` <= 10^-2,5), P2 por
tamano y lado (``media(real_rel) < media(model_rel) / 2``) y NO EVALUABLE con menos
de 25 de 30 diarios validos o menos de 10 con spread del modelo positivo.

Vive en ``eval/`` y no en ``live/`` porque es una funcion pura sobre registros y
porque importar ``live`` arrastra torch: el script tiene que correr en cualquier PC
contra una copia de los datos.

La prediccion se evalua **una vez**, al cerrar la ventana. Antes de eso
:func:`progress` solo cuenta snapshots validos; no calcula ningun cociente.

ASSUMPTION: si un dia tiene mas de un snapshot diario (no deberia: ``next_due`` es
estrictamente posterior a ``now``), cuenta el primero escrito, sin mirar si es
valido. Los demas se cuentan como duplicados en el reporte.

ASSUMPTION: una celda con el libro agotado (``exhausted``) no entra a las medias:
su costo es el de lo llenado y subestima el real. Si una celda de P2 tiene aunque
sea un agotado, P2 no se puede afirmar y esa celda cuenta como fallida.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DIAS_VENTANA = 30
MIN_VALIDOS = 25
MIN_SPREAD_POSITIVO = 10
UMBRAL_P1 = 10**-2.5
# Igual que live.health.GRACIA_DIARIA: un dia cuenta como transcurrido diez minutos
# despues de su snapshot diario.
GRACIA = np.timedelta64(10, "m")
BPS = 1e4
CELDAS_ADR = {(s, lado) for s in (10.0, 100.0, 1000.0) for lado in ("buy", "sell")}
_DIA = np.timedelta64(1, "D")


class ShadowEvalError(ValueError):
    """Datos que no permiten evaluar: se rechaza con el motivo."""


def ts(texto: str) -> np.datetime64:
    return np.datetime64(str(texto).rstrip("Z"), "ns")


def iso(instante: np.datetime64) -> str:
    return str(np.datetime_as_string(np.datetime64(instante, "s"))) + "Z"


@dataclass(frozen=True)
class Snapshots:
    headers: list[dict[str, Any]]
    records: list[dict[str, Any]]
    broken_lines: int
    files: dict[str, str]  # nombre -> SHA256


def read_snapshots(directory: Path) -> Snapshots:
    """Archivos en orden de fecha y lineas en orden de escritura."""
    rutas = sorted(directory.glob("snapshots-*.jsonl"))
    if not rutas:
        raise ShadowEvalError(f"no hay snapshots-*.jsonl en {directory}")
    cabeceras: list[dict[str, Any]] = []
    registros: list[dict[str, Any]] = []
    rotas = 0
    archivos: dict[str, str] = {}
    for ruta in rutas:
        datos = ruta.read_bytes()
        archivos[ruta.name] = hashlib.sha256(datos).hexdigest()
        for linea in datos.decode("utf-8").splitlines():
            if not linea.strip():
                continue
            try:
                dato = json.loads(linea)
            except ValueError:
                rotas += 1  # ultima linea rota por una caida
                continue
            if dato.get("type") == "header":
                cabeceras.append(dato)
            elif dato.get("type") == "snapshot":
                registros.append(dato)
    return Snapshots(cabeceras, registros, rotas, archivos)


@dataclass(frozen=True)
class Window:
    first_valid: np.datetime64  # hora programada del primer diario valido

    @property
    def start(self) -> np.datetime64:
        return np.datetime64(self.first_valid, "D")

    def days(self) -> list[str]:
        return [str(self.start + i * _DIA) for i in range(DIAS_VENTANA)]

    def due(self, dia: str) -> np.datetime64:
        """Hora en que el diario de ``dia`` deja de estar pendiente."""
        hora = self.first_valid - np.datetime64(self.start, "ns")
        return np.datetime64(np.datetime64(dia, "D"), "ns") + hora + GRACIA

    @property
    def closes_at(self) -> np.datetime64:
        return self.due(self.days()[-1])


def window_of(records: Sequence[dict[str, Any]]) -> Window:
    validos = [
        r["scheduled"] for r in records if r.get("kind") == "daily" and r["valid"]
    ]
    if not validos:
        raise ShadowEvalError("no hay ningun snapshot diario valido: no hay ventana")
    # Mismo formato ISO en todos los registros: el orden de texto es el de tiempo.
    return Window(ts(min(map(str, validos))))


def daily_by_day(
    records: Sequence[dict[str, Any]], window: Window
) -> tuple[dict[str, dict[str, Any]], int]:
    """Primer diario de cada dia de la ventana, y cuantos duplicados se ignoraron."""
    dias = set(window.days())
    elegido: dict[str, dict[str, Any]] = {}
    duplicados = 0
    for r in records:
        dia = str(r["scheduled"])[:10]
        if r.get("kind") != "daily" or dia not in dias:
            continue
        if dia in elegido:
            duplicados += 1
        else:
            elegido[dia] = r
    return elegido, duplicados


def spread_ratio(record: dict[str, Any]) -> float | None:
    """``real.half_spread_rel / model.half_spread_rel``; ``None`` si el modelo da 0.

    El medio spread no depende del tamano ni del lado: hay un valor por snapshot.
    Se verifica que todas las celdas coincidan en vez de suponerlo.
    """
    celdas = record["cells"]
    if not celdas or any(c["model"] is None for c in celdas):
        raise ShadowEvalError(f"snapshot {record['scheduled']} valido sin modelo")
    reales = [float(c["real"]["half_spread_rel"]) for c in celdas]
    modelos = {float(c["model"]["half_spread_rel"]) for c in celdas}
    if len(modelos) != 1 or not np.allclose(reales, reales[0], rtol=1e-9, atol=0):
        raise ShadowEvalError(
            f"snapshot {record['scheduled']}: el medio spread difiere entre celdas"
        )
    modelo = modelos.pop()
    return reales[0] / modelo if modelo > 0 else None


@dataclass(frozen=True)
class Progress:
    window: Window
    elapsed: int
    valid: int
    invalid: int
    missing: int
    with_model_spread: int

    @property
    def remaining(self) -> int:
        return DIAS_VENTANA - self.elapsed

    @property
    def max_reachable(self) -> int:
        return self.valid + self.remaining

    @property
    def already_not_evaluable(self) -> bool:
        return self.max_reachable < MIN_VALIDOS


def progress(snaps: Snapshots, now: np.datetime64) -> Progress:
    """Cuenta de validos hasta ``now``. No calcula ningun cociente."""
    ventana = window_of(snaps.records)
    elegido, _ = daily_by_day(snaps.records, ventana)
    ahora = np.datetime64(now, "ns")
    transcurridos = [d for d in ventana.days() if ventana.due(d) <= ahora]
    validos = [
        elegido[d] for d in transcurridos if d in elegido and elegido[d]["valid"]
    ]
    positivos = sum(1 for r in validos if spread_ratio(r) is not None)
    presentes = sum(1 for d in transcurridos if d in elegido)
    return Progress(
        window=ventana,
        elapsed=len(transcurridos),
        valid=len(validos),
        invalid=presentes - len(validos),
        missing=len(transcurridos) - presentes,
        with_model_spread=positivos,
    )


@dataclass(frozen=True)
class CellStats:
    """Una celda (tamano, lado). Medias en bps sobre los snapshots no agotados."""

    size: float
    side: str
    n: int
    exhausted: int
    ratio_none: int
    ratio_total: dict[str, float] | None  # mediana, p25, p75, min, max
    real_half_spread_bps: float | None
    real_depth_impact_bps: float | None
    real_bps: float | None
    real_bps_median: float | None
    model_half_spread_bps: float | None
    model_slippage_bps: float | None
    model_bps: float | None
    model_bps_median: float | None

    @property
    def key(self) -> str:
        return f"{self.size:g}/{self.side}"


def _resumen(valores: list[float]) -> dict[str, float] | None:
    if not valores:
        return None
    a = np.asarray(valores, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _media_bps(valores: list[float]) -> float | None:
    return float(np.mean(valores)) * BPS if valores else None


def _mediana_bps(valores: list[float]) -> float | None:
    return float(np.median(valores)) * BPS if valores else None


def cell_stats(records: Sequence[dict[str, Any]]) -> list[CellStats]:
    """Estadisticas por (tamano, lado) sobre snapshots validos."""
    celdas: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for r in records:
        for c in r["cells"]:
            if c["model"] is None:
                raise ShadowEvalError(f"snapshot {r['scheduled']} valido sin modelo")
            celdas.setdefault((float(c["size"]), str(c["side"])), []).append(c)
    salida: list[CellStats] = []
    for (tamano, lado), lista in celdas.items():
        llenas = [c for c in lista if not c["real"]["exhausted"]]
        cocientes = [c["ratios"]["ratio_total"] for c in llenas]
        real = [float(c["real"]["real_rel"]) for c in llenas]
        modelo = [float(c["model"]["model_rel"]) for c in llenas]
        salida.append(
            CellStats(
                size=tamano,
                side=lado,
                n=len(llenas),
                exhausted=len(lista) - len(llenas),
                ratio_none=sum(1 for x in cocientes if x is None),
                ratio_total=_resumen([float(x) for x in cocientes if x is not None]),
                real_half_spread_bps=_media_bps(
                    [c["real"]["half_spread_rel"] for c in llenas]
                ),
                real_depth_impact_bps=_media_bps(
                    [c["real"]["depth_impact_rel"] for c in llenas]
                ),
                real_bps=_media_bps(real),
                real_bps_median=_mediana_bps(real),
                model_half_spread_bps=_media_bps(
                    [c["model"]["half_spread_rel"] for c in llenas]
                ),
                model_slippage_bps=_media_bps(
                    [c["model"]["slippage_rel"] for c in llenas]
                ),
                model_bps=_media_bps(modelo),
                model_bps_median=_mediana_bps(modelo),
            )
        )
    return salida


def p2_holds(cell: CellStats) -> bool:
    """``media(real_rel) < media(model_rel) / 2``, sin agotados (ver ASSUMPTION)."""
    if cell.exhausted or cell.real_bps is None or cell.model_bps is None:
        return False
    return cell.real_bps < cell.model_bps / 2


@dataclass(frozen=True)
class Evaluation:
    verdict: str
    progress: Progress
    duplicates: int
    p1_median: float | None
    p1: bool | None
    p2_failed: list[str]
    daily: list[CellStats]
    periodic: list[CellStats]
    invalid_reasons: dict[str, list[str]]


def evaluate(snaps: Snapshots, now: np.datetime64, adr_sha256: str) -> Evaluation:
    """El veredicto del ADR 0007 §6. Falla si la ventana sigue abierta."""
    distintas = {h.get("adr_sha256") for h in snaps.headers} - {adr_sha256}
    if not snaps.headers or distintas:
        raise ShadowEvalError(
            f"las cabeceras no llevan el SHA256 del ADR 0007 ({adr_sha256}); "
            f"traen {sorted(map(str, distintas)) or 'ninguna cabecera'}"
        )
    ventana = window_of(snaps.records)
    if np.datetime64(now, "ns") < ventana.closes_at:
        raise ShadowEvalError(
            f"la ventana cierra en {iso(ventana.closes_at)}: la prediccion se "
            "evalua una vez, al cerrar"
        )
    avance = progress(snaps, now)
    elegido, duplicados = daily_by_day(snaps.records, ventana)
    diarios = [r for r in elegido.values() if r["valid"]]
    inicio = np.datetime64(ventana.start, "ns")
    fin = inicio + DIAS_VENTANA * _DIA
    periodicos = [
        r
        for r in snaps.records
        if r.get("kind") == "periodic"
        and r["valid"]
        and inicio <= ts(r["scheduled"]) < fin
    ]
    estadisticas = cell_stats(diarios)
    if {(c.size, c.side) for c in estadisticas} != CELDAS_ADR:
        raise ShadowEvalError(
            "P2 se define sobre 3 tamanos x 2 lados (10, 100 y 1000 USDT); los "
            f"snapshots traen {sorted(c.key for c in estadisticas)}"
        )
    fallidas = [c.key for c in estadisticas if not p2_holds(c)]
    cocientes = [x for x in map(spread_ratio, diarios) if x is not None]
    mediana: float | None = None
    p1: bool | None = None
    if avance.valid < MIN_VALIDOS or len(cocientes) < MIN_SPREAD_POSITIVO:
        veredicto = "NO EVALUABLE"
    else:
        mediana = float(np.median(cocientes))
        p1 = mediana <= UMBRAL_P1
        veredicto = "PARCIAL" if fallidas else "SOSTENIDA"
        if not p1:
            veredicto = "REFUTADA"
    return Evaluation(
        verdict=veredicto,
        progress=avance,
        duplicates=duplicados,
        p1_median=mediana,
        p1=p1,
        p2_failed=fallidas,
        daily=estadisticas,
        periodic=cell_stats(periodicos),
        invalid_reasons={
            d: list(r["reasons"]) for d, r in sorted(elegido.items()) if not r["valid"]
        },
    )
