"""Salud de la sombra (ADR 0007, PR 6): funciones puras sobre el heartbeat.

El proceso sombra escribe ``heartbeat.json`` de forma atomica en cada snapshot.
Este modulo decide si ese estado es sano, y el mismo veredicto alimenta al
``HEALTHCHECK`` de Docker y al watchdog. No consulta el reloj: ``ahora`` entra
por parametro.

La razon de fondo: la prediccion del ADR queda NO EVALUABLE con menos de 25 de
30 snapshots validos de las 00:00:30 UTC, asi que un hueco no detectado a tiempo
es un hueco que no se recupera.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from live.config import ConfigError, ShadowConfig
from live.schedule import parse_daily
from sim.clock import Clock, SystemClock

# ASSUMPTION: 10 minutos de gracia sobre la hora del snapshot diario. El snapshot
# tarda segundos; el margen cubre una respuesta lenta de Binance sin dar falsas
# alarmas. No es un reintento: el snapshot se toma una sola vez.
GRACIA_DIARIA = np.timedelta64(10, "m")
INVALIDOS_SEGUIDOS_MAX = 3


@dataclass(frozen=True)
class Health:
    ok: bool
    reasons: tuple[str, ...] = ()


def parse_ts(texto: str) -> np.datetime64:
    """``2026-10-17T17:30:00.000Z`` -> ``datetime64[ns]`` (UTC naive)."""
    return np.datetime64(texto.rstrip("Z"), "ns")


def read_heartbeat(data_dir: Path) -> Mapping[str, Any] | None:
    """``None`` si falta o esta corrupto (la escritura es atomica, no deberia)."""
    try:
        dato = json.loads((data_dir / "heartbeat.json").read_text())
    except (OSError, ValueError):
        return None
    return dato if isinstance(dato, dict) else None


def evaluate(
    heartbeat: Mapping[str, Any] | None, ahora: np.datetime64, config: ShadowConfig
) -> Health:
    if heartbeat is None:
        return Health(False, ("heartbeat.json falta o es ilegible",))
    motivos: list[str] = []
    try:
        edad = np.datetime64(ahora, "ns") - parse_ts(str(heartbeat["at"]))
    except (KeyError, ValueError):
        return Health(False, ("heartbeat.json sin campo 'at' valido",))
    maxima = np.timedelta64(config.heartbeat_max_age_minutes, "m")
    if edad > maxima:
        # Texto estable: el watchdog deduplica por motivo y una edad que cambia
        # en cada ciclo dispararia un aviso cada 5 minutos.
        motivos.append(
            f"heartbeat atrasado mas de {config.heartbeat_max_age_minutes} min"
        )
    seguidos = heartbeat.get("consecutive_invalid", 0)
    if isinstance(seguidos, int) and seguidos >= INVALIDOS_SEGUIDOS_MAX:
        motivos.append(f"{seguidos} snapshots invalidos seguidos")
    return Health(not motivos, tuple(motivos))


def daily_missing(
    registros: Iterable[Mapping[str, Any]], ahora: np.datetime64, config: ShadowConfig
) -> str | None:
    """Motivo si, pasada la gracia, no hay snapshot diario valido del dia UTC."""
    ahora = np.datetime64(ahora, "ns")
    hoy = ahora.astype("datetime64[D]")
    limite = hoy.astype("datetime64[ns]") + parse_daily(config.daily_snapshot_utc)
    if ahora < limite + GRACIA_DIARIA:
        return None
    dia = str(hoy)
    for r in registros:
        if (
            r.get("type") == "snapshot"
            and r.get("kind") == "daily"
            and r.get("valid") is True
            and str(r.get("scheduled", "")).startswith(dia)
        ):
            return None
    return f"falta el snapshot diario valido de {dia}"


def read_daily_records(data_dir: Path, ahora: np.datetime64) -> list[Mapping[str, Any]]:
    ruta = data_dir / f"snapshots-{np.datetime64(ahora, 'D')}.jsonl"
    try:
        lineas = ruta.read_text().splitlines()
    except OSError:
        return []
    salida: list[Mapping[str, Any]] = []
    for linea in lineas:
        try:
            dato = json.loads(linea)
        except ValueError:
            continue  # ultima linea rota por una caida
        if isinstance(dato, dict):
            salida.append(dato)
    return salida


def check(data_dir: Path, ahora: np.datetime64, config: ShadowConfig) -> Health:
    """Veredicto completo: heartbeat y snapshot diario."""
    base = evaluate(read_heartbeat(data_dir), ahora, config)
    falta = daily_missing(read_daily_records(data_dir, ahora), ahora, config)
    motivos = base.reasons + ((falta,) if falta else ())
    return Health(not motivos, motivos)


def main(
    environ: Mapping[str, str], clock: Clock | None = None
) -> int:  # pragma: no cover - HEALTHCHECK real
    try:
        config = ShadowConfig.from_env(environ)
    except ConfigError as exc:
        print(f"config invalida: {exc}", file=sys.stderr)
        return 1
    # Solo el heartbeat: el HEALTHCHECK de Docker no debe marcar 'unhealthy' a las
    # 00:10 por un hueco diario (eso es del watchdog, que avisa a Telegram).
    salud = evaluate(
        read_heartbeat(config.data_dir), (clock or SystemClock()).now(), config
    )
    for motivo in salud.reasons:
        print(motivo, file=sys.stderr)
    return 0 if salud.ok else 1


if __name__ == "__main__":  # pragma: no cover
    import os

    raise SystemExit(main(os.environ))
