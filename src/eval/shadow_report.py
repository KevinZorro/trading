"""Reporte de la sombra de costos (ADR 0007, secciones 3 y 6) y su linea de comandos.

Uso, contra una copia de ``/var/lib/shadow`` (no necesita torch ni red)::

    uv run python scripts/evaluar_sombra.py <carpeta> [--out results/sombra]

Con la ventana abierta solo imprime el avance. Al cerrar escribe ``REPORTE.md``,
``summary.json`` y ``MANIFEST.json`` (SHA256 de cada archivo de entrada y de las
salidas): los snapshots completos no se commitean, el manifest ata el resumen a ellos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.shadow import (
    DIAS_VENTANA,
    MIN_SPREAD_POSITIVO,
    MIN_VALIDOS,
    UMBRAL_P1,
    CellStats,
    Evaluation,
    Progress,
    evaluate,
    iso,
    p2_holds,
    progress,
    read_snapshots,
)
from sim.clock import Clock, SystemClock

ADR_PATH = (
    Path(__file__).resolve().parents[2] / "docs/adr/0007-modo-sombra-y-testnet.md"
)

LIMITACIONES = """\
- **N = 1.** Un solo camino de mercado, un simbolo y un venue. Treinta dias de un
  mismo libro no son treinta observaciones independientes del regimen de liquidez.
- **Alcance sobre C3.** La conclusion es sobre el test de C3 (2020-05 a 2026-05) y
  exige que el libro haya estado a pocos ticks tambien entonces. No hay L2 historico
  para verificarlo.
- **Sin comision.** La sombra no opera y no ve la tarifa efectiva de una cuenta; eso
  queda para la Etapa 7.
- **Foto, no orden.** `walk_book` recorre una foto del libro: es un piso del costo de
  una orden de mercado, sin latencia ni cola (ADR 0007 §8).
- **Rollover.** El snapshot de las 00:00:30 cae en el cambio de dia UTC, cuya liquidez
  puede no ser la del promedio; los de 15 minutos se reportan aparte y no deciden.
- **Los Agentes A en sombra no deciden.** Sus fills miden la misma realidad con otros
  tamanos de orden (ADR 0009 §3); no entran a este veredicto.
"""


def _f(x: float | None, formato: str = ".3f") -> str:
    return "—" if x is None else format(x, formato)


def render_progress(p: Progress) -> str:
    v = p.window
    lineas = [
        f"Ventana: {v.days()[0]} a {v.days()[-1]} (cierra {iso(v.closes_at)})",
        f"Dias transcurridos: {p.elapsed} de {DIAS_VENTANA}; faltan {p.remaining}",
        f"Diarios validos: {p.valid}  invalidos: {p.invalid}  faltantes: {p.missing}",
        f"Con spread del modelo > 0: {p.with_model_spread} "
        f"(minimo {MIN_SPREAD_POSITIVO})",
        f"Maximo alcanzable: {p.max_reachable} validos (minimo {MIN_VALIDOS})",
    ]
    if p.already_not_evaluable:
        lineas.append("YA ES NO EVALUABLE: no se puede llegar a 25 validos.")
    else:
        margen = p.max_reachable - MIN_VALIDOS
        lineas.append(f"Margen: se pueden perder {margen} diario(s) mas.")
    return "\n".join(lineas)


def _tabla_celdas(celdas: Sequence[CellStats]) -> list[str]:
    filas = [
        "| Tamaño | Lado | n | Agotados | `None` | ratio_total mediana [p25, p75] "
        "(min, max) | Real: medio spread / profundidad / total (bps) | Modelo: medio "
        "spread / slippage / total (bps) | Mediana real / modelo (bps) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in celdas:
        r = c.ratio_total
        cociente = (
            "—"
            if r is None
            else f"{r['median']:.3g} [{r['p25']:.3g}, {r['p75']:.3g}] "
            f"({r['min']:.3g}, {r['max']:.3g})"
        )
        filas.append(
            f"| {c.size:g} | {c.side} | {c.n} | {c.exhausted} | {c.ratio_none} | "
            f"{cociente} | {_f(c.real_half_spread_bps, '.4f')} / "
            f"{_f(c.real_depth_impact_bps, '.4f')} / {_f(c.real_bps, '.4f')} | "
            f"{_f(c.model_half_spread_bps, '.2f')} / {_f(c.model_slippage_bps, '.4f')} "
            f"/ {_f(c.model_bps, '.2f')} | {_f(c.real_bps_median, '.4f')} / "
            f"{_f(c.model_bps_median, '.2f')} |"
        )
    return filas


def render_report(ev: Evaluation, adr_sha256: str, n_headers: int) -> str:
    p = ev.progress
    dias = p.window.days()
    lineas = [
        "# Evaluación de la sombra de costos (ADR 0007 §6)",
        "",
        f"**Veredicto: {ev.verdict}**",
        "",
        f"- Ventana: {dias[0]} a {dias[-1]}, {DIAS_VENTANA} días UTC desde el primer "
        f"snapshot diario válido ({iso(p.window.first_valid)}).",
        f"- ADR 0007 SHA256 `{adr_sha256}`, verificado en {n_headers} cabecera(s).",
        f"- Diarios válidos: {p.valid} de {DIAS_VENTANA} (mínimo {MIN_VALIDOS}); "
        f"inválidos {p.invalid}, faltantes {p.missing}, duplicados ignorados "
        f"{ev.duplicates}.",
        f"- Con spread del modelo > 0: {p.with_model_spread} (mínimo "
        f"{MIN_SPREAD_POSITIVO}).",
        "",
        "## P1: spread",
        "",
        f"Mediana de `ratio_spread` sobre los diarios con spread del modelo > 0: "
        f"**{_f(ev.p1_median, '.3g')}** (umbral 10^-2,5 = {UMBRAL_P1:.3g}). "
        + {True: "Se cumple.", False: "No se cumple.", None: "No se evalúa."}[ev.p1],
        "",
        "## P2: costo total sin comisión, por tamaño y lado",
        "",
        "`media(real_rel) < media(model_rel) / 2`, sobre todos los diarios válidos.",
        "",
        "| Tamaño | Lado | Media real (bps) | Media modelo (bps) | Mitad del modelo "
        "| Agotados | P2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in ev.daily:
        mitad = None if c.model_bps is None else c.model_bps / 2
        lineas.append(
            f"| {c.size:g} | {c.side} | {_f(c.real_bps, '.4f')} | "
            f"{_f(c.model_bps, '.2f')} | {_f(mitad, '.2f')} | {c.exhausted} | "
            f"{'sí' if p2_holds(c) else 'no'} |"
        )
    if ev.p2_failed:
        lineas += ["", f"Celdas donde P2 falla: {', '.join(ev.p2_failed)}."]
    lineas += [
        "",
        "## Costos por celda: snapshots de las 00:00:30 (deciden)",
        "",
        *_tabla_celdas(ev.daily),
        "",
        "## Costos por celda: snapshots de 15 minutos (no deciden)",
        "",
        *_tabla_celdas(ev.periodic),
        "",
        "## Diarios inválidos",
        "",
    ]
    if ev.invalid_reasons:
        for dia, motivos in ev.invalid_reasons.items():
            lineas.append(f"- {dia}: {'; '.join(motivos) or 'sin motivo'}")
    else:
        lineas.append("Ninguno.")
    lineas += ["", "## Limitaciones", "", LIMITACIONES]
    return "\n".join(lineas)


def summary(ev: Evaluation, adr_sha256: str) -> dict[str, Any]:
    p = ev.progress
    return {
        "adr": ADR_PATH.name,
        "adr_sha256": adr_sha256,
        "verdict": ev.verdict,
        "window": {"first_valid": iso(p.window.first_valid), "days": p.window.days()},
        "daily": {
            "valid": p.valid,
            "invalid": p.invalid,
            "missing": p.missing,
            "duplicates": ev.duplicates,
            "with_model_spread": p.with_model_spread,
        },
        "p1": {
            "median_ratio_spread": ev.p1_median,
            "threshold": UMBRAL_P1,
            "holds": ev.p1,
        },
        "p2_failed": ev.p2_failed,
        "cells_daily": [asdict(c) for c in ev.daily],
        "cells_periodic": [asdict(c) for c in ev.periodic],
        "invalid_reasons": ev.invalid_reasons,
    }


def _escribir(ruta: Path, texto: str) -> str:
    datos = texto.encode("utf-8")
    ruta.write_bytes(datos)
    return hashlib.sha256(datos).hexdigest()


def main(argv: Sequence[str], clock: Clock | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="copia de /var/lib/shadow")
    parser.add_argument("--out", type=Path, default=Path("results/sombra"))
    parser.add_argument("--adr", type=Path, default=ADR_PATH)
    args = parser.parse_args(list(argv))
    ahora = np.datetime64((clock or SystemClock()).now(), "ns")
    snaps = read_snapshots(args.data_dir)
    avance = progress(snaps, ahora)
    if ahora < avance.window.closes_at:
        print(render_progress(avance))
        print("Ventana abierta: la prediccion no se evalua todavia.")
        return 0
    adr_sha = hashlib.sha256(args.adr.read_bytes()).hexdigest()
    ev = evaluate(snaps, ahora, adr_sha)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"inputs": snaps.files, "broken_lines": snaps.broken_lines}
    manifest["REPORTE.md"] = _escribir(
        args.out / "REPORTE.md", render_report(ev, adr_sha, len(snaps.headers))
    )
    manifest["summary.json"] = _escribir(
        args.out / "summary.json",
        json.dumps(summary(ev, adr_sha), indent=2, ensure_ascii=False) + "\n",
    )
    _escribir(args.out / "MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(render_progress(avance))
    print(f"Veredicto: {ev.verdict}  (reporte en {args.out / 'REPORTE.md'})")
    return 0
