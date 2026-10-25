"""Reporte de validacion del dataset de GDELT versionado.

Uso (desde la raiz del repo)::

    PYTHONPATH=src uv run python scripts/validar_gdelt.py \\
        > datasets/gdelt/VALIDACION.md

Solo lee el dataset versionado y su manifest, no la descarga cruda. No repara nada:
los huecos y los apagones se cuentan tal como quedaron. La cobertura usa la latencia
de 60 min del ADR 0012 (K2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.manifest import read_manifest
from data.news_aggregate import (
    BATCHES_PER_DAY,
    COVERAGE_MIN_BATCHES,
    bar_healthy_counts,
    healthy_batches,
    read_aggregates,
)

RAIZ = Path(__file__).resolve().parent.parent
MANIFEST = RAIZ / "datasets" / "gdelt" / "gdelt-batches.csv.gz.manifest.json"
LATENCIA = np.timedelta64(60, "m")
APAGON_MIN_LOTES = 12 * 4  # apagon: mas de 12 h sin un solo lote sano


def _tabla(filas: list[list[object]], cabecera: list[str]) -> None:
    print("| " + " | ".join(cabecera) + " |")
    print("|" + "---|" * len(cabecera))
    for f in filas:
        print("| " + " | ".join(str(x) for x in f) + " |")
    print()


def main() -> int:
    manifest = read_manifest(MANIFEST)
    f = read_aggregates(MANIFEST)  # verifica el SHA256
    sano = healthy_batches(f)
    f["sano"] = sano
    ok = f["status"] == "ok"
    print("# Validacion del dataset de GDELT\n")
    print("Generado por `scripts/validar_gdelt.py`. No editar a mano.\n")
    print(f"- `{manifest.file}`, SHA256 verificado: `{manifest.sha256}`")
    print(f"- rotulos: {f['label'].iloc[0]} -> {f['label'].iloc[-1]} (UTC)")
    print(f"- filas: {len(f)}; rejilla de 15 min completa y sin repetidos")
    print(f"- lotes ok: {int(ok.sum())}; missing: {int((~ok).sum())} ")
    print(f"- lotes sanos (ADR 0012, 2.5): {int(sano.sum())}\n")

    print("## Por ano\n")
    f["anio"] = f["label"].dt.year
    filas = []
    for anio, g in f.groupby("anio"):
        gok = g[g["status"] == "ok"]
        filas.append(
            [
                anio,
                len(g),
                f"{100 * (len(g) - len(gok)) / len(g):.1f} %",
                f"{100 * g['sano'].mean():.1f} %",
                int(gok["rows_total"].median()),
                int(gok["rows_theme"].sum()),
                f"{100 * (gok['n_titled'] > 0).mean():.1f} %",
            ]
        )
    _tabla(
        filas,
        [
            "ano",
            "lotes",
            "missing",
            "sanos",
            "mediana rows_total",
            "filas del tema",
            "lotes con titulo",
        ],
    )
    con_titulo = f.loc[f["n_titled"] > 0, "label"]
    print(
        f"Primer lote con titulo: {con_titulo.iloc[0]}. Antes de esa fecha "
        "`no_title` = 1 por construccion.\n"
    )

    print("## Apagones (mas de 12 h sin lote sano)\n")
    corrida = (f["sano"] != f["sano"].shift()).cumsum()
    filas = []
    for _, g in f[~f["sano"]].groupby(corrida[~f["sano"]]):
        if len(g) > APAGON_MIN_LOTES:
            filas.append(
                [g["label"].iloc[0], g["label"].iloc[-1], f"{len(g) / 4:.1f} h"]
            )
    _tabla(filas, ["desde", "hasta", "duracion"])
    print(f"Total: {len(filas)} apagones.\n")

    print("## Latencia observada (Last-Modified menos rotulo)\n")
    off = f["last_modified_offset_s"].dropna() / 60
    print(
        f"Lotes con cabecera: {len(off)} de {int(ok.sum())}. Mediana "
        f"{off.median():.1f} min, p99 {off.quantile(0.99):+.1f} min.\n"
    )
    filas = []
    for corte in (0, 30, 60, 120):
        tarde = f.loc[f["last_modified_offset_s"] > corte * 60]
        filas.append(
            [
                f"> {corte} min",
                len(tarde),
                tarde["anio"].value_counts().sort_index().to_dict(),
            ]
        )
    _tabla(filas, ["publicado despues del rotulo +", "lotes", "por ano"])

    print("## Cobertura por barra diaria (latencia 60 min, ADR 0012 2.3 y 2.5)\n")
    cuentas = bar_healthy_counts(f, sano, LATENCIA).iloc[1:-1]  # bordes incompletos
    malas = cuentas[cuentas < COVERAGE_MIN_BATCHES]
    print(
        f"Barras: {len(cuentas)}; con `coverage_ok` = 0 (menos de "
        f"{COVERAGE_MIN_BATCHES} de {BATCHES_PER_DAY} lotes sanos): {len(malas)} "
        f"({100 * len(malas) / len(cuentas):.1f} %).\n"
    )
    por_anio = malas.groupby(malas.index.year).size()
    _tabla([[a, n] for a, n in por_anio.items()], ["ano", "barras con coverage_ok = 0"])

    print("## Senal disponible por barra (lotes sanos)\n")
    dia = f.assign(
        cierre=(f["label"] + pd.Timedelta(LATENCIA) - pd.Timedelta(1, "ns")).dt.floor(
            "D"
        )
    )
    sanos = (
        dia[dia["sano"]]
        .groupby("cierre")
        .agg(
            total=("rows_total", "sum"),
            tema=("rows_theme", "sum"),
            tono=("tone_sum", "sum"),
        )
    )
    sin_filas = sanos[sanos["tema"] == 0]
    print(
        f"Barras con lotes sanos: {len(sanos)}. Sin ninguna fila del tema "
        f"(`tone_mean` enmascarado): {len(sin_filas)} "
        f"({100 * len(sin_filas) / len(sanos):.1f} %)."
    )
    con = sanos[sanos["tema"] > 0]
    share = np.log1p(1e4 * con["tema"] / con["total"])
    print(
        f"`news_share` (ADR 0012, col. 1): mediana {share.median():.3f}, "
        f"p1 {share.quantile(0.01):.3f}, p99 {share.quantile(0.99):.3f}."
    )
    tono = con["tono"] / con["tema"]
    print(
        f"`tone_mean`: mediana {tono.median():.3f}, p1 {tono.quantile(0.01):.3f}, "
        f"p99 {tono.quantile(0.99):.3f}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
