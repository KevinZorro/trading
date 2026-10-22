"""Descarga el GKG 2.0 de GDELT al formato ``news-*.jsonl`` de la Etapa 4.

Reanudable: volver a correr el mismo comando salta los rotulos ya escritos.
Solo guarda los articulos del tema; el archivo crudo se verifica contra el MD5 del
masterfilelist y se descarta. Ver ``docs/descarga-gdelt.md``.

PowerShell:  $env:PYTHONPATH = "src"; python scripts/descargar_gdelt.py `
                 --out datasets/gdelt/raw --start 2016-08-01 --end 2016-09-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from data.gdelt import THEME_BITCOIN
from data.gdelt_history import (
    MASTERFILELIST_URL,
    HistoryStore,
    download,
    http_fetch,
    parse_master,
)


def _lineas_remotas(url: str) -> Iterator[str]:
    """El masterfilelist pesa ~128 MB: se lee en streaming, nunca entero."""
    with urllib.request.urlopen(url, timeout=120) as r:
        for crudo in r:
            yield crudo.decode("utf-8", errors="replace")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", required=True, help="AAAA-MM-DD, UTC, inclusivo")
    p.add_argument("--end", required=True, help="AAAA-MM-DD, UTC, exclusivo")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--theme", default=THEME_BITCOIN)
    p.add_argument("--masterfilelist", default=MASTERFILELIST_URL)
    a = p.parse_args(argv)
    start, end = np.datetime64(a.start, "ns"), np.datetime64(a.end, "ns")
    if not start < end:
        p.error("--start debe ser anterior a --end")

    t0 = time.monotonic()
    entradas = parse_master(_lineas_remotas(a.masterfilelist), start, end)
    t_lista = time.monotonic() - t0
    store = HistoryStore(
        a.out,
        {"theme": a.theme, "masterfilelist": a.masterfilelist, "workers": a.workers},
    )
    resumen = download(
        entradas,
        start,
        end,
        store,
        http_fetch,
        workers=a.workers,
        retries=a.retries,
        sleep=time.sleep,
        theme=a.theme,
    )
    total_s = time.monotonic() - t0
    mb = sum(e.size for e in entradas) / 1e6
    print(
        json.dumps(
            {
                "range": [a.start, a.end],
                "listed": len(entradas),
                "written_ok": resumen.written_ok,
                "written_missing": resumen.written_missing,
                "skipped_done": resumen.skipped_done,
                "failed": list(resumen.failed),
                "listing_s": round(t_lista, 1),
                "total_s": round(total_s, 1),
                "listed_mb": round(mb, 1),
            },
            indent=2,
        )
    )
    return 1 if resumen.failed else 0  # failed => volver a correr para reintentar


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
