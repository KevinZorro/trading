"""Evalua la prediccion pre-registrada sobre el spread (ADR 0007, seccion 6).

Uso, contra una copia de ``/var/lib/shadow`` (no necesita torch ni red)::

    uv run python scripts/evaluar_sombra.py <carpeta> [--out results/sombra]

Con la ventana de 30 dias abierta solo imprime el avance; no calcula P1 ni P2.
"""

from __future__ import annotations

import sys

from eval.shadow_report import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
