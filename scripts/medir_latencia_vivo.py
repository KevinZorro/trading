"""p99 del retraso de los lotes de GDELT en vivo, para el sello del estudio.

Uso (en la VM, donde estan los datos del grabador; leer con ``sudo``)::

    PYTHONPATH=src sudo -E uv run python scripts/medir_latencia_vivo.py /var/lib/news

Imprime el p99 en minutos: es el valor de ``--live-latency-p99-min`` de
``python -m agents.cli study run``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from live.news_latency import p99_minutes

if __name__ == "__main__":
    p99, n = p99_minutes(Path(sys.argv[1]))
    print(f"lotes medidos: {n}", file=sys.stderr)
    print(f"{p99:.1f}")
