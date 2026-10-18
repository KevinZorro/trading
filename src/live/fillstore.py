"""Persistencia del venue sombra: acuses, fills y cancelaciones (ADR 0007, seccion 2).

Mismo contrato que ``SnapshotStore``: append con ``fsync`` en cada registro, un
archivo por dia UTC, cabecera en la primera linea de cada archivo. Aca el log es
ademas la **fuente de verdad** del venue: en sombra no hay broker externo que
reporte la posicion, asi que ``reconcile`` reconstruye el estado repitiendo este
log. Por eso se guarda tambien el acuse de cada orden y no solo el fill: sin el,
un reinicio entre el envio y el fill perderia la orden, y reenviar el mismo
``client_order_id`` dejaria de ser idempotente.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from live.store import StoreError, _fsync_dir, _linea
from sim.orders import Fill, OrderStatus, RejectReason

log = logging.getLogger(__name__)


def day_of(ts: np.datetime64) -> str:
    return str(np.datetime_as_string(ts, unit="D"))


def fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "t_decision": fill.t_decision,
        "t_fill": fill.t_fill,
        "timestamp_decision": str(np.datetime64(fill.timestamp_decision, "ns")),
        "timestamp_fill": str(np.datetime64(fill.timestamp_fill, "ns")),
        "symbol": fill.symbol,
        "status": fill.status.value,
        "qty_requested": fill.qty_requested,
        "qty_filled": fill.qty_filled,
        "decision_price": fill.decision_price,
        "ref_price": fill.ref_price,
        "fill_price": fill.fill_price,
        "gap": fill.gap,
        "spread_cost": fill.spread_cost,
        "slippage_cost": fill.slippage_cost,
        "commission": fill.commission,
        "participation": fill.participation,
        "reject_reason": fill.reject_reason.value if fill.reject_reason else None,
        "tag": fill.tag,
        "client_order_id": fill.client_order_id,
    }


def fill_from_dict(d: Mapping[str, Any]) -> Fill:
    motivo = d["reject_reason"]
    return Fill(
        t_decision=int(d["t_decision"]),
        t_fill=int(d["t_fill"]),
        timestamp_decision=np.datetime64(d["timestamp_decision"], "ns"),
        timestamp_fill=np.datetime64(d["timestamp_fill"], "ns"),
        symbol=d["symbol"],
        status=OrderStatus(d["status"]),
        qty_requested=float(d["qty_requested"]),
        qty_filled=float(d["qty_filled"]),
        decision_price=float(d["decision_price"]),
        ref_price=float(d["ref_price"]),
        fill_price=float(d["fill_price"]),
        gap=float(d["gap"]),
        spread_cost=float(d["spread_cost"]),
        slippage_cost=float(d["slippage_cost"]),
        commission=float(d["commission"]),
        participation=float(d["participation"]),
        reject_reason=RejectReason(motivo) if motivo else None,
        tag=d["tag"],
        client_order_id=d["client_order_id"],
    )


def venue_header(
    *,
    initial_cash: float,
    spec_sha256: str,
    spec_received_at: np.datetime64,
    git_sha: str = "unknown",
) -> dict[str, Any]:
    """Cabecera de cada archivo. ADR 0008, punto 3: el spec viaja con la corrida."""
    return {
        "type": "header",
        "initial_cash": initial_cash,
        "instrument_spec_sha256": spec_sha256,
        "instrument_spec_received_at": str(np.datetime64(spec_received_at, "ns")),
        "git_sha": git_sha,
    }


class FillStore:
    """Archivos ``venue-AAAA-MM-DD.jsonl``."""

    def __init__(self, data_dir: Path, header: Mapping[str, Any]) -> None:
        self._dir = data_dir
        self._header = dict(header)
        data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def header(self) -> Mapping[str, Any]:
        return self._header

    def append(self, day: str, registro: Mapping[str, Any]) -> None:
        ruta = self._dir / f"venue-{day}.jsonl"
        nuevo = not ruta.exists()
        separar = False
        if not nuevo and ruta.stat().st_size > 0:
            with open(ruta, "rb") as lectura:
                lectura.seek(-1, os.SEEK_END)
                separar = lectura.read(1) != b"\n"  # linea rota por una caida
        with open(ruta, "ab") as f:
            if separar:
                f.write(b"\n")
            if nuevo:
                f.write(_linea(self._header))
            f.write(_linea(registro))
            f.flush()
            os.fsync(f.fileno())
        if nuevo:
            _fsync_dir(self._dir)

    def replay(self) -> list[dict[str, Any]]:
        """Registros en orden cronologico, sin cabeceras.

        Una linea que no parsea es la que una caida dejo a medias: su envio
        nunca llego a confirmarse, asi que se descarta con aviso. Una cabecera
        con otro capital inicial es un error: la contabilidad no se mezcla.
        """
        salida: list[dict[str, Any]] = []
        for ruta in sorted(self._dir.glob("venue-*.jsonl")):
            for n, linea in enumerate(ruta.read_text().splitlines(), start=1):
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    log.warning("linea rota descartada: %s:%d", ruta.name, n)
                    continue
                if registro.get("type") == "header":
                    if registro["initial_cash"] != self._header["initial_cash"]:
                        raise StoreError(
                            f"{ruta.name}: initial_cash={registro['initial_cash']} "
                            f"!= {self._header['initial_cash']} de esta corrida"
                        )
                    continue
                salida.append(registro)
        return salida
