"""Graba respuestas reales de testnet de Binance para testear el venue (PR 7b).

Uso (desde la raiz del repo, con red). Las claves son de **testnet**
(https://testnet.binance.vision) y se pasan por el entorno, nunca por argumento::

    export TESTNET_API_KEY=...  TESTNET_API_SECRET=...
    PYTHONPATH=src uv run python scripts/grabar_respuestas_testnet.py

Compra y vende una cantidad minima en testnet (dinero ficticio) y deja en
``tests/live/respuestas/testnet/`` el cuerpo de cada respuesta, incluidos los
errores, mas un ``manifest.json`` con metodo, path, parametros, estado HTTP,
hora de recepcion y SHA256.

Con ``--solo-errores`` no opera: graba unicamente los rechazos que faltaban
(-1013 filtro LOT_SIZE y NOTIONAL, -2010 saldo insuficiente al comprar y al
vender, -1022 firma invalida) y los **suma** al manifest existente sin tocar el
resto. Cada uno declara el ``code`` que debe devolver Binance; si devuelve otro, no
se escribe nada: una grabacion que no muestra el error buscado no sirve de oraculo.

Lo que **nunca** se escribe a disco: la API key, la firma, el query firmado ni
las cabeceras ``X-MBX-APIKEY``. El manifest guarda los parametros sin ``signature``,
``timestamp`` ni ``recvWindow``; el campo ``uid`` de la cuenta se borra; y antes de
escribir se verifica que ni la clave ni el secreto aparezcan y que nada tenga forma
de clave o firma (``live.testnet_http.find_secret_shapes``). Si algo falla, no se
escribe nada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import numpy as np

from live.testnet_http import (
    TESTNET_BASE_URL,
    BinanceApiError,
    BinanceHttpError,
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    find_secret_shapes,
    urllib_transport,
)
from sim.clock import SystemClock

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "tests" / "live" / "respuestas" / "testnet"
CAMPOS_A_BORRAR = frozenset({"uid"})


def _limpiar(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items() if k not in CAMPOS_A_BORRAR}
    if isinstance(valor, list):
        return [_limpiar(v) for v in valor]
    return valor


def _sin_ceros(valor: Decimal) -> str:
    return format(valor.normalize(), "f")


def pasos_de_error(
    info: dict[str, Any], sym: str, corrida: str
) -> list[tuple[str, str, str, dict[str, str | int], int]]:
    """Rechazos a provocar, con el ``code`` que Binance debe devolver.

    Las cantidades salen de ``exchangeInfo`` y no de numeros fijos, para que el
    rechazo no dependa del precio del dia: bajo el minimo de LOT_SIZE (-1013), el
    minimo de LOT_SIZE (cuyo notional queda bajo ``minNotional``: -1013) y el mayor
    tamano de mercado permitido, que ninguna cuenta de testnet puede pagar ni
    entregar (-2010).
    """
    filtros = {f["filterType"]: f for f in info["symbols"][0]["filters"]}
    lote = filtros["LOT_SIZE"]
    min_qty = Decimal(lote["minQty"])
    paso = Decimal(lote["stepSize"])
    tope = min(Decimal(lote["maxQty"]), Decimal(filtros["MARKET_LOT_SIZE"]["maxQty"]))
    grande = (tope // paso * paso).quantize(paso, rounding=ROUND_DOWN)

    def orden(lado: str, qty: Decimal, tag: str) -> dict[str, str | int]:
        return {
            "symbol": sym,
            "side": lado,
            "type": "MARKET",
            "quantity": _sin_ceros(qty),
            "newClientOrderId": f"rec7e-{tag}-{corrida}",
            "newOrderRespType": "FULL",
        }

    path = "/api/v3/order"
    return [
        ("err_filtro_lot_size", "POST", path, orden("BUY", min_qty / 10, "lot"), -1013),
        ("err_filtro_notional", "POST", path, orden("BUY", min_qty, "not"), -1013),
        ("err_saldo_compra", "POST", path, orden("BUY", grande, "cmp"), -2010),
        ("err_saldo_venta", "POST", path, orden("SELL", grande, "vta"), -2010),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--qty", default="0.001", help="cantidad de la compra/venta")
    parser.add_argument(
        "--solo-errores",
        action="store_true",
        help="no opera: suma al manifest los rechazos -1013, -2010 y -1022",
    )
    args = parser.parse_args(argv)

    creds = BinanceTestnetCredentials.from_env(os.environ)
    reloj = SystemClock()
    cliente = BinanceTestnetClient(creds, reloj)
    cliente.sync_time()
    # Antes de grabar nada: con claves invalidas todas las respuestas serian 401 y
    # la grabacion no serviria para testear el venue.
    sonda = cliente.send("GET", "/api/v3/account")
    if sonda.status != 200:
        raise SystemExit(
            f"las claves no sirven (HTTP {sonda.status}: {sonda.body[:120]!r}). "
            "Revisa TESTNET_API_KEY y TESTNET_API_SECRET: deben ser de "
            "https://testnet.binance.vision, sin comillas ni espacios. "
            "No se grabo nada."
        )
    corrida = str(int(reloj.now().astype("datetime64[s]").astype(np.int64)))
    sym, qty = args.symbol, args.qty
    ids = {k: f"rec7a-{k}-{corrida}" for k in ("buy", "sell")}

    def orden(cid: str, lado: str) -> dict[str, str | int]:
        return {
            "symbol": sym,
            "side": lado,
            "type": "MARKET",
            "quantity": qty,
            "newClientOrderId": cid,
            "newOrderRespType": "FULL",
        }

    consulta = {"symbol": sym, "origClientOrderId": ids["buy"]}
    pasos: list[tuple[str, str, str, dict[str, str | int]]] = (
        []
        if args.solo_errores
        else [
            ("time", "GET", "/api/v3/time", {}),
            ("exchange_info", "GET", "/api/v3/exchangeInfo", {"symbol": sym}),
            ("account_antes", "GET", "/api/v3/account", {}),
            ("order_buy", "POST", "/api/v3/order", orden(ids["buy"], "BUY")),
            ("order_query", "GET", "/api/v3/order", consulta),
            # Reenvio del mismo id ya lleno: define como 7b logra la idempotencia.
            ("order_buy_duplicado", "POST", "/api/v3/order", orden(ids["buy"], "BUY")),
            ("open_orders", "GET", "/api/v3/openOrders", {"symbol": sym}),
            ("my_trades", "GET", "/api/v3/myTrades", {"symbol": sym, "limit": 10}),
            ("order_sell", "POST", "/api/v3/order", orden(ids["sell"], "SELL")),
            ("account_despues", "GET", "/api/v3/account", {}),
            (
                "err_consulta_inexistente",
                "GET",
                "/api/v3/order",
                {"symbol": sym, "origClientOrderId": f"no-existe-{corrida}"},
            ),
            (
                "err_cancelar_inexistente",
                "DELETE",
                "/api/v3/order",
                {"symbol": sym, "origClientOrderId": f"no-existe-{corrida}"},
            ),
            (
                "err_id_invalido",
                "POST",
                "/api/v3/order",
                orden("id con espacios!", "BUY"),
            ),
        ]
    )
    grabado: dict[str, tuple[bytes, dict[str, Any]]] = {}

    def registrar(
        nombre: str,
        metodo: str,
        path: str,
        params: dict[str, str | int],
        esperado: int | None = None,
    ) -> None:
        respuesta = cliente.send(metodo, path, params)
        recibido = reloj.now()
        try:
            datos = json.loads(respuesta.body)
        except ValueError as exc:
            raise SystemExit(f"{nombre}: respuesta que no es JSON") from exc
        if esperado is not None and datos.get("code") != esperado:
            raise SystemExit(
                f"{nombre}: se esperaba code {esperado} y Binance devolvio "
                f"{datos.get('code')} ({datos.get('msg')!r}). No se grabo nada."
            )
        cuerpo = json.dumps(_limpiar(datos)).encode()
        grabado[nombre] = (
            cuerpo,
            {
                "method": metodo,
                "path": path,
                "params": params,
                "http_status": respuesta.status,
                "received_at_utc": np.datetime_as_string(recibido, unit="ms") + "Z",
                "sha256": hashlib.sha256(cuerpo).hexdigest(),
            },
        )

    for nombre, metodo, path, params in pasos:
        registrar(nombre, metodo, path, params)
    info = cliente.call("GET", "/api/v3/exchangeInfo", {"symbol": sym})
    for nombre, metodo, path, params, esperado in pasos_de_error(info, sym, corrida):
        registrar(nombre, metodo, path, params, esperado)

    # Firma invalida a proposito, armada a mano: el cliente no deja firmar mal.
    # Con timestamp valido: si no, Binance rechaza el timestamp (-1130) antes de
    # mirar la firma y la grabacion no muestra el error de firma.
    ahora = int(reloj.now().astype("datetime64[ms]").astype(np.int64))
    mala = urllib.parse.urlencode(
        {"timestamp": ahora, "recvWindow": 5000, "signature": "0" * 64}
    )
    url = f"{TESTNET_BASE_URL}/api/v3/account?{mala}"
    respuesta = urllib_transport(
        HttpRequest("GET", url, {"X-MBX-APIKEY": creds.api_key})
    )
    datos = json.loads(respuesta.body)
    if datos.get("code") != -1022:
        raise SystemExit(
            f"err_firma_invalida: se esperaba code -1022 y Binance devolvio "
            f"{datos.get('code')} ({datos.get('msg')!r}). No se grabo nada."
        )
    cuerpo = json.dumps(datos).encode()
    grabado["err_firma_invalida"] = (
        cuerpo,
        {
            "method": "GET",
            "path": "/api/v3/account",
            "params": {},
            "http_status": respuesta.status,
            "received_at_utc": np.datetime_as_string(reloj.now(), unit="ms") + "Z",
            "sha256": hashlib.sha256(cuerpo).hexdigest(),
        },
    )

    archivos: dict[str, Any] = {}
    if args.solo_errores:
        # Suma, no reemplaza: las grabaciones de ordenes exitosas se conservan tal cual.
        previo = json.loads((DESTINO / "manifest.json").read_text(encoding="utf-8"))
        archivos = previo["files"]
    archivos.update({f"{n}.json": meta for n, (_, meta) in grabado.items()})
    manifest = {
        "script": "scripts/grabar_respuestas_testnet.py",
        "base_url": TESTNET_BASE_URL,
        "symbol": sym,
        "files": archivos,
    }
    texto = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    conjunto = texto + b"".join(c for c, _ in grabado.values()).decode()
    prohibidos = [creds.api_key, creds.api_secret]
    if any(p in conjunto for p in prohibidos) or find_secret_shapes(conjunto):
        raise SystemExit("hay algo con forma de clave o firma: no se escribe nada")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, (cuerpo, _) in grabado.items():
        (DESTINO / f"{nombre}.json").write_bytes(cuerpo)
    (DESTINO / "manifest.json").write_text(texto, encoding="utf-8")
    print(f"{len(grabado)} respuestas en {DESTINO.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BinanceApiError, BinanceHttpError) as exc:
        sys.exit(f"fallo la grabacion: {exc}")
