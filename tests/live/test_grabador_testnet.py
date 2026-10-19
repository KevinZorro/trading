"""El grabador de testnet en modo ``--solo-errores``, sin red.

Los dobles de transporte solo ejercitan el cableado del script (cantidades,
verificacion del ``code`` esperado, suma al manifest). **No son oraculos de Binance**:
la forma de las respuestas reales vive en ``respuestas/testnet`` y los tests del venue
no deben leer de aca.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from live.testnet_http import (
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    HttpResponse,
)

RAIZ = Path(__file__).resolve().parents[2]
GRABADAS = RAIZ / "tests" / "live" / "respuestas" / "testnet"


def _cargar() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "grabar_respuestas_testnet", RAIZ / "scripts" / "grabar_respuestas_testnet.py"
    )
    assert spec is not None
    assert spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _json(nombre: str) -> dict[str, object]:
    datos: dict[str, object] = json.loads((GRABADAS / nombre).read_text())
    return datos


def test_cantidades_de_los_rechazos_calculadas_a_mano() -> None:
    # exchange_info.json real: LOT_SIZE minQty 0.00001 / stepSize 0.00001 / maxQty 9000;
    # MARKET_LOT_SIZE maxQty 133.97193096. El tope de mercado manda y se baja al paso.
    pasos = _cargar().pasos_de_error(_json("exchange_info.json"), "BTCUSDT", "42")
    por_nombre = {n: (p["side"], p["quantity"], c) for n, _, _, p, c in pasos}
    assert por_nombre == {
        "err_filtro_lot_size": ("BUY", "0.000001", -1013),
        "err_filtro_notional": ("BUY", "0.00001", -1013),
        "err_saldo_compra": ("BUY", "133.97193", -2010),
        "err_saldo_venta": ("SELL", "133.97193", -2010),
    }
    ids = [p["newClientOrderId"] for _, _, _, p, _ in pasos]
    assert len(set(ids)) == 4
    assert all(i.endswith("-42") for i in ids)  # type: ignore[union-attr]


class Servidor:
    """Responde por ruta; ``codigos`` fuerza el ``code`` de cada rechazo."""

    def __init__(self, codigos: dict[str, int]) -> None:
        self.codigos = codigos
        self.ordenes: list[dict[str, list[str]]] = []

    def __call__(self, pedido: HttpRequest) -> HttpResponse:
        url = urllib.parse.urlsplit(pedido.url)
        if url.path == "/api/v3/time":
            return HttpResponse(200, b'{"serverTime": 1}')
        if url.path == "/api/v3/exchangeInfo":
            return HttpResponse(200, (GRABADAS / "exchange_info.json").read_bytes())
        if url.path == "/api/v3/account" and "signature=000" in url.query:
            return HttpResponse(
                401, json.dumps({"code": self.codigos["firma"]}).encode()
            )
        if url.path == "/api/v3/account":
            return HttpResponse(200, (GRABADAS / "account_antes.json").read_bytes())
        consulta = urllib.parse.parse_qs(pedido.body.decode())
        self.ordenes.append(consulta)
        lado, qty = consulta["side"][0], consulta["quantity"][0]
        clave = {"0.000001": "lot", "0.00001": "notional"}.get(qty, f"saldo_{lado}")
        return HttpResponse(400, json.dumps({"code": self.codigos[clave]}).encode())


CODIGOS_OK = {
    "lot": -1013,
    "notional": -1013,
    "saldo_BUY": -2010,
    "saldo_SELL": -2010,
    "firma": -1022,
}


@pytest.fixture
def grabador(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[dict[str, int]], tuple[ModuleType, Servidor, Path]]:
    def armar(codigos: dict[str, int]) -> tuple[ModuleType, Servidor, Path]:
        modulo = _cargar()
        destino = tmp_path / "respuestas"
        shutil.copytree(GRABADAS, destino)
        servidor = Servidor(codigos)
        monkeypatch.setattr(modulo, "DESTINO", destino)
        monkeypatch.setattr(modulo, "RAIZ", tmp_path)
        monkeypatch.setattr(modulo, "urllib_transport", servidor)
        monkeypatch.setattr(
            modulo,
            "BinanceTestnetClient",
            lambda c, r: BinanceTestnetClient(c, r, transport=servidor),
        )
        monkeypatch.setattr(
            modulo.BinanceTestnetCredentials,
            "from_env",
            lambda _env: BinanceTestnetCredentials("K" * 64, "S" * 64),
        )
        return modulo, servidor, destino

    return armar


def test_solo_errores_suma_al_manifest_sin_tocar_lo_demas(
    grabador: Callable[[dict[str, int]], tuple[ModuleType, Servidor, Path]],
) -> None:
    modulo, _, destino = grabador(CODIGOS_OK)
    antes = {p.name: p.read_bytes() for p in destino.glob("*.json")}
    previo = json.loads((destino / "manifest.json").read_text())["files"]

    assert modulo.main(["--solo-errores"]) == 0

    nuevos = {
        "err_filtro_lot_size.json",
        "err_filtro_notional.json",
        "err_saldo_compra.json",
        "err_saldo_venta.json",
    }
    despues = {p.name: p.read_bytes() for p in destino.glob("*.json")}
    assert set(despues) - set(antes) == nuevos
    intactos = set(antes) - {"manifest.json", "err_firma_invalida.json"}
    assert all(despues[n] == antes[n] for n in intactos)
    assert json.loads(despues["err_firma_invalida.json"]) == {"code": -1022}
    manifest = json.loads(despues["manifest.json"])["files"]
    assert set(manifest) == set(previo) | nuevos
    assert all(manifest[n] == previo[n] for n in intactos)
    assert manifest["err_saldo_venta.json"]["params"]["side"] == "SELL"
    assert manifest["err_firma_invalida.json"]["http_status"] == 401


@pytest.mark.parametrize("clave", ["lot", "saldo_BUY", "firma"])
def test_un_code_inesperado_aborta_sin_escribir(
    grabador: Callable[[dict[str, int]], tuple[ModuleType, Servidor, Path]], clave: str
) -> None:
    modulo, _, destino = grabador({**CODIGOS_OK, clave: -1130})
    antes = {p.name: p.read_bytes() for p in destino.iterdir()}

    with pytest.raises(SystemExit, match=r"se esperaba code -\d+ y Binance devolvio"):
        modulo.main(["--solo-errores"])

    assert {p.name: p.read_bytes() for p in destino.iterdir()} == antes


def test_solo_errores_no_envia_ordenes_exitosas(
    grabador: Callable[[dict[str, int]], tuple[ModuleType, Servidor, Path]],
) -> None:
    modulo, servidor, _ = grabador(CODIGOS_OK)
    modulo.main(["--solo-errores"])
    # Solo las cuatro ordenes que deben ser rechazadas; ni compra ni venta de 0.001.
    assert [o["quantity"][0] for o in servidor.ordenes] == [
        "0.000001",
        "0.00001",
        "133.97193",
        "133.97193",
    ]
