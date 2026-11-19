"""Veredicto del ADR 0007 §6 sobre snapshots armados a mano, con oraculos a mano."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.shadow import (
    UMBRAL_P1,
    CellStats,
    ShadowEvalError,
    cell_stats,
    daily_by_day,
    evaluate,
    p2_holds,
    progress,
    read_snapshots,
    spread_ratio,
    window_of,
)
from eval.shadow_report import main, render_progress

SHA = "a" * 64
INICIO = np.datetime64("2026-10-20", "D")
CERRADA = np.datetime64("2026-11-18T00:10:30", "ns")  # 30.o dia + 00:00:30 + gracia
TAMANOS = (10.0, 100.0, 1000.0)


def dia(i: int) -> str:
    return str(INICIO + np.timedelta64(i, "D"))


def snap(
    fecha: str,
    *,
    kind: str = "daily",
    valid: bool = True,
    real_hs: float = 1e-7,
    model_hs: float = 3e-3,
    real_rel: dict[tuple[float, str], float] | None = None,
    model_rel: float | None = None,
    exhausted: tuple[float, str] | None = None,
) -> dict[str, Any]:
    hora = "00:00:30" if kind == "daily" else "12:15:00"
    celdas = []
    if valid:
        for tamano in TAMANOS:
            for lado in ("buy", "sell"):
                real = (real_rel or {}).get((tamano, lado), real_hs)
                modelo = model_hs if model_rel is None else model_rel
                celdas.append(
                    {
                        "size": tamano,
                        "side": lado,
                        "real": {
                            "half_spread_rel": real_hs,
                            "depth_impact_rel": real - real_hs,
                            "real_rel": real,
                            "exhausted": exhausted == (tamano, lado),
                        },
                        "model": {
                            "half_spread_rel": model_hs,
                            "slippage_rel": modelo - model_hs,
                            "model_rel": modelo,
                        },
                        "ratios": {
                            "ratio_total": real / modelo if modelo > 0 else None,
                            "ratio_spread": real_hs / model_hs if model_hs else None,
                        },
                    }
                )
    return {
        "type": "snapshot",
        "kind": kind,
        "scheduled": f"{fecha}T{hora}.000Z",
        "valid": valid,
        "reasons": [] if valid else ["depth: timeout"],
        "cells": celdas,
    }


def escribir(directorio: Path, registros: list[dict[str, Any]], sha: str = SHA) -> None:
    cabecera = json.dumps({"type": "header", "adr_sha256": sha})
    for r in registros:
        ruta = directorio / f"snapshots-{r['scheduled'][:10]}.jsonl"
        previo = ruta.read_text() if ruta.exists() else cabecera + "\n"
        ruta.write_text(previo + json.dumps(r) + "\n")


def treinta(**kw: Any) -> list[dict[str, Any]]:
    return [snap(dia(i), **kw) for i in range(30)]


def cargar(tmp_path: Path, registros: list[dict[str, Any]]) -> Any:
    escribir(tmp_path, registros)
    return read_snapshots(tmp_path)


def test_ventana_arranca_en_el_primer_diario_valido(tmp_path: Path) -> None:
    previo = snap("2026-10-19", valid=False)  # dia del despliegue
    ventana = window_of([previo, *treinta()])
    assert ventana.days()[0] == "2026-10-20"
    assert ventana.days()[-1] == "2026-11-18"
    assert len(ventana.days()) == 30
    assert ventana.closes_at == CERRADA


def test_sin_diario_valido_no_hay_ventana() -> None:
    with pytest.raises(ShadowEvalError, match="ningun snapshot diario valido"):
        window_of([snap(dia(0), valid=False)])


def test_avance_cuenta_validos_invalidos_y_faltantes(tmp_path: Path) -> None:
    # Dias 0..10: el 3 y el 7 invalidos, el 5 falta. A las 01:00 del dia 10 ya
    # transcurrieron 11 dias (el 10 vence a las 00:10:30).
    registros = [snap(dia(i), valid=i not in (3, 7)) for i in range(11) if i != 5]
    p = progress(cargar(tmp_path, registros), np.datetime64(f"{dia(10)}T01:00"))
    assert (p.elapsed, p.valid, p.invalid, p.missing) == (11, 8, 2, 1)
    assert p.with_model_spread == 8
    assert (p.remaining, p.max_reachable) == (19, 27)
    assert not p.already_not_evaluable
    assert "se pueden perder 2" in render_progress(p)


def test_avance_detecta_que_ya_no_se_llega_a_25(tmp_path: Path) -> None:
    registros = [snap(dia(i)) for i in range(10) if i not in (1, 2, 3, 4, 5, 6)]
    p = progress(cargar(tmp_path, registros), np.datetime64(f"{dia(9)}T00:10:30"))
    assert (p.valid, p.missing, p.max_reachable) == (4, 6, 24)
    assert p.already_not_evaluable
    assert "YA ES NO EVALUABLE" in render_progress(p)


def test_no_se_evalua_con_la_ventana_abierta(tmp_path: Path) -> None:
    snaps = cargar(tmp_path, treinta())
    with pytest.raises(ShadowEvalError, match="la ventana cierra"):
        evaluate(snaps, CERRADA - np.timedelta64(1, "s"), SHA)


def test_cabecera_con_otro_sha_del_adr_no_se_evalua(tmp_path: Path) -> None:
    snaps = cargar(tmp_path, treinta())
    with pytest.raises(ShadowEvalError, match="SHA256 del ADR 0007"):
        evaluate(snaps, CERRADA, "b" * 64)


def test_sostenida_con_libro_a_un_tick(tmp_path: Path) -> None:
    # ratio_spread = 1e-7 / 3e-3 = 3,3e-5 <= 3,2e-3; real 0,001 bps << 15 bps.
    ev = evaluate(cargar(tmp_path, treinta()), CERRADA, SHA)
    assert ev.verdict == "SOSTENIDA"
    assert ev.p1_median == pytest.approx(1e-7 / 3e-3)
    assert ev.p1 is True
    assert ev.p2_failed == []
    assert ev.progress.valid == 30


def test_parcial_nombra_la_celda_que_falla(tmp_path: Path) -> None:
    # 1000 USDT compra: real 20 bps contra modelo 30 bps; 20 >= 30 / 2.
    caro = {(1000.0, "buy"): 2e-3}
    ev = evaluate(cargar(tmp_path, treinta(real_rel=caro)), CERRADA, SHA)
    assert ev.verdict == "PARCIAL"
    assert ev.p2_failed == ["1000/buy"]


def test_refutada_si_la_mediana_supera_el_umbral(tmp_path: Path) -> None:
    # 3e-5 / 3e-3 = 1e-2 > 3,2e-3.
    ev = evaluate(cargar(tmp_path, treinta(real_hs=3e-5)), CERRADA, SHA)
    assert ev.verdict == "REFUTADA"
    assert ev.p1 is False


def test_p1_en_el_umbral_exacto_se_cumple(tmp_path: Path) -> None:
    ev = evaluate(
        cargar(tmp_path, treinta(real_hs=UMBRAL_P1, model_hs=1.0)), CERRADA, SHA
    )
    assert ev.p1_median == UMBRAL_P1
    assert ev.p1 is True


def test_24_validos_no_evaluable_25_si(tmp_path: Path) -> None:
    registros = [snap(dia(i), valid=i not in range(1, 7)) for i in range(30)]
    ev = evaluate(cargar(tmp_path, registros), CERRADA, SHA)
    assert ev.progress.valid == 24
    assert ev.verdict == "NO EVALUABLE"
    assert ev.p1_median is None
    assert ev.p1 is None

    otro = tmp_path / "b"
    otro.mkdir()
    registros = [snap(dia(i), valid=i not in range(1, 6)) for i in range(30)]
    assert evaluate(cargar(otro, registros), CERRADA, SHA).verdict == "SOSTENIDA"


def test_menos_de_10_con_spread_del_modelo_es_no_evaluable(tmp_path: Path) -> None:
    # 21 dias con Corwin-Schultz truncado a 0: quedan 9 con cociente.
    registros = [
        snap(
            dia(i), model_hs=0.0 if i < 21 else 3e-3, model_rel=1e-6 if i < 21 else None
        )
        for i in range(30)
    ]
    ev = evaluate(cargar(tmp_path, registros), CERRADA, SHA)
    assert ev.progress.with_model_spread == 9
    assert ev.verdict == "NO EVALUABLE"


def test_p2_es_estricta() -> None:
    base = dict.fromkeys(
        (
            "real_half_spread_bps",
            "real_depth_impact_bps",
            "real_bps_median",
            "model_half_spread_bps",
            "model_slippage_bps",
            "model_bps_median",
        ),
        None,
    )
    celda = CellStats(
        size=10.0,
        side="buy",
        n=30,
        exhausted=0,
        ratio_none=0,
        ratio_total=None,
        real_bps=1.0,
        model_bps=2.0,
        **base,
    )
    assert not p2_holds(celda)  # 1 no es < 2 / 2
    assert p2_holds(CellStats(**{**celda.__dict__, "real_bps": 0.99}))
    assert not p2_holds(
        CellStats(**{**celda.__dict__, "real_bps": 0.1, "exhausted": 1})
    )


def test_medias_de_p2_incluyen_los_dias_con_modelo_cero() -> None:
    # Modelo 0 y 4 bps alternados: media 2 bps. Real 0,9 bps < 1 bps -> P2 vale.
    # Con solo los dias de modelo > 0 la media del modelo seria 4 bps.
    registros = [
        snap(
            dia(i),
            real_hs=0.0,
            model_hs=0.0,
            model_rel=4e-4 * (i % 2),
            real_rel=dict.fromkeys(
                ((t, s) for t in TAMANOS for s in ("buy", "sell")), 9e-5
            ),
        )
        for i in range(4)
    ]
    celda = cell_stats(registros)[0]
    assert celda.model_bps == pytest.approx(2.0)
    assert celda.real_bps == pytest.approx(0.9)
    assert celda.ratio_none == 2  # los dos dias con modelo 0
    assert p2_holds(celda)


def test_resumen_de_ratio_total_a_mano() -> None:
    # real 1, 2, 3, 4, 5 (x1e-5) sobre modelo 1e-4: cocientes 0,1 .. 0,5.
    registros = [snap(dia(i), real_hs=1e-5 * (i + 1), model_hs=1e-4) for i in range(5)]
    r = cell_stats(registros)[0].ratio_total
    assert r == pytest.approx(
        {"median": 0.3, "p25": 0.2, "p75": 0.4, "min": 0.1, "max": 0.5}
    )


def test_libro_agotado_fuera_de_las_medias_y_p2_falla(tmp_path: Path) -> None:
    registros = treinta()
    registros[4] = snap(dia(4), exhausted=(1000.0, "sell"))
    ev = evaluate(cargar(tmp_path, registros), CERRADA, SHA)
    celda = next(c for c in ev.daily if c.key == "1000/sell")
    assert (celda.n, celda.exhausted) == (29, 1)
    assert ev.verdict == "PARCIAL"
    assert ev.p2_failed == ["1000/sell"]


def test_dia_duplicado_cuenta_el_primero() -> None:
    primero = snap(dia(0), valid=False)
    segundo = snap(dia(0))
    ventana = window_of([primero, segundo, snap(dia(1))])
    elegido, duplicados = daily_by_day([primero, segundo, snap(dia(1))], ventana)
    assert elegido[dia(0)] is primero
    assert duplicados == 1


def test_medio_spread_distinto_entre_celdas_es_error() -> None:
    r = snap(dia(0))
    r["cells"][3]["real"]["half_spread_rel"] = 2e-7
    with pytest.raises(ShadowEvalError, match="difiere entre celdas"):
        spread_ratio(r)


def test_valido_sin_modelo_es_error() -> None:
    r = snap(dia(0))
    r["cells"][0]["model"] = None
    with pytest.raises(ShadowEvalError, match="valido sin modelo"):
        spread_ratio(r)
    with pytest.raises(ShadowEvalError, match="valido sin modelo"):
        cell_stats([r])


def test_faltan_celdas_del_adr(tmp_path: Path) -> None:
    registros = treinta()
    for r in registros:
        r["cells"] = [c for c in r["cells"] if c["size"] != 1000.0]
    with pytest.raises(ShadowEvalError, match="3 tamanos x 2 lados"):
        evaluate(cargar(tmp_path, registros), CERRADA, SHA)


def test_lectura_salta_linea_rota_y_separa_cabeceras(tmp_path: Path) -> None:
    escribir(tmp_path, [snap(dia(0))])
    with open(tmp_path / f"snapshots-{dia(0)}.jsonl", "a") as f:
        f.write('{"type": "snap')
    snaps = read_snapshots(tmp_path)
    assert (len(snaps.headers), len(snaps.records), snaps.broken_lines) == (1, 1, 1)
    with pytest.raises(ShadowEvalError, match="no hay snapshots"):
        read_snapshots(tmp_path / "vacio")


def test_periodicos_solo_dentro_de_la_ventana(tmp_path: Path) -> None:
    fuera = snap("2026-11-19", kind="periodic")
    dentro = snap(dia(3), kind="periodic", real_hs=2e-7)
    ev = evaluate(cargar(tmp_path, [*treinta(), fuera, dentro]), CERRADA, SHA)
    assert [c.n for c in ev.periodic] == [1] * 6
    assert ev.periodic[0].real_half_spread_bps == pytest.approx(2e-3)


class Reloj:
    def __init__(self, ahora: np.datetime64) -> None:
        self._ahora = ahora

    def now(self) -> np.datetime64:
        return self._ahora


def test_main_con_ventana_abierta_no_escribe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    datos, salida = tmp_path / "datos", tmp_path / "out"
    datos.mkdir()
    escribir(datos, treinta()[:10])
    assert (
        main(
            [str(datos), "--out", str(salida)], Reloj(np.datetime64(f"{dia(9)}T01:00"))
        )
        == 0
    )
    assert "Ventana abierta" in capsys.readouterr().out
    assert not salida.exists()


def test_main_al_cerrar_escribe_reporte_resumen_y_manifest(tmp_path: Path) -> None:
    datos, salida, adr = tmp_path / "datos", tmp_path / "out", tmp_path / "adr.md"
    datos.mkdir()
    adr.write_bytes(b"pre-registro")
    sha = hashlib.sha256(b"pre-registro").hexdigest()
    escribir(datos, [*treinta()[:29], snap(dia(29), valid=False)], sha=sha)
    args = [str(datos), "--out", str(salida), "--adr", str(adr)]
    assert main(args, Reloj(CERRADA)) == 0

    resumen = json.loads((salida / "summary.json").read_text())
    assert resumen["verdict"] == "SOSTENIDA"
    assert resumen["daily"]["invalid"] == 1
    reporte = (salida / "REPORTE.md").read_text()
    assert "**Veredicto: SOSTENIDA**" in reporte
    assert f"- {dia(29)}: depth: timeout" in reporte
    manifest = json.loads((salida / "MANIFEST.json").read_text())
    for nombre in ("REPORTE.md", "summary.json"):
        cuerpo = (salida / nombre).read_bytes()
        assert manifest[nombre] == hashlib.sha256(cuerpo).hexdigest()
    assert len(manifest["inputs"]) == 30
