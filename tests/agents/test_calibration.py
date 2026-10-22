"""Calibracion del presupuesto (ADR 0010): sellado, reanudacion y regla de parada.

Nada de esto entrena: los criterios se aplican sobre registros fabricados a mano,
como en ``test_protocol``. Lo unico que corre una politica es ``train_unit`` con
un peso constante, para fijar la forma del registro.
"""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agents import calibration as cal
from agents.cli import PATH_SEEDS, main
from agents.policy import ConstantWeightPolicy
from agents.protocol import STUDY_SEEDS, ProtocolThresholds, Verdict, evaluate_level_4b
from tests.agents.test_protocol import brazo, corrida

RAIZ = Path(__file__).resolve().parents[2]


def registro(
    unit: cal.Unit, *, invertido: float, dispersion: float, drift_t: float = 5.0
) -> dict[str, Any]:
    arm = brazo("level_4b", time_invested=invertido, action_std=dispersion).describe()
    run = corrida(
        unit.agent_seed, time_invested=invertido, action_std=dispersion
    ).describe()
    arm["runs"] = [run]
    return {
        "unit": unit.key,
        "point": unit.point,
        "timesteps": unit.timesteps,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        "drift_t": drift_t,
        "wall_seconds": 60.0,
        "machine": {"platform": "test"},
        "arm": arm,
    }


def poblar(
    out: Path, point: str, *, invertido: float = 0.9, dispersion: float = 0.05
) -> None:
    for unit in cal.point_units(point):
        cal.write_atomic(
            cal.unit_path(out, unit),
            registro(unit, invertido=invertido, dispersion=dispersion),
        )


# ---------------------------------------------------------------------------
# Lo que el ADR fija
# ---------------------------------------------------------------------------


def test_la_grilla_duplica_en_cada_paso_y_es_la_del_adr() -> None:
    pasos = [t for _, t in cal.GRID]
    assert pasos == [120_000, 240_000, 480_000, 960_000]
    assert [b / a for a, b in pairwise(pasos)] == [2.0, 2.0, 2.0]


def test_los_caminos_de_calibracion_no_son_los_del_protocolo() -> None:
    """Elegir y juzgar sobre la misma muestra: lo que el ADR 0010 §3 evita."""
    assert cal.CALIBRATION_PATH_SEEDS == (809, 811, 821, 823, 827)
    assert set(cal.CALIBRATION_PATH_SEEDS).isdisjoint(PATH_SEEDS)


def test_cada_punto_son_cincuenta_entrenamientos_distintos() -> None:
    unidades = cal.point_units("G2")
    assert len(unidades) == 50
    assert len({u.key for u in unidades}) == 50
    assert {u.agent_seed for u in unidades} == set(STUDY_SEEDS)
    assert {u.timesteps for u in unidades} == {240_000}


def test_punto_desconocido_se_rechaza() -> None:
    with pytest.raises(cal.CalibrationError, match="punto desconocido 'G9'"):
        cal.point_units("G9")


def test_el_protocolo_sigue_exigiendo_diez_caminos() -> None:
    """El override de 5 caminos es local a la calibracion, no un cambio de default."""
    assert ProtocolThresholds().level_4_min_paths == 10
    local = cal.thresholds().describe()
    base = ProtocolThresholds().describe()
    assert local["level_4_min_paths"] == 5
    assert {k: v for k, v in local.items() if k != "level_4_min_paths"} == {
        k: v for k, v in base.items() if k != "level_4_min_paths"
    }


def test_el_adr_en_disco_es_el_sellado() -> None:
    assert cal.verify_adr(RAIZ) == cal.ADR_SHA256


def test_el_adr_con_finales_de_linea_de_windows_sigue_siendo_el_sellado(
    tmp_path: Path,
) -> None:
    """Medido en el PC de Kevin: git con autocrlf entrega \\r\\n y el SHA256 cambia."""
    destino = tmp_path / cal.ADR_PATH
    destino.parent.mkdir(parents=True)
    destino.write_bytes((RAIZ / cal.ADR_PATH).read_bytes().replace(b"\n", b"\r\n"))
    assert cal.verify_adr(tmp_path) == cal.ADR_SHA256


def test_un_adr_modificado_no_corre(tmp_path: Path) -> None:
    destino = tmp_path / cal.ADR_PATH
    destino.parent.mkdir(parents=True)
    destino.write_text("otro texto")
    with pytest.raises(cal.CalibrationError, match="el criterio cambio"):
        cal.verify_adr(tmp_path)


# ---------------------------------------------------------------------------
# Sello y reparto
# ---------------------------------------------------------------------------


def test_el_sello_se_crea_y_reanudar_con_lo_mismo_es_valido(tmp_path: Path) -> None:
    cal.acquire_seal(tmp_path, cal.ADR_SHA256)
    cal.acquire_seal(tmp_path, cal.ADR_SHA256)
    sello = json.loads((tmp_path / "SEAL.json").read_text())
    assert sello == {
        "adr_sha256": cal.ADR_SHA256,
        "config_digest": cal.config_digest(),
    }


def test_reanudar_con_otro_adr_falla(tmp_path: Path) -> None:
    cal.acquire_seal(tmp_path, cal.ADR_SHA256)
    with pytest.raises(cal.CalibrationError, match="es de otra configuracion"):
        cal.acquire_seal(tmp_path, "0" * 64)


def test_el_reparto_cubre_todo_sin_solaparse() -> None:
    unidades = cal.point_units("G1")
    partes = [cal.shard(unidades, k, 4) for k in range(4)]
    assert [len(p) for p in partes] == [13, 13, 12, 12]
    claves = [u.key for p in partes for u in p]
    assert sorted(claves) == sorted(u.key for u in unidades)


def test_un_shard_fuera_de_rango_falla() -> None:
    with pytest.raises(cal.CalibrationError, match="shard 4/4"):
        cal.shard(cal.point_units("G1"), 4, 4)


def test_horas_proyectadas_contra_el_adr() -> None:
    """0,78 ms por paso, G2, 50 entrenamientos, 4 procesos: ~40 min (ADR §6).

    A mano: 0.00078 s * 240000 = 187.2 s; * 50 = 9360 s; / 4 = 2340 s = 0.65 h.
    """
    assert cal.projected_hours(0.00078, 240_000, 50, 4) == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Reanudacion
# ---------------------------------------------------------------------------


def test_un_corte_pierde_solo_el_entrenamiento_en_curso(tmp_path: Path) -> None:
    unidades = cal.point_units("G1")[:5]
    llamadas: list[str] = []

    def entrenar(unit: cal.Unit) -> dict[str, Any]:
        if len(llamadas) == 2:
            raise KeyboardInterrupt
        llamadas.append(unit.key)
        return {"wall_seconds": 1.0, "unit": unit.key}

    with pytest.raises(KeyboardInterrupt):
        cal.run_units(tmp_path, unidades, train=entrenar, log=lambda _: None)
    assert sorted(p.name for p in (tmp_path / "units").iterdir()) == sorted(
        f"{k}.json" for k in llamadas
    )

    reanudadas: list[str] = []

    def reanudar(unit: cal.Unit) -> dict[str, Any]:
        reanudadas.append(unit.key)
        return {"wall_seconds": 1.0, "unit": unit.key}

    hechos = cal.run_units(tmp_path, unidades, train=reanudar, log=lambda _: None)
    assert hechos == 3
    assert reanudadas == [u.key for u in unidades[2:]]


def test_run_point_no_lanza_nada_si_todo_esta_hecho(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poblar(tmp_path, "G1")

    def no_debe_lanzar(*args: object, **kwargs: object) -> None:
        raise AssertionError("no habia nada pendiente")

    monkeypatch.setattr(cal.subprocess, "Popen", no_debe_lanzar)
    cal.run_point(tmp_path, "G1", 4)


class _Proceso:
    lanzados: ClassVar[list[tuple[list[str], dict[str, str]]]] = []
    codigo = 0

    def __init__(self, args: list[str], env: dict[str, str]) -> None:
        self.args = args
        _Proceso.lanzados.append((args, env))

    def wait(self) -> int:
        return _Proceso.codigo


def test_run_point_lanza_un_proceso_por_shard_de_un_hilo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Proceso.lanzados, _Proceso.codigo = [], 0
    monkeypatch.setattr(cal.subprocess, "Popen", _Proceso)
    cal.run_point(tmp_path, "G2", 2)
    shards = [a[a.index("--shard") + 1] for a, _ in _Proceso.lanzados]
    assert shards == ["0/2", "1/2"]
    assert all(e["OMP_NUM_THREADS"] == "1" for _, e in _Proceso.lanzados)
    assert all("G2" in a for a, _ in _Proceso.lanzados)


def test_run_point_falla_si_un_hijo_falla(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Proceso.lanzados, _Proceso.codigo = [], 1
    monkeypatch.setattr(cal.subprocess, "Popen", _Proceso)
    with pytest.raises(cal.CalibrationError, match="fallaron 2 de 2 procesos en G1"):
        cal.run_point(tmp_path, "G1", 2)


# ---------------------------------------------------------------------------
# Veredicto y regla de parada
# ---------------------------------------------------------------------------


def test_con_cinco_caminos_el_protocolo_por_defecto_no_puede_juzgar(
    tmp_path: Path,
) -> None:
    """Por eso existe el override local: sin el, 4b falla por falta de caminos."""
    poblar(tmp_path, "G1")
    nivel, resultado = cal.judge_point(cal.load_point(tmp_path, "G1"))
    por_defecto = evaluate_level_4b(resultado, ProtocolThresholds())
    assert por_defecto.verdict is Verdict.FAIL
    assert "solo hay 5 caminos" in por_defecto.finding
    assert nivel.verdict is Verdict.PASS


def test_no_converger_a_invertido_es_fail(tmp_path: Path) -> None:
    poblar(tmp_path, "G1", invertido=0.5)
    nivel, _ = cal.judge_point(cal.load_point(tmp_path, "G1"))
    assert nivel.verdict is Verdict.FAIL
    assert "NO converge a estar invertido" in nivel.finding


def test_rotar_es_fail(tmp_path: Path) -> None:
    poblar(tmp_path, "G1", dispersion=0.4)
    nivel, _ = cal.judge_point(cal.load_point(tmp_path, "G1"))
    assert nivel.verdict is Verdict.FAIL
    assert "Rota" in nivel.finding


def test_faltan_entrenamientos_se_dice_cuantos(tmp_path: Path) -> None:
    unidades = cal.point_units("G1")
    cal.write_atomic(
        cal.unit_path(tmp_path, unidades[0]),
        registro(unidades[0], invertido=0.9, dispersion=0.05),
    )
    with pytest.raises(cal.CalibrationError, match="G1: faltan 49 entrenamientos"):
        cal.load_point(tmp_path, "G1")


def test_se_para_en_el_primer_punto_que_pasa(tmp_path: Path) -> None:
    poblar(tmp_path, "G1", invertido=0.5)
    poblar(tmp_path, "G2", invertido=0.9)
    entrenados: list[str] = []
    resultado = cal.decide(tmp_path, run=entrenados.append, log=lambda _: None)
    assert resultado["selected_timesteps"] == 240_000
    assert entrenados == ["G1", "G2"]
    assert [p["verdict"] for p in resultado["points"]] == ["FAIL", "PASS"]


def test_si_ningun_punto_pasa_no_se_fija_presupuesto(tmp_path: Path) -> None:
    for punto, _ in cal.GRID:
        poblar(tmp_path, punto, invertido=0.5)
    resultado = cal.decide(tmp_path, log=lambda _: None)
    assert resultado["selected_timesteps"] is None
    assert len(resultado["points"]) == 4
    assert "NO se fija presupuesto" in cal.render(resultado)


def test_el_resumen_lleva_sigma_pared_y_maquina(tmp_path: Path) -> None:
    poblar(tmp_path, "G1")
    resumen = cal.summarize_point("G1", cal.load_point(tmp_path, "G1"))
    assert resumen["wall_seconds"]["median"] == 60.0
    assert resumen["machine"] == {"platform": "test"}
    assert resumen["pooled_over_paths_and_seeds"]["policy_sigma"]["n_missing"] == 50
    assert "time_invested" in resumen["decompositions"]


def test_el_manifest_lleva_el_sha256_de_cada_entrenamiento(tmp_path: Path) -> None:
    poblar(tmp_path, "G1")
    resultado = cal.decide(tmp_path, log=lambda _: None)
    cal.finalize(tmp_path, resultado)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    unidad = cal.point_units("G1")[0]
    esperado = hashlib.sha256(cal.unit_path(tmp_path, unidad).read_bytes()).hexdigest()
    assert manifest["units"][f"{unidad.key}.json"] == esperado
    assert len(manifest["units"]) == 50
    assert (tmp_path / "summary.json").exists()


# ---------------------------------------------------------------------------
# CLI y registro de un entrenamiento
# ---------------------------------------------------------------------------


def test_cli_report_juzga_lo_guardado_y_sella(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(RAIZ)
    poblar(tmp_path, "G1")
    assert main(["calibrate", "report", "--out", str(tmp_path)]) == 0
    salida = capsys.readouterr().out
    assert "T* = 120000 timesteps" in salida
    assert (tmp_path / "SEAL.json").exists()


def test_cli_bench_proyecta_las_horas(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cal, "train_unit", lambda unit, timesteps: {"wall_seconds": 100.0}
    )
    assert main(["calibrate", "bench", "--workers", "4"]) == 0
    # 100 s / 20000 pasos = 0.005 s/paso; G2: 0.005*240000*50/4/3600 = 4.17 h.
    assert "G2 (240000 pasos): 4.2 h con 4 procesos" in capsys.readouterr().out


def test_cli_worker_entrena_solo_su_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cal, "train_unit", lambda unit: {"wall_seconds": 1.0, "unit": unit.key}
    )
    args = ["calibrate", "worker", "--point", "G1", "--shard", "1/4"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    assert len(list((tmp_path / "units").glob("*.json"))) == 13


def test_un_entrenamiento_deja_su_registro() -> None:
    unidad = cal.point_units("G1")[0]
    registro_unidad = cal.train_unit(
        unidad, policy_factory=lambda train, seed: ConstantWeightPolicy(1.0)
    )
    assert registro_unidad["unit"] == "G1_p809_s11"
    assert registro_unidad["agent_seed"] == 11
    assert registro_unidad["wall_seconds"] > 0
    assert len(registro_unidad["arm"]["runs"]) == 1
    assert registro_unidad["arm"]["runs"][0]["policy_sigma"] is None
    assert registro_unidad["arm"]["runs"][0]["time_invested"] == pytest.approx(1.0)
    assert json.loads(json.dumps(registro_unidad)) == registro_unidad
