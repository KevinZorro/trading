"""Confirmacion de C* (ADR 0011 §5): unidades, reanudacion y armado de los brazos.

Nada entrena: los registros se fabrican a mano. Lo unico que corre episodios es
``train_unit`` con un peso constante, para fijar que C* llega al entorno.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from agents import calibration as cal
from agents import cli
from agents import confirmation as conf
from agents import diagnosis as dg
from agents import regularization as reg
from agents.cli import FIXTURE_SEED, PATH_SEEDS, main
from agents.policy import ConstantWeightPolicy
from agents.protocol import STUDY_SEEDS, ArmResult, MultiPathResult
from tests.agents.test_calibration import registro
from tests.agents.test_diagnosis import registro as registro_dg

RAIZ = Path(__file__).resolve().parents[2]


def poblar(out: Path, point: str) -> None:
    unidades = (
        conf.rediagnosis_units()
        if point == conf.REDIAGNOSIS_POINT
        else conf.arm_units(point)
    )
    cerrado = dict.fromkeys(
        ("retornos", "tecnicos", "posicion", "resultado_orden"), 0.1
    )
    for u in unidades:
        datos = (
            registro_dg(u.key, cerrado)
            if point == conf.REDIAGNOSIS_POINT
            else registro(u, invertido=0.95, dispersion=0.05)
        )
        cal.write_atomic(cal.unit_path(out, u), datos)


def poblar_todo(out: Path) -> None:
    for a in conf.ARMS:
        poblar(out, a.point)


# ---------------------------------------------------------------------------
# Lo que el ADR fija
# ---------------------------------------------------------------------------


def test_son_los_480_entrenamientos_del_adr_0010() -> None:
    unidades = conf.all_units()
    assert len(unidades) == 480
    assert len({u.key for u in unidades}) == 480
    assert sum(not a.multipath for a in conf.ARMS) == 8
    assert sum(a.multipath for a in conf.ARMS) == 4


def test_las_etiquetas_son_las_de_la_etapa_3() -> None:
    """``cli assemble`` busca los brazos por etiqueta; tienen que ser las mismas."""
    versionados = {
        p.stem.split("_", 1)[1] for p in (RAIZ / "results").glob("multipath_*.json")
    } | {p.stem.split("_", 1)[1] for p in (RAIZ / "results").glob("arm_*.json")}
    # "level_4" sale de arm_level_4 y de multipath_level_4; solo se re-corre el
    # multicamino, que es el que juzga 4a.
    assert {a.label for a in conf.ARMS} == versionados
    assert {a.label for a in conf.ARMS if a.multipath} == {
        "level_3_mlp",
        "level_3_lstm",
        "level_4",
        "level_4b",
    }


def test_los_multicamino_usan_los_caminos_registrados() -> None:
    caminos = {u.path_seed for u in conf.arm_units("C-level_4b")}
    assert caminos == set(PATH_SEEDS)
    assert {u.path_seed for u in conf.arm_units("C-level_0")} == {FIXTURE_SEED}
    assert all(u.timesteps == 120_000 for u in conf.all_units())


def test_la_configuracion_confirmada_es_r3() -> None:
    assert conf.CONFIRMED_RUNG == "R3"
    assert "0.9" in json.dumps(reg.rung("R3").describe())


def test_brazo_desconocido_se_rechaza() -> None:
    with pytest.raises(cal.CalibrationError, match="brazo de confirmacion desconocido"):
        conf.arm_units("C-level_9")


def test_no_se_mezcla_con_la_escalera(tmp_path: Path) -> None:
    cal.acquire_seal(tmp_path, reg.ADR_SHA256, reg.config_digest())
    with pytest.raises(cal.CalibrationError, match="es de otra configuracion"):
        cal.acquire_seal(tmp_path, reg.ADR_SHA256, conf.config_digest())


# ---------------------------------------------------------------------------
# Armado de los brazos
# ---------------------------------------------------------------------------


def test_sin_terminar_se_dice_cuantos_faltan(tmp_path: Path) -> None:
    poblar(tmp_path, "C-level_0")
    with pytest.raises(cal.CalibrationError, match="faltan 470 de 480"):
        conf.collect(tmp_path)


def test_los_brazos_armados_se_leen_como_los_de_la_etapa_3(tmp_path: Path) -> None:
    poblar_todo(tmp_path)
    destino = conf.collect(tmp_path)
    brazo = ArmResult.from_dict(
        json.loads((destino / "arm_level_2_beta-0.3.json").read_text())
    )
    assert brazo.label == "level_2_beta-0.3"
    assert [r.seed for r in brazo.runs] == list(STUDY_SEEDS)
    multi = MultiPathResult.from_dict(
        json.loads((destino / "multipath_level_4b.json").read_text())
    )
    assert multi.path_seeds == PATH_SEEDS
    assert len(multi.arms) == 10
    assert all(len(a.runs) == 10 for a in multi.arms)
    assert len(list(destino.glob("*.json"))) == 12


def test_el_manifest_lleva_los_entrenamientos_hechos(tmp_path: Path) -> None:
    poblar(tmp_path, "C-level_0")
    datos = json.loads(conf.manifest(tmp_path).read_text())
    assert len(datos["units"]) == 10
    assert datos["rung"] == "R3"
    assert datos["adr_sha256"] == reg.ADR_SHA256


# ---------------------------------------------------------------------------
# CLI y registro de un entrenamiento
# ---------------------------------------------------------------------------


def test_cli_confirm_entrena_cada_brazo_y_el_rediagnostico(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(RAIZ)
    lanzados: list[str] = []

    def run_point(out: Path, point: str, workers: int, units: object) -> None:
        lanzados.append(point)
        poblar(out, point)

    juzgado: list[argparse.Namespace] = []
    monkeypatch.setattr(cal, "run_point", run_point)
    monkeypatch.setattr(cli, "cmd_assemble", lambda a: juzgado.append(a) or 0)
    assert main(["calibrate", "confirm", "--out", str(tmp_path)]) == 0
    assert lanzados == [a.point for a in conf.ARMS] + [conf.REDIAGNOSIS_POINT]
    assert juzgado[0].out == str(tmp_path / "protocolo")
    assert json.loads(juzgado[0].provenance)["rung"] == "R3"
    assert "Re-diagnostico con R3" in capsys.readouterr().out
    assert (tmp_path / "rediagnosis.json").exists()


def test_cli_confirm_report_sin_rediagnostico_juzga_igual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(RAIZ)
    poblar_todo(tmp_path)
    monkeypatch.setattr(cli, "cmd_assemble", lambda a: 1)
    assert main(["calibrate", "confirm-report", "--out", str(tmp_path)]) == 1
    assert not (tmp_path / "rediagnosis.json").exists()


@pytest.mark.parametrize(
    ("point", "funcion", "esperados"),
    [
        ("C-level_4b", "train_unit", 34),
        (conf.REDIAGNOSIS_POINT, "train_rediagnosis_unit", 4),
    ],
)
def test_cli_worker_reconoce_la_confirmacion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    funcion: str,
    esperados: int,
) -> None:
    vistos: list[str] = []

    def entrenar(unit: cal.Unit) -> dict[str, object]:
        vistos.append(unit.point)
        return {"wall_seconds": 1.0}

    monkeypatch.setattr(conf, funcion, entrenar)
    args = ["calibrate", "worker", "--point", point, "--shard", "0/3"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    assert vistos == [point] * esperados


def test_el_registro_lleva_c_estrella_y_el_lstm_donde_toca() -> None:
    unidad = conf.arm_units("C-level_3_beta-0.3_lstm")[0]
    datos = conf.train_unit(
        unidad, policy_factory=lambda train, seed: ConstantWeightPolicy(1.0)
    )
    assert datos["observation"]["lookback"] == 1
    assert datos["arm"]["ppo"]["gamma"] == 0.9
    assert datos["arm"]["ppo"]["recurrent"] is True
    assert datos["drift_t"] is None


def test_un_multicamino_guarda_el_t_del_drift() -> None:
    unidad = conf.arm_units("C-level_4b")[0]
    datos = conf.train_unit(
        unidad, policy_factory=lambda train, seed: ConstantWeightPolicy(1.0)
    )
    assert datos["drift_t"] > 3.0
    assert datos["arm"]["label"] == f"level_4b:path{PATH_SEEDS[0]}"


def test_el_rediagnostico_usa_la_observacion_de_c_estrella() -> None:
    vistos: list[object] = []

    def entrenar(factory: object, cfg: object, seed: int) -> ConstantWeightPolicy:
        vistos.append(cfg)
        return ConstantWeightPolicy(1.0)

    r = reg.rung("R3")
    datos = dg.train_unit(
        conf.rediagnosis_units()[0],
        trainer=entrenar,  # type: ignore[arg-type]
        ppo=r.ppo,
        observation=r.observation,
    )
    assert vistos == [r.ppo]
    assert datos["unit"] == "D-R3_p839_s11"
    assert set(datos["diagnosis"]["groups"]) == {
        "retornos",
        "tecnicos",
        "posicion",
        "resultado_orden",
    }
