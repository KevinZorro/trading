"""Puerta del Agente B (ADR 0012, 5): unidades, sello y ensamblado.

Sin torch: las politicas son constantes y el universo se achica por monkeypatch
(el real son 400 entrenamientos).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents import calibration as cal
from agents import confirmation, gate
from agents.cli import PATH_SEEDS
from agents.policy import ConstantWeightPolicy
from agents.ppo import SB3Policy
from agents.protocol import STUDY_SEEDS
from features.news import NEWS_FEATURE_NAMES


def test_son_400_entrenamientos_y_ninguno_repite_clave() -> None:
    unidades = gate.all_units()
    assert len(unidades) == 400
    assert len({u.key for u in unidades}) == 400
    assert {u.timesteps for u in unidades} == {120_000}


def test_los_caminos_de_la_puerta_son_disjuntos_de_los_demas() -> None:
    from agents.calibration import CALIBRATION_PATH_SEEDS
    from agents.diagnosis import DIAGNOSIS_PATH_SEEDS

    usados = {*PATH_SEEDS, *CALIBRATION_PATH_SEEDS, *DIAGNOSIS_PATH_SEEDS}
    assert not usados & set(gate.GATE_PATH_SEEDS)
    assert len(set(gate.GATE_PATH_SEEDS)) == 10


def test_b_ve_las_siete_columnas_y_a_ninguna() -> None:
    b, a = gate.arm("N6-5a_B"), gate.arm("N6-5a_A")
    assert tuple(c.name for c in gate.observation_for(b).exog) == NEWS_FEATURE_NAMES
    assert gate.observation_for(a).exog == ()
    assert a.blind
    assert not b.blind


def test_un_brazo_desconocido_falla_con_su_nombre() -> None:
    with pytest.raises(cal.CalibrationError, match="N6-9z"):
        gate.arm_units("N6-9z")


def test_el_digest_cambia_si_cambia_un_camino(monkeypatch: pytest.MonkeyPatch) -> None:
    antes = gate.config_digest()
    monkeypatch.setattr(gate, "GATE_PATH_SEEDS", gate.GATE_PATH_SEEDS[:-1])
    assert gate.config_digest() != antes


def test_collect_falla_diciendo_cuantos_faltan(tmp_path: Path) -> None:
    with pytest.raises(cal.CalibrationError, match="faltan 400 de 400"):
        gate.collect(tmp_path)


def test_ensamblado_con_un_universo_chico(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "GATE_PATH_SEEDS", gate.GATE_PATH_SEEDS[:2])
    monkeypatch.setattr(gate, "STUDY_SEEDS", STUDY_SEEDS[:2])
    monkeypatch.setattr(gate, "DEFAULT_N_BARS", 500)
    monkeypatch.setattr(confirmation, "STUDY_SEEDS", STUDY_SEEDS[:2])

    def entrenar(unit: cal.Unit) -> dict[str, object]:
        return gate.train_unit(
            unit, policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0)
        )

    for a in gate.ARMS:
        cal.run_units(
            tmp_path, gate.arm_units(a.point), train=entrenar, log=lambda _m: None
        )
    assert len(list((tmp_path / "units").glob("*.json"))) == 16
    # Reanudar no reentrena nada.
    hechos = cal.run_units(
        tmp_path, gate.arm_units("N6-5a_B"), train=entrenar, log=lambda _m: None
    )
    assert hechos == 0

    nivel_5a, nivel_5b = gate.assess(tmp_path)
    # Con 2 caminos el criterio de 10 no se cumple: la puerta bloquea, y lo dice.
    assert nivel_5a.verdict.value == "FAIL"
    assert "solo hay 2 caminos" in nivel_5a.finding
    resumen = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert resumen["blocks_real_data"] is True
    assert len(json.loads((tmp_path / "MANIFEST.json").read_text())["units"]) == 16
    assert (tmp_path / "gate" / "multipath_5a_B.json").exists()


# -- politicas guardadas para el re-diagnostico ----------------------------


class _ModeloFalso:
    """Lo minimo de un modelo SB3: ``save`` escribe un archivo y ``predict`` es fijo."""

    def predict(self, observation: object, **_k: object) -> tuple[list[float], None]:
        return [1.0], None

    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_bytes(b"pesos")


class _PoliticaFalsa(SB3Policy):
    """Una SB3Policy que no necesita torch: siempre invertida, sigma fijo."""

    def act(self, observation: object) -> float:  # type: ignore[override]
        return 1.0

    def sigma(self) -> float:
        return 0.1


def _entrena_con(policy_dir: Path, unit: cal.Unit) -> dict[str, object]:
    return gate.train_unit(
        unit,
        policy_factory=lambda _f, _s: _PoliticaFalsa(_ModeloFalso(), recurrent=False),
        policy_dir=policy_dir,
    )


@pytest.fixture
def chico(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "DEFAULT_N_BARS", 500)


def test_b_guarda_la_politica_de_las_dos_primeras_semillas(
    tmp_path: Path, chico: None
) -> None:
    unidad = cal.Unit("N6-5a_B", 1, gate.GATE_PATH_SEEDS[0], STUDY_SEEDS[0])
    registro = _entrena_con(tmp_path, unidad)
    assert registro["policy_saved"] is True
    assert (tmp_path / f"{unidad.key}.zip").read_bytes() == b"pesos"


def test_no_guarda_la_de_a_ni_la_de_las_demas_semillas(
    tmp_path: Path, chico: None
) -> None:
    ciega = cal.Unit("N6-5a_A", 1, gate.GATE_PATH_SEEDS[0], STUDY_SEEDS[0])
    tercera = cal.Unit("N6-5b_B", 1, gate.GATE_PATH_SEEDS[0], STUDY_SEEDS[2])
    for unidad in (ciega, tercera):
        assert _entrena_con(tmp_path, unidad)["policy_saved"] is False
    assert not list(tmp_path.glob("*.zip"))


def test_sin_policy_dir_no_guarda_nada(chico: None) -> None:
    unidad = cal.Unit("N6-5b_B", 1, gate.GATE_PATH_SEEDS[0], STUDY_SEEDS[0])
    registro = gate.train_unit(
        unidad, policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0)
    )
    assert registro["policy_saved"] is False


def test_run_arm_avisa_de_cada_politica_antes_de_evaluarla() -> None:
    from agents.ppo import PPOConfig
    from agents.protocol import run_arm
    from data.fixtures import level_1_noisy

    vistas: list[int] = []
    run_arm(
        level_1_noisy(400, seed=1),
        label="x",
        seeds=[3, 4],
        ppo_config=PPOConfig(total_timesteps=1),
        policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0),
        allow_fewer_seeds=True,
        on_trained=lambda _p, semilla: vistas.append(semilla),
    )
    assert vistas == [3, 4]
