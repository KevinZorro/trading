"""Diagnostico de la rotacion de 4b: oraculos a mano con politicas de juguete.

Una politica que depende de una sola columna de la observacion tiene una
respuesta conocida: fijar esa columna deja la accion constante (dispersion 0) y
fijar cualquier otra no la cambia. Nada de esto entrena.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import diagnosis as dg
from agents.calibration import CALIBRATION_PATH_SEEDS, write_atomic
from agents.cli import PATH_SEEDS, main
from agents.policy import Policy
from agents.protocol import _sim_config
from agents.runner import build_env, scaler_for
from data.fixtures import level_4b_detectable_drift
from data.schema import FloatArray
from envs.observation import ObservationBuilder, ObservationSpec
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig, TradingEnv
from sim.sizing import TargetWeightSizer

NOMBRES = ObservationBuilder(ObservationSpec()).names
GRUPOS = dg.feature_groups(NOMBRES)


class _PorColumna:
    """Accion = 0.5 + 0.4 * tanh(columna): depende solo de esa columna."""

    def __init__(self, columna: str) -> None:
        self._i = NOMBRES.index(columna)

    @property
    def name(self) -> str:
        return "por_columna"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: FloatArray) -> float:
        return float(0.5 + 0.4 * np.tanh(observation[self._i]))

    def describe(self) -> dict[str, object]:
        return {"policy": "por_columna"}


@pytest.fixture(scope="module")
def entorno() -> tuple[Any, Any]:
    fixture = level_4b_detectable_drift(1500, seed=839)
    train, validacion, _ = fixture.split()
    spec = ObservationSpec()
    escalador = scaler_for(train.series, spec)
    config_sim = _sim_config(fixture, 100_000.0, 0.0)
    env_config = EnvConfig(observation=spec, sizer=TargetWeightSizer(safety=0.98))

    def construir() -> TradingEnv:
        return build_env(
            validacion.series,
            config_sim,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=100.0),
        )

    return construir, validacion


def test_los_caminos_del_diagnostico_son_nuevos() -> None:
    assert set(dg.DIAGNOSIS_PATH_SEEDS).isdisjoint(PATH_SEEDS)
    assert set(dg.DIAGNOSIS_PATH_SEEDS).isdisjoint(CALIBRATION_PATH_SEEDS)
    assert len(dg.diagnosis_units()) == 10


def test_los_grupos_cubren_toda_la_observacion_sin_repetir() -> None:
    indices = sorted(i for idx in GRUPOS.values() for i in idx)
    assert indices == list(range(len(NOMBRES)))
    assert {NOMBRES[i] for i in GRUPOS["posicion"]} == {
        "position_weight",
        "cash_weight",
    }


def test_fijar_la_columna_de_la_que_depende_deja_la_accion_constante(
    entorno: tuple[Any, Any],
) -> None:
    construir, _ = entorno
    res = dg.diagnose_policy(_PorColumna("position_weight"), construir, GRUPOS)
    posicion = res["groups"]["posicion"]
    # Sin la posicion visible la accion es 0.5 + 0.4*tanh(media): una constante.
    assert posicion["open_loop_action_std"] == pytest.approx(0.0, abs=1e-12)
    assert posicion["closed_loop_action_std"] == pytest.approx(0.0, abs=1e-12)
    # Fijar un grupo del que no depende no cambia ninguna accion.
    retornos = res["groups"]["retornos"]
    assert retornos["open_loop_mean_abs_change"] == pytest.approx(0.0, abs=1e-12)
    assert retornos["open_loop_action_std"] == pytest.approx(res["action_std"])


def test_una_politica_de_retornos_se_explica_por_los_retornos(
    entorno: tuple[Any, Any],
) -> None:
    construir, _ = entorno
    res = dg.diagnose_policy(_PorColumna("log_return_0"), construir, GRUPOS)
    assert res["action_std"] > 0.0
    assert res["groups"]["retornos"]["open_loop_action_std"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert res["groups"]["posicion"]["open_loop_mean_abs_change"] == pytest.approx(
        0.0, abs=1e-12
    )


def registro(
    unit: str, cerrado: dict[str, float], base: float = 0.35
) -> dict[str, Any]:
    return {
        "unit": unit,
        "diagnosis": {
            "action_std": base,
            "groups": {
                g: {
                    "open_loop_action_std": c,
                    "open_loop_mean_abs_change": 0.1,
                    "closed_loop_action_std": c,
                }
                for g, c in cerrado.items()
            },
        },
    }


def test_lectura_nombra_el_grupo_cuyo_fijado_baja_la_rotacion_al_umbral() -> None:
    cerrado = {
        "retornos": 0.34,
        "tecnicos": 0.33,
        "posicion": 0.02,
        "resultado_orden": 0.35,
    }
    res = dg.summarize_diagnosis([registro(f"u{i}", cerrado) for i in range(10)])
    assert res["sufficient_groups"] == ["posicion"]
    assert "posicion" in res["reading"]
    assert res["threshold"] == 0.15


def test_lectura_cuando_ningun_grupo_basta() -> None:
    cerrado = {
        "retornos": 0.3,
        "tecnicos": 0.3,
        "posicion": 0.3,
        "resultado_orden": 0.3,
    }
    res = dg.summarize_diagnosis([registro(f"u{i}", cerrado) for i in range(10)])
    assert res["sufficient_groups"] == []
    assert "ningun grupo" in res["reading"]


def test_un_entrenamiento_deja_su_registro_con_el_diagnostico() -> None:
    unidad = dg.diagnosis_units()[0]
    reg = dg.train_unit(unidad, trainer=lambda factory, cfg, seed: _PorColumna("rsi"))
    assert reg["unit"] == "D_p839_s11"
    assert reg["timesteps"] == 120_000
    assert set(reg["diagnosis"]["groups"]) == set(GRUPOS)
    assert reg["policy_sigma"] is None


def test_cli_diagnose_resume_lo_ya_entrenado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cerrado = {
        "retornos": 0.3,
        "tecnicos": 0.3,
        "posicion": 0.02,
        "resultado_orden": 0.3,
    }
    for u in dg.diagnosis_units():
        write_atomic(tmp_path / "units" / f"{u.key}.json", registro(u.key, cerrado))
    assert main(["calibrate", "diagnose", "--out", str(tmp_path)]) == 0
    assert '"sufficient_groups"' in capsys.readouterr().out
    assert (tmp_path / "summary.json").exists()


def test_las_politicas_de_juguete_cumplen_el_protocol() -> None:
    assert isinstance(_PorColumna("rsi"), Policy)
