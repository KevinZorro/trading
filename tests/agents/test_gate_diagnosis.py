"""Re-diagnostico de la puerta con el grupo noticias (ADR 0012, 5).

Sin torch: el cargador devuelve modelos falsos cuya accion es una funcion
conocida de la observacion, asi que el efecto de fijar un grupo se calcula a mano.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import calibration as cal
from agents import gate, gate_diagnosis
from agents.protocol import STUDY_SEEDS
from envs.observation import ObservationBuilder


def _col(nombre: str) -> int:
    return ObservationBuilder(gate.observation_for(gate.arm("N6-5a_B"))).names.index(
        nombre
    )


class _Sigue:
    """Accion = 0.5 + 0.4 * tone_mean: depende de las noticias."""

    def predict(self, obs: Any, **_k: object) -> tuple[Any, None]:
        return np.clip(0.5 + 0.4 * obs[_col("tone_mean")], 0.0, 1.0), None


class _Ignora:
    """Siempre invertido: las noticias no cambian nada."""

    def predict(self, obs: Any, **_k: object) -> tuple[Any, None]:
        return np.array([1.0]), None


@pytest.fixture
def chico(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "DEFAULT_N_BARS", 500)
    monkeypatch.setattr(gate_diagnosis, "DEFAULT_N_BARS", 500)
    monkeypatch.setattr(gate, "GATE_PATH_SEEDS", gate.GATE_PATH_SEEDS[:5])


def _guarda_politicas(out: Path) -> None:
    for a in gate_diagnosis.B_ARMS:
        for u in gate_diagnosis.saved_units(out, a.point):
            ruta = Path(f"{gate.policy_path(out / 'policies', u)}.zip")
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(b"x")


def test_son_diez_politicas_por_nivel_con_el_universo_chico(
    tmp_path: Path, chico: None
) -> None:
    unidades = gate_diagnosis.saved_units(tmp_path, "N6-5a_B")
    assert len(unidades) == 10
    assert {u.agent_seed for u in unidades} == set(STUDY_SEEDS[:2])


def test_con_politicas_que_siguen_las_noticias_el_reporte_dice_que_las_usa(
    tmp_path: Path, chico: None
) -> None:
    _guarda_politicas(tmp_path)
    resumen = gate_diagnosis.run(tmp_path, loader=lambda _p: _Sigue())
    assert resumen["5a_B"]["uses_news"] is True
    assert resumen["5a_B"]["news_median_abs_change"] > 0.02
    assert "B usa las noticias" in resumen["5a_B"]["reading"]
    assert (tmp_path / "rediagnosis.json").is_file()


def test_con_politicas_que_las_ignoran_el_reporte_dice_que_no(
    tmp_path: Path, chico: None
) -> None:
    _guarda_politicas(tmp_path)
    resumen = gate_diagnosis.run(tmp_path, loader=lambda _p: _Ignora())
    for nivel in ("5a_B", "5b_B"):
        assert resumen[nivel]["news_median_abs_change"] == pytest.approx(0.0)
        assert "B no usa las noticias" in resumen[nivel]["reading"]


def test_en_el_5a_se_informa_el_capture_con_las_noticias_fijas(
    tmp_path: Path, chico: None
) -> None:
    _guarda_politicas(tmp_path)
    r = gate_diagnosis.run(tmp_path, loader=lambda _p: _Sigue())["5a_B"]
    assert r["capture"]["policies_that_lose_capture"] >= 0
    assert "politicas lo pierden al fijar las noticias" in r["reading"]


def test_falta_una_politica_guardada(tmp_path: Path, chico: None) -> None:
    with pytest.raises(cal.CalibrationError, match="falta la politica guardada"):
        gate_diagnosis.run(tmp_path, loader=lambda _p: _Ignora())


# -- regla de lectura con registros a mano ---------------------------------


def _registro(cambio: float, captura: float | None, fijo: float | None) -> dict:
    return {
        "diagnosis": {
            "action_std": 0.1,
            "capture": captura,
            "groups": {
                "noticias": {
                    "open_loop_mean_abs_change": cambio,
                    "closed_loop_action_std": 0.1,
                    "closed_loop_capture": fijo,
                }
            },
        }
    }


def test_la_regla_de_0_02_separa_usar_de_no_usar() -> None:
    debajo = [_registro(0.0199 + 1e-6 * i, None, None) for i in range(10)]
    arriba = [_registro(0.0201 + 1e-6 * i, None, None) for i in range(10)]
    assert gate_diagnosis.summarize_arm(debajo)["uses_news"] is False
    assert gate_diagnosis.summarize_arm(arriba)["uses_news"] is True


def test_cuenta_las_politicas_que_pierden_capture_y_sin_capture_no_lo_informa() -> None:
    con = [_registro(0.1, 0.8, 0.8 - 0.1 * (i < 7)) for i in range(10)]
    r = gate_diagnosis.summarize_arm(con)
    assert r["capture"]["policies_that_lose_capture"] == 7
    assert "capture" not in gate_diagnosis.summarize_arm(
        [_registro(0.1, None, None) for _ in range(10)]
    )
