"""Re-diagnostico de B sobre datos reales (N8e, ADR 0012, 5).

Sin torch: el cargador devuelve modelos falsos cuya accion es una funcion conocida
de la observacion. Lee los datasets versionados, que se verifican por SHA256.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import study_diagnosis as sd
from agents.study_run import ARM_B, StudyRunConfig, Unit, arm_observation, policy_path
from agents.walkforward import SealError, folds, load_dataset
from envs.observation import ObservationBuilder

RAIZ = Path(__file__).resolve().parents[2]
# Paso de 1000: 3 folds, no 11. Una semilla guardada.
CONFIG = StudyRunConfig(walk=replace(StudyRunConfig().walk, step=1000))
CONFIG = replace(CONFIG, saved_policy_seeds=(1,))
TONE = ObservationBuilder(arm_observation(ARM_B, CONFIG)).names.index("tone_mean")


class _Sigue:
    """Accion = 0.5 + 0.4 * tone_mean: depende de las noticias."""

    def predict(self, obs: Any, **_k: object) -> tuple[Any, None]:
        return np.clip(0.5 + 0.4 * obs[TONE], 0.0, 1.0), None


class _Ignora:
    """Siempre invertido: las noticias no cambian nada."""

    def predict(self, obs: Any, **_k: object) -> tuple[Any, None]:
        return np.array([1.0]), None


def _guarda(carpeta: Path) -> None:
    n_bars = len(load_dataset(RAIZ)[0])
    for v in folds(n_bars, CONFIG.walk):
        ruta = Path(f"{policy_path(carpeta, Unit(ARM_B, v.index, 1))}.zip")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(b"x")


def test_una_politica_que_ignora_las_noticias_da_cero_y_la_regla_del_adr(
    tmp_path: Path,
) -> None:
    _guarda(tmp_path)
    r = sd.run(
        RAIZ, tmp_path, tmp_path / "o", config=CONFIG, loader=lambda _p: _Ignora()
    )
    assert r["n_policies"] == 3
    assert r["news_median_abs_change"] == pytest.approx(0.0)
    assert r["placebo_swap_mean_abs_change"]["median"] == pytest.approx(0.0)
    assert r["uses_news_by_adr_rule"] is False
    assert "B no usa las noticias" in r["reading"]
    assert r["n_constant_policies"] == 3
    assert r["news_median_abs_change_non_constant"] is None
    assert (tmp_path / "o" / "rediagnosis.json").is_file()


def test_una_politica_que_sigue_el_tono_se_ve_al_fijar_y_al_reemplazar(
    tmp_path: Path,
) -> None:
    _guarda(tmp_path)
    r = sd.run(
        RAIZ, tmp_path, tmp_path / "o", config=CONFIG, loader=lambda _p: _Sigue()
    )
    assert r["news_median_abs_change"] > sd.NEWS_USE_MIN_CHANGE
    assert r["placebo_swap_mean_abs_change"]["median"] > 0.0
    assert r["uses_news_by_adr_rule"] is True
    assert set(r["groups"]) == {
        "retornos",
        "tecnicos",
        "posicion",
        "resultado_orden",
        "noticias",
    }
    assert sorted(r["news_change_by_fold"]) == [0, 1, 2]


def test_falta_una_politica_guardada(tmp_path: Path) -> None:
    with pytest.raises(SealError, match="falta la politica guardada"):
        sd.run(
            RAIZ, tmp_path, tmp_path / "o", config=CONFIG, loader=lambda _p: _Ignora()
        )


@pytest.mark.parametrize(
    ("mediana", "frase"),
    [
        (0.0106, "B no usa las noticias"),
        (0.02, "no se distingue del piso de ruido"),
        (0.0252, "no se distingue del piso de ruido"),
        (0.05, "indicio debil, no concluyente"),
        (0.100, "indicio debil, no concluyente"),
        (0.101, "B reacciona a las noticias"),
    ],
)
def test_lectura_contra_el_piso_de_ruido_a_mano(mediana: float, frase: str) -> None:
    assert frase in sd.read_news_change(mediana)


def _reg(fold: int, std: float, cambio: float, swap: float) -> dict[str, Any]:
    grupo = {"open_loop_mean_abs_change": cambio, "closed_loop_action_std": std}
    return {
        "fold": fold,
        "seed": 1,
        "n_steps": 10,
        "diagnosis": {
            "action_std": std,
            "placebo_swap_mean_abs_change": swap,
            "groups": {"noticias": grupo, "tecnicos": grupo},
        },
    }


def test_resumen_a_mano_separa_las_politicas_constantes() -> None:
    # 10 politicas (el minimo de ``summarize``): 4 constantes con cambio 0, el
    # resto con cambio 0.04; la mediana de las 10 es 0.04 (6 valores 0.04 > 5.o).
    regs = [_reg(i, 0.0, 0.0, 0.0) for i in range(4)]
    regs += [_reg(4 + i, 0.3, 0.04, 0.1) for i in range(6)]
    r = sd.summarize_policies(regs)
    assert r["news_median_abs_change"] == pytest.approx(0.04)
    assert r["n_constant_policies"] == 4
    assert r["news_median_abs_change_non_constant"] == pytest.approx(0.04)
    assert r["news_change_by_fold"][0] == 0.0
    assert r["news_change_by_fold"][9] == pytest.approx(0.04)
    assert "indicio debil" in r["reading"]
