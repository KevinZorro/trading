"""Escalera de regularizacion (ADR 0011): configuracion, parada y contraste con G1.

Como en ``test_calibration``, nada entrena: los criterios se aplican sobre
registros fabricados a mano. Lo unico que corre un episodio es ``train_unit`` con
un peso constante, para fijar que la configuracion del peldano llega al entorno.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import calibration as cal
from agents import regularization as reg
from agents.cli import main
from agents.policy import ConstantWeightPolicy
from agents.ppo import PPOConfig
from agents.protocol import STUDY_SEEDS
from tests.agents.test_calibration import registro

RAIZ = Path(__file__).resolve().parents[2]


def poblar(
    out: Path,
    point: str,
    *,
    invertido: float = 0.9,
    dispersion: float = 0.05,
    paso_por_camino: float = 0.0,
) -> None:
    """``paso_por_camino`` suma ``i * paso`` a la dispersion del camino ``i``."""
    unidades = reg.rung_units(point) if reg.is_rung(point) else cal.point_units(point)
    for unit in unidades:
        i = cal.CALIBRATION_PATH_SEEDS.index(unit.path_seed)
        cal.write_atomic(
            cal.unit_path(out, unit),
            registro(
                unit,
                invertido=invertido,
                dispersion=dispersion + i * paso_por_camino,
            ),
        )


# ---------------------------------------------------------------------------
# Lo que el ADR fija
# ---------------------------------------------------------------------------


def test_la_escalera_es_la_elegida_en_el_adr() -> None:
    r1, r2, r3 = reg.RUNGS
    assert [r.name for r in reg.RUNGS] == ["R1", "R2", "R3"]
    assert r1.observation.lookback == 1
    assert (r1.ppo.learning_rate, r1.ppo.n_epochs) == (1e-4, 5)
    assert (r1.ppo.net_arch, r1.ppo.gamma) == ((64, 64), 0.99)
    assert (r2.ppo.net_arch, r2.ppo.gamma) == ((32, 32), 0.99)
    assert (r3.ppo.net_arch, r3.ppo.gamma) == ((32, 32), 0.9)
    assert all(r.ppo.total_timesteps == 120_000 for r in reg.RUNGS)


def test_los_peldanos_son_acumulativos_y_no_tocan_nada_mas() -> None:
    """Fuera de lo declarado, cada peldano es la configuracion de main (G1)."""
    base = PPOConfig(total_timesteps=120_000).describe()
    declarado = {"learning_rate", "n_epochs", "net_arch", "gamma"}
    for r in reg.RUNGS:
        propio = r.ppo.describe()
        assert {k: v for k, v in propio.items() if k not in declarado} == {
            k: v for k, v in base.items() if k not in declarado
        }
    assert reg.RUNGS[0].ppo.clip_range == 0.2


def test_el_adr_0011_en_disco_es_el_sellado() -> None:
    assert reg.verify_adr(RAIZ) == reg.ADR_SHA256


def test_un_adr_0011_modificado_no_corre(tmp_path: Path) -> None:
    destino = tmp_path / reg.ADR_PATH
    destino.parent.mkdir(parents=True)
    destino.write_text("otro texto")
    with pytest.raises(cal.CalibrationError, match="0011.*el criterio cambio"):
        reg.verify_adr(tmp_path)


def test_cada_peldano_se_parea_con_los_entrenamientos_de_g1() -> None:
    for r in reg.RUNGS:
        propios = {(u.path_seed, u.agent_seed) for u in reg.rung_units(r.name)}
        g1 = {(u.path_seed, u.agent_seed) for u in cal.point_units("G1")}
        assert propios == g1
        assert len(propios) == 50


def test_peldano_desconocido_se_rechaza() -> None:
    with pytest.raises(cal.CalibrationError, match="peldano desconocido 'R9'"):
        reg.rung_units("R9")


def test_no_se_mezclan_entrenamientos_de_las_dos_rondas(tmp_path: Path) -> None:
    cal.acquire_seal(tmp_path, cal.ADR_SHA256)
    with pytest.raises(cal.CalibrationError, match="es de otra configuracion"):
        cal.acquire_seal(tmp_path, reg.ADR_SHA256, reg.config_digest())


# ---------------------------------------------------------------------------
# Regla de parada y contraste pareado
# ---------------------------------------------------------------------------


def test_se_para_en_el_primer_peldano_que_pasa(tmp_path: Path) -> None:
    poblar(tmp_path, "R1", dispersion=0.30)
    poblar(tmp_path, "R2", dispersion=0.05)
    # R3 no existe en disco: si decide lo cargara, fallaria por faltantes.
    resultado = reg.decide(tmp_path, tmp_path, log=lambda _: None)
    assert resultado["selected_rung"] == "R2"
    assert [p["verdict"] for p in resultado["rungs"]] == ["FAIL", "PASS"]
    assert resultado["rungs"][1]["config"]["observation"]["lookback"] == 1


def test_si_ningun_peldano_pasa_no_se_fija_configuracion(tmp_path: Path) -> None:
    for nombre in ("R1", "R2", "R3"):
        poblar(tmp_path, nombre, dispersion=0.30)
    resultado = reg.decide(tmp_path, tmp_path, log=lambda _: None)
    assert resultado["selected_rung"] is None
    assert "Ningun peldano paso" in reg.render(resultado)


def test_el_contraste_contra_g1_es_por_camino(tmp_path: Path) -> None:
    """G1 en 0.34 en todo; R1 en 0.24 - 0.01*i en el camino i.

    Diferencia por camino: -0.10, -0.11, -0.12, -0.13, -0.14 -> media -0.12,
    desvio entre caminos 0.0158, error estandar 0.0071, t = -16.97.
    """
    base, propio = tmp_path / "g", tmp_path / "r"
    poblar(base, "G1", dispersion=0.34)
    poblar(propio, "R1", dispersion=0.24, paso_por_camino=-0.01)
    poblar(propio, "R2", dispersion=0.05)
    resultado = reg.decide(propio, base, log=lambda _: None)
    pareado = resultado["rungs"][0]["paired_vs_baseline"]["action_std"]
    assert pareado["path_medians"] == pytest.approx([-0.10, -0.11, -0.12, -0.13, -0.14])
    assert pareado["mean"] == pytest.approx(-0.12)
    assert pareado["t_statistic"] == pytest.approx(-16.97, abs=0.01)
    tiempo = resultado["rungs"][0]["paired_vs_baseline"]["time_invested"]
    assert tiempo["path_medians"] == pytest.approx([0.0] * 5)
    assert "action_std contra G1 -0.120" in reg.render(resultado)


def test_sin_g1_completo_el_contraste_se_declara_ausente(tmp_path: Path) -> None:
    poblar(tmp_path, "R1", dispersion=0.05)
    resultado = reg.decide(tmp_path, tmp_path / "vacia", log=lambda _: None)
    assert resultado["baseline_found"] is False
    assert resultado["rungs"][0]["paired_vs_baseline"] is None
    assert "contraste pareado no se calculo" in reg.render(resultado)


def test_el_manifest_lleva_solo_los_peldanos_juzgados(tmp_path: Path) -> None:
    poblar(tmp_path, "R1", dispersion=0.05)
    resultado = reg.decide(tmp_path, tmp_path, log=lambda _: None)
    reg.finalize(tmp_path, resultado)
    manifest = (tmp_path / "MANIFEST.json").read_text()
    assert manifest.count("R1_p") == 50
    assert "R2_p" not in manifest
    assert reg.ADR_SHA256 in manifest


# ---------------------------------------------------------------------------
# CLI y registro de un entrenamiento
# ---------------------------------------------------------------------------


def test_cli_ladder_report_juzga_lo_guardado_y_sella(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(RAIZ)
    poblar(tmp_path, "R1", dispersion=0.05)
    args = ["calibrate", "ladder-report", "--out", str(tmp_path)]
    assert main([*args, "--baseline", str(tmp_path)]) == 0
    assert "C* = R1 a 120000 timesteps" in capsys.readouterr().out
    assert reg.config_digest() in (tmp_path / "SEAL.json").read_text()


def test_cli_ladder_entrena_cada_peldano_con_sus_unidades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(RAIZ)
    lanzados: list[str] = []

    def run_point(out: Path, point: str, workers: int, units: object) -> None:
        lanzados.append(point)
        poblar(out, point, dispersion=0.30)

    monkeypatch.setattr(cal, "run_point", run_point)
    args = ["calibrate", "ladder", "--out", str(tmp_path), "--workers", "3"]
    assert main(args) == 0
    assert lanzados == ["R1", "R2", "R3"]


def test_cli_worker_de_un_peldano_usa_su_configuracion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vistos: list[str] = []

    def entrenar(unit: cal.Unit) -> dict[str, object]:
        vistos.append(unit.point)
        return {"wall_seconds": 1.0}

    monkeypatch.setattr(reg, "train_unit", entrenar)
    args = ["calibrate", "worker", "--point", "R2", "--shard", "0/5"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    assert vistos == ["R2"] * 10


def test_el_registro_lleva_la_observacion_y_el_ppo_del_peldano() -> None:
    unidad = reg.rung_units("R3")[0]
    registro_unidad = reg.train_unit(
        unidad, policy_factory=lambda train, seed: ConstantWeightPolicy(1.0)
    )
    assert registro_unidad["unit"] == f"R3_p809_s{STUDY_SEEDS[0]}"
    assert registro_unidad["observation"]["lookback"] == 1
    assert registro_unidad["arm"]["ppo"]["gamma"] == 0.9
    assert registro_unidad["arm"]["ppo"]["net_arch"] == [32, 32]
    assert registro_unidad["timesteps"] == 120_000
