"""Agentes A congelados dentro de la sombra: aislamiento del snapshot, libro comun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.protocol import STUDY_SEEDS
from envs.observation import ObservationBuilder
from features.scaler import FeatureScaler
from live.config import ConfigError, ShadowConfig
from live.fillstore import FillStore, venue_header
from live.frozen import agent_dir, write_manifest
from live.schedule import Due
from live.shadow import build_runner
from live.shadow_agents import AGENTS_DIR
from sim.clock import SimulatedClock

from .conftest import Servidor, grabacion

AHORA = np.datetime64("2026-10-17T00:00:30", "ns")
DIARIA = Due(AHORA, "daily")


class Modelo:
    def __init__(self, accion: float = 1.0, falla: bool = False) -> None:
        self.accion = accion
        self.falla = falla

    def predict(self, obs: Any, deterministic: bool) -> tuple[Any, None]:
        if self.falla:
            raise RuntimeError("pesos corruptos")
        return np.array([self.accion]), None


def congelar(raiz: Path) -> None:
    nombres = ObservationBuilder().names
    escalador = FeatureScaler.identity(nombres)
    for seed in STUDY_SEEDS:
        base = agent_dir(raiz, seed)
        base.mkdir(parents=True)
        (base / "model.zip").write_bytes(f"pesos-{seed}".encode())
        (base / "scaler.json").write_text(json.dumps(escalador.to_dict()))
        (base / "config.json").write_text(json.dumps({"ppo": {"recurrent": False}}))
    write_manifest(raiz, {"git_commit": "abc"})


def armar(
    tmp_path: Path,
    frozen: Path | None,
    modelo: Modelo | None = None,
    servidor: Servidor | None = None,
) -> Any:
    modelo = modelo or Modelo()
    reloj = SimulatedClock(AHORA)
    return build_runner(
        ShadowConfig(data_dir=tmp_path / "datos", frozen_dir=frozen),
        reloj,
        sleep=lambda s: None,
        fetch=servidor or Servidor(),
        agent_loader=lambda _p: modelo,
    )


def fills_de(tmp_path: Path, seed: int) -> list[dict[str, Any]]:
    almacen = FillStore(
        tmp_path / "datos" / AGENTS_DIR / f"seed-{seed}",
        venue_header(initial_cash=100.0, spec_sha256="", spec_received_at=AHORA),
    )
    return [r for r in almacen.replay() if r["type"] == "fill"]


def heartbeat(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "datos" / "heartbeat.json").read_text())


def test_los_diez_agentes_operan_el_mismo_libro_del_snapshot(tmp_path: Path) -> None:
    congelar(tmp_path / "frozen")
    servidor = Servidor()
    runner = armar(tmp_path, tmp_path / "frozen", servidor=servidor)
    registro = runner.step(DIARIA)
    assert registro.valid
    # El libro se pide una sola vez: snapshot y diez agentes ven el mismo.
    assert sum(u.split("?")[0].endswith("/depth") for u in servidor.urls) == 1
    mejor_ask = float(grabacion("depth.json")["asks"][0][0])
    for seed in STUDY_SEEDS:
        [fill] = fills_de(tmp_path, seed)
        assert fill["fill"]["fill_price"] == mejor_ask
        assert fill["fill"]["client_order_id"].startswith(f"A{seed}-")
    estado = heartbeat(tmp_path)["agents"]
    assert estado["loaded"] == list(STUDY_SEEDS)
    assert estado["errors"] == {}


def test_cada_agente_tiene_su_propio_ledger(tmp_path: Path) -> None:
    congelar(tmp_path / "frozen")
    armar(tmp_path, tmp_path / "frozen").step(DIARIA)
    dirs = sorted(p.name for p in (tmp_path / "datos" / AGENTS_DIR).iterdir())
    assert dirs == sorted(f"seed-{s}" for s in STUDY_SEEDS)


def test_el_snapshot_periodico_no_corre_agentes(tmp_path: Path) -> None:
    congelar(tmp_path / "frozen")
    runner = armar(tmp_path, tmp_path / "frozen")
    runner.step(Due(AHORA, "periodic"))
    assert not (tmp_path / "datos" / AGENTS_DIR).exists()
    assert heartbeat(tmp_path)["agents"]["loaded"] == []


def test_sin_pesos_el_snapshot_se_toma_igual(tmp_path: Path) -> None:
    runner = armar(tmp_path, tmp_path / "no-existe")
    registro = runner.step(DIARIA)
    assert registro.valid
    errores = heartbeat(tmp_path)["agents"]["errors"]
    assert sorted(errores) == sorted(str(s) for s in STUDY_SEEDS)
    assert "FrozenError" in errores[str(STUDY_SEEDS[0])]


def test_un_agente_con_pesos_adulterados_no_frena_a_los_otros(tmp_path: Path) -> None:
    congelar(tmp_path / "frozen")
    malo = STUDY_SEEDS[3]
    (agent_dir(tmp_path / "frozen", malo) / "model.zip").write_bytes(b"otro")
    runner = armar(tmp_path, tmp_path / "frozen")
    assert runner.step(DIARIA).valid
    estado = heartbeat(tmp_path)["agents"]
    assert list(estado["errors"]) == [str(malo)]
    assert "SHA256" in estado["errors"][str(malo)]
    assert estado["loaded"] == [s for s in STUDY_SEEDS if s != malo]


def test_un_bug_del_agente_queda_registrado_y_el_snapshot_ya_estaba_escrito(
    tmp_path: Path,
) -> None:
    congelar(tmp_path / "frozen")
    runner = armar(tmp_path, tmp_path / "frozen", Modelo(falla=True))
    assert runner.step(DIARIA).valid
    snapshots = (tmp_path / "datos" / "snapshots-2026-10-17.jsonl").read_text()
    assert '"valid": true' in snapshots
    errores = heartbeat(tmp_path)["agents"]["errors"]
    assert all("pesos corruptos" in m for m in errores.values())
    assert len(errores) == len(STUDY_SEEDS)


def test_repetir_la_barra_no_duplica_ordenes(tmp_path: Path) -> None:
    congelar(tmp_path / "frozen")
    runner = armar(tmp_path, tmp_path / "frozen")
    runner.step(DIARIA)
    runner.step(DIARIA)
    assert all(len(fills_de(tmp_path, s)) == 1 for s in STUDY_SEEDS)


def test_config_lee_frozen_dir_y_lo_describe(tmp_path: Path) -> None:
    sin = ShadowConfig.from_env({"SHADOW_DATA_DIR": str(tmp_path)})
    assert sin.frozen_dir is None
    assert sin.describe()["frozen_dir"] is None
    con = ShadowConfig.from_env(
        {"SHADOW_DATA_DIR": str(tmp_path), "SHADOW_FROZEN_DIR": "/frozen"}
    )
    assert con.frozen_dir == Path("/frozen")
    assert con.describe()["frozen_dir"] == "/frozen"


def test_sin_frozen_dir_no_hay_agentes(tmp_path: Path) -> None:
    runner = armar(tmp_path, None)
    assert runner.step(DIARIA).valid
    assert "agents" not in heartbeat(tmp_path)


def test_config_invalida_sigue_siendo_error() -> None:
    with pytest.raises(ConfigError, match="falta SHADOW_DATA_DIR"):
        ShadowConfig.from_env({"SHADOW_FROZEN_DIR": "/frozen"})
