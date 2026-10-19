"""FrozenPolicy: el manifest es lo que prueba que opera lo que se congelo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.protocol import STUDY_SEEDS
from features.scaler import FeatureScaler
from live.frozen import (
    ARCHIVOS,
    MANIFEST,
    FrozenError,
    FrozenPolicy,
    agent_dir,
    sha256_file,
    write_manifest,
)

NOMBRES = ("a", "b")


class ModeloFalso:
    """Registra la observacion que recibe; devuelve siempre 0.5."""

    def __init__(self) -> None:
        self.vistas: list[Any] = []

    def predict(self, obs: Any, deterministic: bool) -> tuple[Any, None]:
        assert deterministic
        self.vistas.append(obs)
        return np.array([0.5]), None


def congelar(raiz: Path) -> None:
    escalador = FeatureScaler(
        names=NOMBRES,
        mean=np.array([10.0, 0.0]),
        std=np.array([2.0, 1.0]),
        scale_mask=(True, False),
        n_samples=5,
    )
    for seed in STUDY_SEEDS:
        base = agent_dir(raiz, seed)
        base.mkdir(parents=True)
        (base / "model.zip").write_bytes(f"pesos-{seed}".encode())
        (base / "scaler.json").write_text(json.dumps(escalador.to_dict()))
        (base / "config.json").write_text(json.dumps({"ppo": {"recurrent": False}}))
    write_manifest(raiz, {"git_commit": "abc"})


def test_carga_y_normaliza_con_el_escalador_congelado(tmp_path: Path) -> None:
    congelar(tmp_path)
    modelo = ModeloFalso()
    agente = FrozenPolicy.load(tmp_path, 23, loader=lambda _p: modelo)
    assert agente.act(np.array([14.0, 0.7])) == 0.5
    # (14 - 10) / 2 = 2.0 en la columna escalada; la otra pasa sin tocar.
    assert modelo.vistas[0].tolist() == [2.0, 0.7]


def test_el_manifest_lista_las_diez_semillas_y_los_tres_archivos(
    tmp_path: Path,
) -> None:
    congelar(tmp_path)
    m = json.loads((tmp_path / MANIFEST).read_text())
    assert m["seeds"] == list(STUDY_SEEDS)
    assert set(m["agents"]["11"]) == set(ARCHIVOS)
    assert m["agents"]["11"]["model.zip"] == sha256_file(
        agent_dir(tmp_path, 11) / "model.zip"
    )


def test_un_peso_alterado_no_carga(tmp_path: Path) -> None:
    congelar(tmp_path)
    (agent_dir(tmp_path, 41) / "model.zip").write_bytes(b"otros pesos")
    with pytest.raises(FrozenError, match=r"seed-41/model\.zip: SHA256"):
        FrozenPolicy.load(tmp_path, 41, loader=lambda _p: ModeloFalso())
    FrozenPolicy.load(tmp_path, 11, loader=lambda _p: ModeloFalso())  # las demas siguen


def test_un_escalador_alterado_no_carga(tmp_path: Path) -> None:
    congelar(tmp_path)
    (agent_dir(tmp_path, 11) / "scaler.json").write_text("{}")
    with pytest.raises(FrozenError, match=r"seed-11/scaler\.json: SHA256"):
        FrozenPolicy.load(tmp_path, 11, loader=lambda _p: ModeloFalso())


def test_no_se_congela_con_nueve_agentes(tmp_path: Path) -> None:
    congelar(tmp_path)
    (agent_dir(tmp_path, 101) / "config.json").unlink()
    with pytest.raises(FrozenError, match=r"seed-101: faltan \['config\.json'\]"):
        write_manifest(tmp_path, {})


def test_un_manifest_sin_alguna_semilla_no_carga(tmp_path: Path) -> None:
    congelar(tmp_path)
    ruta = tmp_path / MANIFEST
    m = json.loads(ruta.read_text())
    m["seeds"] = m["seeds"][:-1]
    ruta.write_text(json.dumps(m))
    with pytest.raises(FrozenError, match="ningun agente se descarta"):
        FrozenPolicy.load(tmp_path, 11, loader=lambda _p: ModeloFalso())


def test_sin_manifest_no_carga(tmp_path: Path) -> None:
    with pytest.raises(FrozenError, match="no hay manifest.json"):
        FrozenPolicy.load(tmp_path, 11, loader=lambda _p: ModeloFalso())
