"""El congelado versionado en frozen/agente_a verifica contra su manifest.

Existe por un fallo real: el congelado se generó en Windows con CRLF, el repo guarda
LF y `FrozenPolicy.load` fallaba en cualquier checkout Linux (el VPS de la sombra).
Corre sin torch: solo compara SHA256.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.protocol import STUDY_SEEDS
from live.frozen import ARCHIVOS, MANIFEST, agent_dir, sha256_file

RAIZ = Path(__file__).resolve().parents[2] / "frozen" / "agente_a"

pytestmark = pytest.mark.skipif(
    not (RAIZ / MANIFEST).is_file(), reason="no hay congelado versionado"
)


@pytest.mark.parametrize("seed", STUDY_SEEDS)
def test_los_archivos_versionados_coinciden_con_el_manifest(seed: int) -> None:
    manifest = json.loads((RAIZ / MANIFEST).read_text())
    for archivo in ARCHIVOS:
        assert (
            sha256_file(agent_dir(RAIZ, seed) / archivo)
            == (manifest["agents"][str(seed)][archivo])
        ), f"seed-{seed}/{archivo}"


def test_ningun_json_congelado_tiene_crlf() -> None:
    for seed in STUDY_SEEDS:
        for archivo in ("scaler.json", "config.json"):
            assert b"\r" not in (agent_dir(RAIZ, seed) / archivo).read_bytes()
