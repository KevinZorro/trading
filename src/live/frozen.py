"""Agente A congelado: pesos, escalador y configuracion verificados por SHA256.

ADR 0009. Un agente por semilla de ``STUDY_SEEDS``; cada uno vive en
``<raiz>/seed-<n>/`` con ``model.zip``, ``scaler.json`` y ``config.json``, y el
``manifest.json`` de la raiz lista el SHA256 de cada archivo. Cargar un agente cuyo
archivo no coincide con el manifest es un error: lo que opera en sombra tiene que ser
exactamente lo que se congelo, y el manifest es lo que lo prueba.

``torch`` no es dependencia de este modulo. El modelo se carga con un ``loader``
inyectable cuyo default importa ``stable_baselines3`` de forma diferida, igual que
``agents.ppo``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.policy import Policy
from agents.ppo import SB3Policy
from agents.protocol import STUDY_SEEDS
from data.schema import FloatArray
from features.scaler import FeatureScaler

MANIFEST = "manifest.json"
ARCHIVOS = ("model.zip", "scaler.json", "config.json")

Loader = Callable[[Path], Any]


class FrozenError(RuntimeError):
    """El agente congelado no es el que el manifest declara."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def agent_dir(root: Path, seed: int) -> Path:
    return root / f"seed-{seed}"


def write_manifest(root: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Hashea los archivos de las diez semillas y escribe ``manifest.json``.

    Falla si falta cualquiera: un congelado con nueve agentes no es el del ADR.
    """
    agentes: dict[str, dict[str, str]] = {}
    for seed in STUDY_SEEDS:
        base = agent_dir(root, seed)
        faltan = [a for a in ARCHIVOS if not (base / a).is_file()]
        if faltan:
            raise FrozenError(f"seed-{seed}: faltan {faltan}")
        agentes[str(seed)] = {a: sha256_file(base / a) for a in ARCHIVOS}
    manifest = {**meta, "seeds": list(STUDY_SEEDS), "agents": agentes}
    (root / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n"
    )
    return manifest


def _default_loader(path: Path) -> Any:  # pragma: no cover - necesita torch
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


@dataclass(frozen=True)
class FrozenPolicy:
    """``Policy`` determinista mas el escalador con que se entreno."""

    seed: int
    policy: Policy
    scaler: FeatureScaler
    config: Mapping[str, Any]

    @classmethod
    def load(
        cls, root: Path, seed: int, *, loader: Loader = _default_loader
    ) -> FrozenPolicy:
        manifest_path = root / MANIFEST
        if not manifest_path.is_file():
            raise FrozenError(f"no hay {MANIFEST} en {root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("seeds") != list(STUDY_SEEDS):
            raise FrozenError(
                f"el manifest declara semillas {manifest.get('seeds')}, el ADR 0009 "
                f"exige {list(STUDY_SEEDS)}: ningun agente se descarta ni se agrega"
            )
        esperado = manifest["agents"].get(str(seed))
        if esperado is None:
            raise FrozenError(f"el manifest no lista la semilla {seed}")
        base = agent_dir(root, seed)
        for archivo in ARCHIVOS:
            real = sha256_file(base / archivo)
            if real != esperado[archivo]:
                raise FrozenError(
                    f"seed-{seed}/{archivo}: SHA256 {real[:12]}... != "
                    f"{esperado[archivo][:12]}... del manifest"
                )
        scaler = FeatureScaler.from_dict(json.loads((base / "scaler.json").read_text()))
        config = json.loads((base / "config.json").read_text())
        modelo = loader(base / "model.zip")
        return cls(
            seed=seed,
            policy=SB3Policy(
                modelo, recurrent=bool(config["ppo"]["recurrent"]), label=f"A{seed}"
            ),
            scaler=scaler,
            config=config,
        )

    def act(self, observation_raw: FloatArray) -> float:
        """Normaliza con el escalador congelado y decide. Nunca reajusta."""
        return self.policy.act(self.scaler.transform(observation_raw))
