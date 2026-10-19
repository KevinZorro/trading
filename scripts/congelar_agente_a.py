"""Congela el Agente A: un PPO por semilla de ``STUDY_SEEDS`` (ADR 0009).

Uso (desde la raiz, sin red, con el grupo ``rl`` instalado)::

    uv sync --frozen --group rl
    PYTHONPATH=src uv run python scripts/congelar_agente_a.py --out frozen/agente_a

Entrena con las ultimas 1000 barras de ``datasets/binance/BTCUSDT-1d.csv`` (verificado
por su SHA256), el escenario de costos ``principal`` y el ``min_notional`` del
``exchangeInfo`` grabado (ADR 0008). Escribe ``seed-<n>/{model.zip,scaler.json,
config.json}`` y ``manifest.json`` con el SHA256 de todo. Se niega a correr con el
arbol sucio y a pisar un congelado existente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path

# torch antes que pandas/pyarrow: en Windows, el orden inverso falla con
# WinError 1114 al cargar c10.dll (conflicto de runtimes OpenMP/CRT).
import torch  # noqa: F401  isort: skip

from agents.ppo import train_ppo
from agents.protocol import STUDY_SEEDS
from agents.runner import build_env, scaler_for
from agents.walkforward import PRINCIPAL, WalkForwardConfig, load_dataset
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig
from live.config import MARKET_DATA_URL
from live.frozen import MANIFEST, agent_dir, sha256_file, write_manifest
from live.marketdata import BinanceRestMarketData
from sim.clock import SystemClock
from sim.sizing import TargetWeightSizer

RAIZ = Path(__file__).resolve().parent.parent
ADR = RAIZ / "docs/adr/0009-agente-a-congelado-todas-las-semillas.md"
EXCHANGE_INFO = RAIZ / "tests/live/respuestas/exchange_info.json"
BARRAS = 1000


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args(argv).out
    if (out / MANIFEST).exists():
        sys.exit(f"{out / MANIFEST} ya existe: un congelado no se pisa")
    if _git("status", "--porcelain", "--untracked-files=no"):
        sys.exit(
            "el arbol tiene cambios sin commitear: el commit del manifest mentiria"
        )

    config = WalkForwardConfig()
    serie, _, dataset_sha = load_dataset(RAIZ)
    spec = BinanceRestMarketData(
        MARKET_DATA_URL, SystemClock(), fetch=lambda _url: EXCHANGE_INFO.read_bytes()
    ).instrument("BTCUSDT")
    train = replace(serie.slice(len(serie) - BARRAS, len(serie)), instrument=spec)

    observacion = EnvConfig().observation
    escalador = scaler_for(train, observacion)
    env_config = EnvConfig(
        observation=observacion, sizer=TargetWeightSizer(safety=config.safety)
    )
    sim_config = config.sim_config(PRINCIPAL)
    ppo = config.ppo_config()

    for seed in STUDY_SEEDS:
        politica = train_ppo(
            lambda: build_env(
                train,
                sim_config,
                scaler=escalador,
                env_config=env_config,
                reward=NetReturnReward(scale=config.reward_scale),
            ),
            ppo,
            seed=seed,
        )
        base = agent_dir(out, seed)
        base.mkdir(parents=True, exist_ok=True)
        politica.save(str(base / "model.zip"))
        (base / "scaler.json").write_text(
            json.dumps(escalador.to_dict(), indent=2), newline="\n"
        )
        (base / "config.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "ppo": ppo.describe(),
                    "env": env_config.describe(),
                    "sim": sim_config.describe(),
                    "train_bars": BARRAS,
                    "train_first_timestamp": str(train.timestamp[0]),
                    "train_last_timestamp": str(train.timestamp[-1]),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            newline="\n",
        )
        print(f"seed {seed}: listo", flush=True)

    write_manifest(
        out,
        {
            "git_commit": _git("rev-parse", "HEAD"),
            "adr_0009_sha256": sha256_file(ADR),
            "dataset_sha256": dataset_sha,
            "exchange_info_sha256": hashlib.sha256(
                EXCHANGE_INFO.read_bytes()
            ).hexdigest(),
            "versions": {
                p: metadata.version(p) for p in ("torch", "stable-baselines3", "numpy")
            },
            "walkforward_config": config.describe(),
        },
    )
    print(f"manifest en {out / MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
