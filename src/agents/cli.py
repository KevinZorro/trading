"""Corredor del protocolo de validacion. ``python -m agents.cli --help``.

Dos subcomandos y no uno, porque entrenar 90 configuraciones en un solo proceso
tarda horas y no se puede paralelizar por dentro sin que los procesos de torch
se peleen por los mismos nucleos:

- ``arm`` entrena y evalua **un** brazo y deja su JSON. Se pueden correr varios
  en paralelo.
- ``assemble`` lee los JSON y aplica los criterios. Los criterios se aplican
  siempre aca, sobre resultados guardados, asi que se pueden revisar sin
  reentrenar -y se ve en el diff si alguien los cambio despues de ver los
  numeros-.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agents.experiment import ExperimentLog, jsonable
from agents.ppo import PPOConfig
from agents.protocol import (
    ArmResult,
    MultiPathResult,
    ProtocolThresholds,
    assemble_protocol,
    run_arm,
    run_multipath_arm,
)
from data.fixtures import (
    DEFAULT_N_BARS,
    SNR_SWEEP_BETAS,
    Fixture,
    level_0_deterministic,
    level_1_noisy,
    level_2_costly,
    level_3_regime_flip,
    level_4_control,
)

# Semillas del estudio. Fijas y explicitas: un rango generado al vuelo hace que
# "10 semillas" signifique cosas distintas en dos corridas.
STUDY_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 59, 67, 73, 89, 97, 101)

# Semilla del proceso generador de cada fixture. Separada de las del agente para
# que las 10 semillas midan la varianza **del entrenamiento** sobre una serie
# fija, y no la varianza combinada de serie y entrenamiento, que son dos fuentes
# distintas y se confundirian en un solo numero.
FIXTURE_SEED = 20240115


def build_fixture_with_seed(
    level: str, *, beta: float | None, n_bars: int, seed: int
) -> Fixture:
    """El fixture de un nivel con la semilla del **proceso** explicita.

    Separar esta semilla de las del agente es lo que permite muestrear caminos:
    cambiarla da otra realizacion del mismo mercado, no otro mercado.
    """
    if level == "level_0":
        # Determinista por construccion: no admite semilla, y por eso su
        # varianza de mercado es exactamente cero. N=1 ahi es completo.
        return level_0_deterministic(min(n_bars, 4_000))
    if level == "level_1":
        return level_1_noisy(n_bars, seed=seed, beta=beta or -0.3)
    if level == "level_2":
        return level_2_costly(n_bars, seed=seed, beta=beta or -0.3)
    if level == "level_3":
        return level_3_regime_flip(n_bars, seed=seed, beta=beta or -0.3)
    if level == "level_4":
        return level_4_control(n_bars, seed=seed)
    raise SystemExit(f"nivel desconocido: {level}")


def build_fixture(level: str, *, beta: float | None, n_bars: int) -> Fixture:
    return build_fixture_with_seed(level, beta=beta, n_bars=n_bars, seed=FIXTURE_SEED)


#: Semillas de los caminos del nivel 4. Separadas de las del agente: estas
#: muestrean el **mercado** y aquellas el entrenamiento, y son las dos varianzas
#: que el reporte tiene que separar.
PATH_SEEDS: tuple[int, ...] = (
    701,
    709,
    719,
    727,
    733,
    739,
    743,
    751,
    757,
    761,
)


def cmd_arm(args: argparse.Namespace) -> int:
    fixture = build_fixture(args.level, beta=args.beta, n_bars=args.bars)
    config = PPOConfig(
        total_timesteps=args.timesteps,
        recurrent=args.recurrent,
    )
    # El beta solo entra en la etiqueta donde el nivel lo usa: los niveles 0 y 4
    # no tienen barrido, y una etiqueta "level_4_beta-0.3" haria creer que si.
    usa_beta = args.level in ("level_1", "level_2", "level_3")
    etiqueta = args.label or (
        f"{args.level}"
        + (f"_beta{args.beta}" if usa_beta and args.beta is not None else "")
        + ("_lstm" if args.recurrent else "")
    )
    resultado = run_arm(
        fixture,
        label=etiqueta,
        seeds=list(STUDY_SEEDS[: args.seeds]),
        ppo_config=config,
        allow_fewer_seeds=args.seeds < 10,
    )
    destino = Path(args.out)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"arm_{etiqueta}.json"
    ruta.write_text(json.dumps(jsonable(resultado.describe()), indent=2))
    print(f"escrito {ruta}")
    for clave in ("capture", "total_return_mark", "turnover_annualized"):
        print("  " + resultado.distributions[clave].render())
    return 0


def cmd_multipath(args: argparse.Namespace) -> int:
    """Un nivel sobre N caminos independientes x M semillas.

    Es la unica forma de separar la varianza de mercado de la de entrenamiento
    sobre un fixture sintetico. En datos reales no hay equivalente: N=1 por
    construccion.
    """
    config = PPOConfig(total_timesteps=args.timesteps, recurrent=args.recurrent)
    etiqueta = args.label or args.level
    caminos = list(PATH_SEEDS[: args.paths])
    resultado = run_multipath_arm(
        lambda semilla: build_fixture_with_seed(
            args.level, beta=args.beta, n_bars=args.bars, seed=semilla
        ),
        label=etiqueta,
        path_seeds=caminos,
        agent_seeds=list(STUDY_SEEDS[: args.seeds]),
        ppo_config=config,
        allow_fewer_seeds=args.seeds < 10,
    )
    destino = Path(args.out)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"multipath_{etiqueta}.json"
    ruta.write_text(json.dumps(jsonable(resultado.describe()), indent=2))
    print(f"escrito {ruta}")
    for descomposicion in resultado.decompositions.values():
        print("  " + descomposicion.render())
    return 0


def _load_multipath(directory: Path) -> dict[str, MultiPathResult]:
    salida: dict[str, MultiPathResult] = {}
    for ruta in sorted(directory.glob("multipath_*.json")):
        resultado = MultiPathResult.from_dict(json.loads(ruta.read_text()))
        salida[resultado.label] = resultado
    return salida


def _load_arms(directory: Path) -> dict[str, ArmResult]:
    brazos: dict[str, ArmResult] = {}
    for ruta in sorted(directory.glob("arm_*.json")):
        datos: dict[str, Any] = json.loads(ruta.read_text())
        brazo = ArmResult.from_dict(datos)
        brazos[brazo.label] = brazo
    return brazos


def cmd_assemble(args: argparse.Namespace) -> int:
    carpeta = Path(args.out)
    brazos = _load_arms(carpeta)
    if not brazos:
        raise SystemExit(f"no hay brazos en {carpeta}")

    def buscar(prefijo: str) -> list[ArmResult]:
        return [b for k, b in brazos.items() if k.startswith(prefijo)]

    nivel_0 = brazos.get("level_0")
    if nivel_0 is None:
        raise SystemExit("falta el brazo del nivel 0: el protocolo arranca ahi")

    nivel_1 = [b for b in buscar("level_1") if not b.label.endswith("_lstm")]
    nivel_2 = brazos.get("level_2_beta-0.3") or brazos.get("level_2")
    referencia = brazos.get("level_1_beta-0.3")
    nivel_3 = buscar("level_3")
    multicamino = _load_multipath(carpeta)
    nivel_4 = multicamino.get("level_4")

    reporte = assemble_protocol(
        ProtocolThresholds(),
        level_0=nivel_0,
        level_1=nivel_1 or None,
        level_2=nivel_2,
        level_2_reference=referencia,
        level_3=nivel_3 or None,
        level_4=nivel_4,
    )
    registro = ExperimentLog(
        name="protocolo_agente_a",
        config={
            "seeds": list(STUDY_SEEDS),
            "fixture_seed": FIXTURE_SEED,
            "bars": DEFAULT_N_BARS,
            "snr_sweep": list(SNR_SWEEP_BETAS),
            "path_seeds": list(PATH_SEEDS),
            "thresholds": ProtocolThresholds().describe(),
        },
        provenance=json.loads(args.provenance) if args.provenance else {},
    )
    registro.add("protocol", reporte.describe())
    ruta = registro.record().save(carpeta, filename="protocol.json")
    print(reporte.render())
    print(f"\nescrito {ruta}")
    return 0 if reporte.stopped_at is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agents.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("arm", help="entrena y evalua un brazo")
    a.add_argument("--level", required=True)
    a.add_argument("--beta", type=float, default=None)
    a.add_argument("--seeds", type=int, default=10)
    a.add_argument("--timesteps", type=int, default=60_000)
    a.add_argument("--bars", type=int, default=DEFAULT_N_BARS)
    a.add_argument("--recurrent", action="store_true")
    a.add_argument("--label", default=None)
    a.add_argument("--out", default="results")
    a.set_defaults(func=cmd_arm)

    m = sub.add_parser(
        "multipath", help="un nivel sobre N caminos independientes x M semillas"
    )
    m.add_argument("--level", required=True)
    m.add_argument("--beta", type=float, default=None)
    m.add_argument("--paths", type=int, default=10)
    m.add_argument("--seeds", type=int, default=10)
    m.add_argument("--timesteps", type=int, default=60_000)
    m.add_argument("--bars", type=int, default=DEFAULT_N_BARS)
    m.add_argument("--recurrent", action="store_true")
    m.add_argument("--label", default=None)
    m.add_argument("--out", default="results")
    m.set_defaults(func=cmd_multipath)

    b = sub.add_parser("assemble", help="aplica los criterios sobre los brazos")
    b.add_argument("--out", default="results")
    b.add_argument("--provenance", default=None)
    b.set_defaults(func=cmd_assemble)

    args = parser.parse_args(argv)
    resultado: int = args.func(args)
    return resultado


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
