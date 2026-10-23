"""Confirmacion de ``C*`` con el protocolo completo, reanudable. ADR 0011 §5.

La escalera (``agents.regularization``) eligio R3 sobre los caminos de
calibracion. Esto re-corre **el protocolo entero** con esa configuracion sobre
los caminos y semillas registrados: niveles 0 a 3, 4a, 4b y el par. No solo 4b,
porque mas regularizacion puede apagar la senal en los niveles 1 y 2, y gamma
0.9 pone en riesgo justamente el nivel 2.

Los subcomandos ``arm`` y ``multipath`` no se reanudan por dentro: un corte a la
hora diez pierde el brazo entero. Aca la unidad es **un entrenamiento** (brazo,
camino, semilla) con su JSON atomico, igual que en la calibracion, y relanzar
el mismo comando salta lo hecho. Al final los entrenamientos se juntan en los
``arm_*.json`` y ``multipath_*.json`` de siempre y los juzga ``cli assemble``,
sin una segunda implementacion de los criterios.

Tambien re-corre el diagnostico de #57 sobre politicas con ``C*`` (ADR 0011 §6),
para ver que grupo de la observacion queda moviendo la accion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from agents import calibration as cal
from agents import diagnosis as dg
from agents import regularization as reg
from agents.experiment import jsonable
from agents.protocol import STUDY_SEEDS, PolicyFactory, run_arm
from data.fixtures import DEFAULT_N_BARS, SNR_SWEEP_BETAS
from eval.distribution import drift_t_statistic

#: El peldano que paso la escalera. Corrida de Kevin del 2026-10-23, resumen en
#: ``results/regularizacion/summary.json``. Cambiarlo invalida el sello.
CONFIRMED_RUNG = "R3"
#: Prefijo de las unidades de la confirmacion, para que el worker las reconozca.
PREFIX = "C-"
#: Prefijo del re-diagnostico sobre ``C*``.
REDIAGNOSIS_POINT = "D-R3"


@dataclass(frozen=True)
class ConfirmArm:
    """Un brazo del protocolo. ``multipath`` usa ``PATH_SEEDS``; si no, un camino."""

    label: str
    level: str
    beta: float | None
    recurrent: bool
    multipath: bool

    @property
    def point(self) -> str:
        return f"{PREFIX}{self.label}"


def _arms() -> tuple[ConfirmArm, ...]:
    """Los brazos de la Etapa 3, con las etiquetas que ``cli assemble`` espera."""
    unicos = [ConfirmArm("level_0", "level_0", None, False, False)]
    unicos += [
        ConfirmArm(f"level_1_beta{b}", "level_1", b, False, False)
        for b in SNR_SWEEP_BETAS
    ]
    unicos += [
        ConfirmArm("level_2_beta-0.3", "level_2", -0.3, False, False),
        ConfirmArm("level_3_beta-0.3", "level_3", -0.3, False, False),
        ConfirmArm("level_3_beta-0.3_lstm", "level_3", -0.3, True, False),
    ]
    caminos = [
        ConfirmArm("level_3_mlp", "level_3", -0.3, False, True),
        ConfirmArm("level_3_lstm", "level_3", -0.3, True, True),
        ConfirmArm("level_4", "level_4", None, False, True),
        ConfirmArm("level_4b", "level_4b", None, False, True),
    ]
    return (*unicos, *caminos)


ARMS: tuple[ConfirmArm, ...] = _arms()


def arm(point: str) -> ConfirmArm:
    for a in ARMS:
        if a.point == point:
            return a
    raise cal.CalibrationError(f"brazo de confirmacion desconocido {point!r}")


def is_point(point: str) -> bool:
    return point.startswith(PREFIX)


def _path_seeds(a: ConfirmArm) -> tuple[int, ...]:
    from agents.cli import FIXTURE_SEED, PATH_SEEDS

    return PATH_SEEDS if a.multipath else (FIXTURE_SEED,)


def arm_units(point: str) -> tuple[cal.Unit, ...]:
    a = arm(point)
    return tuple(
        cal.Unit(point, reg.TIMESTEPS, camino, semilla)
        for camino in _path_seeds(a)
        for semilla in STUDY_SEEDS
    )


def all_units() -> tuple[cal.Unit, ...]:
    return tuple(u for a in ARMS for u in arm_units(a.point))


def rediagnosis_units() -> tuple[cal.Unit, ...]:
    return tuple(
        cal.Unit(REDIAGNOSIS_POINT, reg.TIMESTEPS, camino, semilla)
        for camino in dg.DIAGNOSIS_PATH_SEEDS
        for semilla in dg.DIAGNOSIS_AGENT_SEEDS
    )


def config_digest() -> str:
    """Hash de todo lo que define la confirmacion. Cambiarlo invalida el sello."""
    from agents.cli import FIXTURE_SEED, PATH_SEEDS

    crudo = {
        "rung": reg.rung(CONFIRMED_RUNG).describe(),
        "arms": [vars(a) for a in ARMS],
        "path_seeds": list(PATH_SEEDS),
        "fixture_seed": FIXTURE_SEED,
        "agent_seeds": list(STUDY_SEEDS),
        "n_bars": DEFAULT_N_BARS,
    }
    return hashlib.sha256(json.dumps(crudo, sort_keys=True).encode()).hexdigest()


def train_unit(
    unit: cal.Unit, *, policy_factory: PolicyFactory | None = None
) -> dict[str, Any]:
    """Un (brazo, camino, semilla) con la configuracion de ``C*``."""
    import time

    from agents.cli import build_fixture_with_seed

    a = arm(unit.point)
    r = reg.rung(CONFIRMED_RUNG)
    fixture = build_fixture_with_seed(
        a.level, beta=a.beta, n_bars=DEFAULT_N_BARS, seed=unit.path_seed
    )
    train, _, _ = fixture.split()
    config = replace(r.ppo, recurrent=a.recurrent)
    etiqueta = f"{a.label}:path{unit.path_seed}" if a.multipath else a.label
    # perf_counter mide duracion; no consulta la hora de pared (CLAUDE.md, Clock).
    inicio = time.perf_counter()
    brazo = run_arm(
        fixture,
        label=etiqueta,
        seeds=[unit.agent_seed],
        ppo_config=config,
        policy_factory=policy_factory,
        observation=r.observation,
        allow_fewer_seeds=True,
    )
    return {
        "unit": unit.key,
        "point": unit.point,
        "timesteps": config.total_timesteps,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        # El t del drift solo lo usan los multicamino (ADR 0004).
        "drift_t": (
            drift_t_statistic(np.asarray(train.signal)[1:]) if a.multipath else None
        ),
        "wall_seconds": time.perf_counter() - inicio,
        "machine": cal.machine(),
        "observation": r.observation.describe(),
        "arm": jsonable(brazo.describe()),
    }


def train_rediagnosis_unit(unit: cal.Unit) -> dict[str, Any]:
    r = reg.rung(CONFIRMED_RUNG)
    return dg.train_unit(unit, ppo=r.ppo, observation=r.observation)


def _merge(registros: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Un brazo con las diez semillas, en el orden de ``STUDY_SEEDS``."""
    propios = {r["agent_seed"]: r for r in registros}
    ordenados = [propios[s] for s in STUDY_SEEDS]
    brazo = dict(ordenados[0]["arm"])
    brazo["label"] = label
    brazo["runs"] = [r["arm"]["runs"][0] for r in ordenados]
    return brazo


def collect(out: Path) -> Path:
    """Junta los entrenamientos en los JSON que lee ``cli assemble``.

    Falla diciendo cuantos faltan si la confirmacion no termino.
    """
    destino = out / "protocolo"
    faltan = [u.key for u in all_units() if not cal.unit_path(out, u).exists()]
    if faltan:
        raise cal.CalibrationError(
            f"faltan {len(faltan)} de {len(all_units())} entrenamientos, "
            f"p. ej. {faltan[0]}"
        )
    for a in ARMS:
        registros = cal.load_point(out, a.point, arm_units(a.point))
        if not a.multipath:
            cal.write_atomic(
                destino / f"arm_{a.label}.json", _merge(registros, a.label)
            )
            continue
        brazos, drift_t = [], []
        for camino in _path_seeds(a):
            propios = [r for r in registros if r["path_seed"] == camino]
            brazos.append(_merge(propios, f"{a.label}:path{camino}"))
            drift_t.append(float(propios[0]["drift_t"]))
        multicamino = {
            "label": a.label,
            "path_seeds": list(_path_seeds(a)),
            "agent_seeds": list(STUDY_SEEDS),
            "arms": brazos,
            "drift_t_by_path": drift_t,
            "ppo": brazos[0]["ppo"],
        }
        cal.write_atomic(destino / f"multipath_{a.label}.json", multicamino)
    return destino


def manifest(out: Path) -> Path:
    """SHA256 de cada entrenamiento, para verificar el resumen contra ellos."""
    unidades = (*all_units(), *rediagnosis_units())
    datos = {
        "adr_sha256": reg.ADR_SHA256,
        "rung": CONFIRMED_RUNG,
        "units": {
            cal.unit_path(out, u).name: hashlib.sha256(
                cal.unit_path(out, u).read_bytes()
            ).hexdigest()
            for u in unidades
            if cal.unit_path(out, u).exists()
        },
    }
    ruta = out / "MANIFEST.json"
    cal.write_atomic(ruta, datos)
    return ruta
