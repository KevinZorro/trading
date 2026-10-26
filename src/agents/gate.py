"""Puerta del Agente B sobre fixtures, reanudable. ADR 0012 seccion 5 (N6).

Cuatro brazos de 10 caminos x 10 semillas con la configuracion del estudio
(``agents.study``: R3 a 120 000 pasos): B y A (ciego) sobre el 5a, B y A sobre el
5b. Son 400 entrenamientos. La unidad de trabajo es **uno** (brazo, camino,
semilla) con su JSON atomico, como la confirmacion del ADR 0011, y relanzar el
mismo comando salta lo hecho.

A y B corren sobre **el mismo mercado**: el mismo camino con las mismas columnas.
A no las ve (``run_arm(blind=True)``) pero el techo sigue calculandose con ellas,
asi que ``capture`` es comparable y el contraste B - A es pareado por camino.

Este modulo ejecuta y junta; los criterios son ``evaluate_level_5a`` y
``evaluate_level_5b`` (``agents.protocol``). El re-diagnostico con el grupo
``noticias`` (ADR 0012, 5) no esta aca: va en el PR siguiente.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from agents import calibration as cal
from agents.confirmation import _merge
from agents.experiment import jsonable
from agents.policy import Policy
from agents.ppo import SB3Policy
from agents.protocol import (
    STUDY_SEEDS,
    LevelResult,
    MultiPathResult,
    PolicyFactory,
    ProtocolThresholds,
    evaluate_level_5a,
    evaluate_level_5b,
    run_arm,
)
from agents.study import STUDY_OBSERVATION, STUDY_PPO
from data.fixtures import DEFAULT_N_BARS, LEVELS
from envs.observation import ObservationSpec
from eval.distribution import drift_t_statistic
from features.news import NEWS_EXOG

#: Disjuntos de ``PATH_SEEDS`` (701 a 761) y de los caminos de calibracion (809 a
#: 827) y de diagnostico (839 a 863): el nivel 5 no reutiliza ningun mercado.
GATE_PATH_SEEDS: tuple[int, ...] = (877, 881, 883, 887, 907, 911, 919, 929, 937, 941)
PREFIX = "N6-"


@dataclass(frozen=True)
class GateArm:
    label: str
    level: str
    blind: bool

    @property
    def point(self) -> str:
        return f"{PREFIX}{self.label}"


ARMS: tuple[GateArm, ...] = (
    GateArm("5a_B", "level_5a", False),
    GateArm("5a_A", "level_5a", True),
    GateArm("5b_B", "level_5b", False),
    GateArm("5b_A", "level_5b", True),
)


def arm(point: str) -> GateArm:
    for a in ARMS:
        if a.point == point:
            return a
    raise cal.CalibrationError(f"brazo de la puerta desconocido {point!r}")


def is_point(point: str) -> bool:
    return point.startswith(PREFIX)


def arm_units(point: str) -> tuple[cal.Unit, ...]:
    arm(point)
    return tuple(
        cal.Unit(point, STUDY_PPO.total_timesteps, camino, semilla)
        for camino in GATE_PATH_SEEDS
        for semilla in STUDY_SEEDS
    )


def all_units() -> tuple[cal.Unit, ...]:
    return tuple(u for a in ARMS for u in arm_units(a.point))


def config_digest() -> str:
    """Hash de todo lo que define la puerta. Cambiarlo invalida el sello."""
    crudo = {
        "ppo": STUDY_PPO.describe(),
        "observation": STUDY_OBSERVATION.describe(),
        "news_exog": [vars(c) for c in NEWS_EXOG],
        "arms": [vars(a) for a in ARMS],
        "path_seeds": list(GATE_PATH_SEEDS),
        "agent_seeds": list(STUDY_SEEDS),
        "n_bars": DEFAULT_N_BARS,
        "saved_policy_seeds": list(SAVED_POLICY_SEEDS),
        "thresholds": ProtocolThresholds().describe(),
    }
    return hashlib.sha256(json.dumps(crudo, sort_keys=True).encode()).hexdigest()


def observation_for(a: GateArm) -> ObservationSpec:
    """B ve las siete columnas; A, la observacion de siempre."""
    return STUDY_OBSERVATION if a.blind else replace(STUDY_OBSERVATION, exog=NEWS_EXOG)


#: Semillas de agente de B cuyas politicas se guardan para el re-diagnostico: con
#: 10 caminos son 20 politicas por nivel, el doble del minimo de ``summarize``.
SAVED_POLICY_SEEDS: tuple[int, ...] = STUDY_SEEDS[:2]


def policy_path(policy_dir: Path, unit: cal.Unit) -> Path:
    """Donde queda la politica de una unidad (SB3 le agrega ``.zip``)."""
    return policy_dir / unit.key


def train_unit(
    unit: cal.Unit,
    *,
    policy_factory: PolicyFactory | None = None,
    policy_dir: Path | None = None,
) -> dict[str, Any]:
    """Un (brazo, camino, semilla) con la configuracion del estudio.

    Con ``policy_dir`` guarda la politica de los brazos de B para las semillas de
    ``SAVED_POLICY_SEEDS``: el re-diagnostico (ADR 0012, 5) corre sobre **las
    politicas de la puerta**, no sobre otras reentrenadas.
    """
    a = arm(unit.point)
    fixture = LEVELS[a.level](DEFAULT_N_BARS, seed=unit.path_seed)
    train, _, _ = fixture.split()
    # perf_counter mide duracion; no consulta la hora de pared (CLAUDE.md, Clock).
    guardar = (
        policy_dir is not None and not a.blind and unit.agent_seed in SAVED_POLICY_SEEDS
    )

    def guardar_politica(politica: Policy, _semilla: int) -> None:
        if guardar and policy_dir is not None and isinstance(politica, SB3Policy):
            policy_dir.mkdir(parents=True, exist_ok=True)
            politica.save(str(policy_path(policy_dir, unit)))

    inicio = time.perf_counter()
    brazo = run_arm(
        fixture,
        label=f"{a.label}:path{unit.path_seed}",
        seeds=[unit.agent_seed],
        ppo_config=STUDY_PPO,
        policy_factory=policy_factory,
        observation=observation_for(a),
        blind=a.blind,
        allow_fewer_seeds=True,
        on_trained=guardar_politica,
    )
    return {
        "unit": unit.key,
        "point": unit.point,
        "timesteps": STUDY_PPO.total_timesteps,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        "drift_t": drift_t_statistic(np.asarray(train.signal)[1:]),
        "wall_seconds": time.perf_counter() - inicio,
        "machine": cal.machine(),
        "observation": observation_for(a).describe(),
        "policy_saved": bool(guardar),
        "arm": jsonable(brazo.describe()),
    }


def collect(out: Path) -> dict[str, MultiPathResult]:
    """Junta los entrenamientos en un ``MultiPathResult`` por brazo.

    Falla diciendo cuantos faltan si la puerta no termino.
    """
    faltan = [u.key for u in all_units() if not cal.unit_path(out, u).exists()]
    if faltan:
        raise cal.CalibrationError(
            f"faltan {len(faltan)} de {len(all_units())} entrenamientos, "
            f"p. ej. {faltan[0]}"
        )
    salida: dict[str, MultiPathResult] = {}
    for a in ARMS:
        registros = cal.load_point(out, a.point, arm_units(a.point))
        brazos, drift_t = [], []
        for camino in GATE_PATH_SEEDS:
            propios = [r for r in registros if r["path_seed"] == camino]
            brazos.append(_merge(propios, f"{a.label}:path{camino}"))
            drift_t.append(float(propios[0]["drift_t"]))
        datos = {
            "label": a.label,
            "path_seeds": list(GATE_PATH_SEEDS),
            "agent_seeds": list(STUDY_SEEDS),
            "arms": brazos,
            "drift_t_by_path": drift_t,
            "ppo": brazos[0]["ppo"],
        }
        cal.write_atomic(out / "gate" / f"multipath_{a.label}.json", datos)
        salida[a.label] = MultiPathResult.from_dict(datos)
    return salida


def assess(out: Path) -> tuple[LevelResult, LevelResult]:
    """Aplica los criterios del ADR 0012, 5 y deja ``summary.json`` y el manifest."""
    r = collect(out)
    umbrales = ProtocolThresholds()
    nivel_5a = evaluate_level_5a(r["5a_B"], r["5a_A"], umbrales)
    nivel_5b = evaluate_level_5b(r["5b_B"], r["5b_A"], umbrales)
    cal.write_atomic(
        out / "summary.json",
        jsonable(
            {
                "thresholds": umbrales.describe(),
                "levels": [
                    {
                        "level": n.level,
                        "label": n.label,
                        "verdict": str(n.verdict),
                        "statement": n.statement,
                        "finding": n.finding,
                    }
                    for n in (nivel_5a, nivel_5b)
                ],
                "blocks_real_data": str(nivel_5a.verdict) != "PASS",
            }
        ),
    )
    cal.write_atomic(
        out / "MANIFEST.json",
        {
            "units": {
                cal.unit_path(out, u).name: hashlib.sha256(
                    cal.unit_path(out, u).read_bytes()
                ).hexdigest()
                for u in all_units()
            }
        },
    )
    return nivel_5a, nivel_5b
