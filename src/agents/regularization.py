"""Segunda ronda de calibracion: escalera de regularizacion sobre 4b. ADR 0011.

Todo lo que decide el resultado esta fijado en ese ADR y no aca: los peldanos,
su orden, los caminos, el presupuesto y el criterio de 4b. Este modulo solo lo
ejecuta, y se niega a hacerlo si el ADR en disco no es el sellado.

Reusa la maquinaria del ADR 0010 (``agents.calibration``): la unidad es un
entrenamiento con su JSON atomico, reanudar es relanzar, y el veredicto es
``evaluate_level_4b`` con los umbrales de la calibracion. Lo que agrega es la
configuracion de cada peldano y el contraste **pareado contra G1**: mismos
caminos, mismas semillas y mismo presupuesto, asi que la diferencia por
(camino, semilla) cancela el ruido del camino.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agents import calibration as cal
from agents.experiment import jsonable
from agents.ppo import PPOConfig
from agents.protocol import STUDY_SEEDS, PolicyFactory, Verdict
from data.fixtures import DEFAULT_N_BARS
from envs.observation import ObservationSpec
from eval.distribution import decompose_variance

ADR_PATH = Path("docs/adr/0011-segunda-ronda-de-calibracion-regularizacion.md")
# Cambiar el ADR, aunque sea una coma, obliga a cambiar esta constante.
ADR_SHA256 = "2fa1fd425b2c68cf89c47203a3ac98ed39b0a9e1396b937dd685921843c942d9"

#: ADR 0011 §1: el presupuesto de G1, fijo para todos los peldanos.
TIMESTEPS = 120_000
#: ADR 0011 §5: el punto del ADR 0010 contra el que se parea cada peldano.
BASELINE_POINT = "G1"
#: Las metricas del criterio de 4b que se contrastan contra G1.
PAIRED_METRICS: tuple[str, ...] = ("action_std", "time_invested")


@dataclass(frozen=True)
class Rung:
    name: str
    ppo: PPOConfig
    observation: ObservationSpec

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ppo": self.ppo.describe(),
            "observation": self.observation.describe(),
        }


def _rungs() -> tuple[Rung, ...]:
    """ADR 0011 §4, la escalera elegida por Kevin. Cada peldano suma al anterior."""
    r1 = Rung(
        "R1",
        PPOConfig(total_timesteps=TIMESTEPS, learning_rate=1e-4, n_epochs=5),
        ObservationSpec(lookback=1),
    )
    r2 = replace(r1, name="R2", ppo=replace(r1.ppo, net_arch=(32, 32)))
    r3 = replace(r2, name="R3", ppo=replace(r2.ppo, gamma=0.9))
    return (r1, r2, r3)


RUNGS: tuple[Rung, ...] = _rungs()


def rung(name: str) -> Rung:
    for r in RUNGS:
        if r.name == name:
            return r
    raise cal.CalibrationError(
        f"peldano desconocido {name!r}; la escalera es {[r.name for r in RUNGS]}"
    )


def is_rung(name: str) -> bool:
    return any(r.name == name for r in RUNGS)


def rung_units(name: str) -> tuple[cal.Unit, ...]:
    rung(name)
    return tuple(
        cal.Unit(name, TIMESTEPS, camino, semilla)
        for camino in cal.CALIBRATION_PATH_SEEDS
        for semilla in STUDY_SEEDS
    )


def config_digest() -> str:
    """Hash de todo lo que define la ronda. Cambiarlo invalida el sello."""
    crudo = {
        "rungs": [r.describe() for r in RUNGS],
        "path_seeds": list(cal.CALIBRATION_PATH_SEEDS),
        "agent_seeds": list(STUDY_SEEDS),
        "n_bars": DEFAULT_N_BARS,
        "baseline": BASELINE_POINT,
        "thresholds": cal.thresholds().describe(),
    }
    return hashlib.sha256(json.dumps(crudo, sort_keys=True).encode()).hexdigest()


def verify_adr(root: Path) -> str:
    return cal.verify_adr(root, ADR_SHA256, ADR_PATH)


def train_unit(
    unit: cal.Unit, *, policy_factory: PolicyFactory | None = None
) -> dict[str, Any]:
    r = rung(unit.point)
    return cal.train_unit(
        unit, policy_factory=policy_factory, ppo=r.ppo, observation=r.observation
    )


def paired_against_baseline(
    registros: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Diferencia peldano − G1 por (camino, semilla), con su ``t`` entre caminos.

    Cada camino aporta la mediana de sus diez diferencias: las semillas de un
    camino comparten la serie y no son observaciones independientes.
    """
    base = {(b["path_seed"], b["agent_seed"]): b for b in baseline}
    salida: dict[str, Any] = {}
    for metrica in PAIRED_METRICS:
        por_camino: list[list[float | None]] = []
        for camino in cal.CALIBRATION_PATH_SEEDS:
            diferencias: list[float | None] = []
            for semilla in STUDY_SEEDS:
                propio = next(
                    r
                    for r in registros
                    if r["path_seed"] == camino and r["agent_seed"] == semilla
                )
                valor = propio["arm"]["runs"][0].get(metrica)
                ref = base[(camino, semilla)]["arm"]["runs"][0].get(metrica)
                diferencias.append(
                    None if valor is None or ref is None else valor - ref
                )
            por_camino.append(diferencias)
        salida[metrica] = decompose_variance(
            f"{metrica}_minus_{BASELINE_POINT}", por_camino
        ).describe()
    return salida


def load_baseline(baseline_dir: Path) -> list[dict[str, Any]] | None:
    """Los 50 entrenamientos de G1, o ``None`` si no estan todos."""
    try:
        return cal.load_point(baseline_dir, BASELINE_POINT)
    except cal.CalibrationError:
        return None


def decide(
    out: Path,
    baseline_dir: Path,
    *,
    run: Callable[[str], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """ADR 0011 §5: en orden, el primer peldano que pasa 4b es ``C*``.

    Ningun peldano posterior al primero que pasa se entrena ni se mira.
    """
    base = load_baseline(baseline_dir)
    peldanos = []
    elegido: str | None = None
    for r in RUNGS:
        if run is not None:
            run(r.name)
        registros = cal.load_point(out, r.name, rung_units(r.name))
        resumen = cal.summarize_point(r.name, registros)
        resumen["config"] = r.describe()
        resumen["paired_vs_baseline"] = (
            None if base is None else paired_against_baseline(registros, base)
        )
        peldanos.append(resumen)
        log(f"{r.name}: {resumen['verdict']}. {resumen['finding']}")
        if resumen["verdict"] == Verdict.PASS.value:
            elegido = r.name
            break
    return {
        "selected_rung": elegido,
        "timesteps": TIMESTEPS,
        "baseline_found": base is not None,
        "rungs": peldanos,
    }


def finalize(out: Path, resultado: dict[str, Any]) -> Path:
    """Resumen y manifest con el SHA256 de cada entrenamiento usado."""
    usados = [
        cal.unit_path(out, u)
        for p in resultado["rungs"]
        for u in rung_units(p["point"])
    ]
    manifest = {r.name: hashlib.sha256(r.read_bytes()).hexdigest() for r in usados}
    cal.write_atomic(
        out / "MANIFEST.json", {"adr_sha256": ADR_SHA256, "units": manifest}
    )
    resumen = out / "summary.json"
    cal.write_atomic(resumen, jsonable(resultado))
    return resumen


def render(resultado: dict[str, Any]) -> str:
    """Texto corto para pegar en el hilo: veredicto y contraste contra G1."""
    lineas = []
    for p in resultado["rungs"]:
        pooled = p["pooled_over_paths_and_seeds"]
        sigma = pooled["policy_sigma"]["median"]
        linea = (
            f"{p['point']}: {p['verdict']}. {p['finding']}. sigma (mediana) "
            + ("n/a" if sigma is None else f"{sigma:.3f}")
            + f"; pared por entrenamiento (mediana) {p['wall_seconds']['median']:.0f} s"
        )
        pareado = p["paired_vs_baseline"]
        if pareado is not None:
            for metrica in PAIRED_METRICS:
                d = pareado[metrica]
                # Sin dispersion entre caminos el t no esta definido.
                t = d["t_statistic"]
                linea += (
                    f"; {metrica} contra {BASELINE_POINT} {d['mean']:+.3f} "
                    "(t entre caminos " + ("n/a" if t is None else f"{t:+.2f}") + ")"
                )
        lineas.append(linea)
    if not resultado["baseline_found"]:
        lineas.append(
            f"No se encontraron los 50 entrenamientos de {BASELINE_POINT}: el "
            "contraste pareado no se calculo (pasa --baseline con su carpeta)."
        )
    elegido = resultado["selected_rung"]
    lineas.append(
        f"C* = {elegido} a {TIMESTEPS} timesteps: sigue la confirmacion del ADR 0011 "
        "seccion 5."
        if elegido
        else "Ningun peldano paso: NO se fija configuracion (ADR 0011 seccion 5)."
    )
    return "\n".join(lineas)
