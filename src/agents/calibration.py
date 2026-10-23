"""Calibracion del presupuesto de entrenamiento sobre 4b. Implementa el ADR 0010.

Todo lo que decide el resultado esta fijado en ese ADR y no aca: la grilla, los
caminos, la regla de parada y el criterio de 4b. Este modulo solo lo ejecuta, y
se niega a hacerlo si el ADR en disco no es el sellado (``ADR_SHA256``).

La unidad de trabajo es **un entrenamiento** (punto, camino, semilla) y deja su
propio JSON, escrito de forma atomica. Reanudar es volver a lanzar el mismo
comando: lo que ya existe se salta, y un corte a mitad de un entrenamiento solo
pierde ese entrenamiento. Corre en el PC de Kevin (decision D6 del plan).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from agents.experiment import jsonable
from agents.ppo import PPOConfig
from agents.protocol import (
    STUDY_SEEDS,
    LevelResult,
    MultiPathResult,
    PolicyFactory,
    ProtocolThresholds,
    Verdict,
    evaluate_level_4b,
    run_arm,
)
from data.fixtures import DEFAULT_N_BARS, level_4b_detectable_drift
from eval.distribution import drift_t_statistic, summarize

ADR_PATH = Path("docs/adr/0010-presupuesto-de-entrenamiento-sobre-4b.md")
# Cambiar el ADR, aunque sea una coma, obliga a cambiar esta constante.
ADR_SHA256 = "06b7fbeb5145f88375c359f7f37a520ae4f08a0b1fcac34e22e32385accdde1e"

#: ADR 0010 §2. Duplicar en cada paso; 60 000 no esta porque ya se sabe que falla.
GRID: tuple[tuple[str, int], ...] = (
    ("G1", 120_000),
    ("G2", 240_000),
    ("G3", 480_000),
    ("G4", 960_000),
)
#: ADR 0010 §3. Disjuntos de ``PATH_SEEDS`` (701 a 761).
CALIBRATION_PATH_SEEDS: tuple[int, ...] = (809, 811, 821, 823, 827)
#: Entrenamientos de la confirmacion (ADR 0010 §6): 8 brazos de un camino x 10
#: semillas mas 4 multicamino de 10 x 10.
CONFIRMATION_TRAININGS = 480


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Unit:
    point: str
    timesteps: int
    path_seed: int
    agent_seed: int

    @property
    def key(self) -> str:
        return f"{self.point}_p{self.path_seed}_s{self.agent_seed}"


def point_units(point: str) -> tuple[Unit, ...]:
    pasos = dict(GRID)
    if point not in pasos:
        raise CalibrationError(
            f"punto desconocido {point!r}; la grilla es {list(pasos)}"
        )
    return tuple(
        Unit(point, pasos[point], camino, semilla)
        for camino in CALIBRATION_PATH_SEEDS
        for semilla in STUDY_SEEDS
    )


def thresholds() -> ProtocolThresholds:
    """Los umbrales de ``main`` con **un** cambio, local a la calibracion.

    El ADR juzga sobre 5 caminos y ``evaluate_level_4b`` exige 10. El default del
    protocolo (la confirmacion) no se toca: ``dataclasses.replace`` devuelve una
    copia y un test fija que el protocolo sigue exigiendo 10.
    """
    return replace(ProtocolThresholds(), level_4_min_paths=len(CALIBRATION_PATH_SEEDS))


def config_digest() -> str:
    """Hash de todo lo que define la calibracion. Cambiarlo invalida el sello."""
    base = PPOConfig(total_timesteps=1).describe()
    base.pop("total_timesteps")
    crudo = {
        "grid": [list(g) for g in GRID],
        "path_seeds": list(CALIBRATION_PATH_SEEDS),
        "agent_seeds": list(STUDY_SEEDS),
        "n_bars": DEFAULT_N_BARS,
        "ppo": base,
        "thresholds": thresholds().describe(),
    }
    return hashlib.sha256(json.dumps(crudo, sort_keys=True).encode()).hexdigest()


def verify_adr(root: Path, expected: str = ADR_SHA256) -> str:
    # El sello es del texto con \n: en Windows git puede entregar \r\n (autocrlf)
    # y el mismo ADR tendria otro SHA256. Se normaliza antes de comparar.
    crudo = (root / ADR_PATH).read_bytes().replace(b"\r\n", b"\n")
    real = hashlib.sha256(crudo).hexdigest()
    if real != expected:
        raise CalibrationError(
            f"el ADR 0010 en disco tiene SHA256 {real} y el sellado es {expected}: "
            "el criterio cambio despues de sellarse"
        )
    return real


def acquire_seal(out: Path, adr_sha: str) -> None:
    """Crea el sello o confirma que es el mismo. Reanudar con otra config falla."""
    payload = {"adr_sha256": adr_sha, "config_digest": config_digest()}
    ruta = out / "SEAL.json"
    out.mkdir(parents=True, exist_ok=True)
    try:
        with ruta.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return
    except FileExistsError:
        existente = json.loads(ruta.read_text(encoding="utf-8"))
    if existente != payload:
        raise CalibrationError(
            f"{ruta} es de otra configuracion: {existente} != {payload}. "
            "Mezclar entrenamientos de dos configuraciones invalidaria la grilla"
        )


def unit_path(out: Path, unit: Unit) -> Path:
    return out / "units" / f"{unit.key}.json"


def machine() -> dict[str, Any]:
    versiones: dict[str, str | None] = {}
    for paquete in ("torch", "stable-baselines3"):
        try:
            versiones[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:  # pragma: no cover - depende del entorno
            versiones[paquete] = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        **versiones,
    }


def train_unit(
    unit: Unit,
    *,
    timesteps: int | None = None,
    policy_factory: PolicyFactory | None = None,
) -> dict[str, Any]:
    """Entrena y evalua un (camino, semilla). ``timesteps`` solo lo usa ``bench``."""
    fixture = level_4b_detectable_drift(DEFAULT_N_BARS, seed=unit.path_seed)
    train, _, _ = fixture.split()
    config = PPOConfig(total_timesteps=timesteps or unit.timesteps)
    # perf_counter mide duracion; no consulta la hora de pared (CLAUDE.md, Clock).
    inicio = time.perf_counter()
    brazo = run_arm(
        fixture,
        label=f"level_4b:path{unit.path_seed}",
        seeds=[unit.agent_seed],
        ppo_config=config,
        policy_factory=policy_factory,
        allow_fewer_seeds=True,
    )
    return {
        "unit": unit.key,
        "point": unit.point,
        "timesteps": config.total_timesteps,
        "path_seed": unit.path_seed,
        "agent_seed": unit.agent_seed,
        "drift_t": drift_t_statistic(np.asarray(train.signal)[1:]),
        "wall_seconds": time.perf_counter() - inicio,
        "machine": machine(),
        "arm": jsonable(brazo.describe()),
    }


def write_atomic(path: Path, datos: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(".tmp")
    with temporal.open("w", encoding="utf-8") as handle:
        json.dump(datos, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporal, path)


def run_units(
    out: Path,
    units: Sequence[Unit],
    *,
    train: Callable[[Unit], dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Entrena lo que falta. Devuelve cuantos entrenamientos hizo."""
    entrenar = train or train_unit
    hechos = 0
    for unit in units:
        ruta = unit_path(out, unit)
        if ruta.exists():
            continue
        registro = entrenar(unit)
        write_atomic(ruta, registro)
        hechos += 1
        log(f"[{unit.point}] {unit.key} listo en {registro['wall_seconds']:.0f} s")
    return hechos


def shard(units: Sequence[Unit], index: int, total: int) -> tuple[Unit, ...]:
    """Reparto fijo por posicion: cada proceso toma las suyas sin coordinarse."""
    if not 0 <= index < total:
        raise CalibrationError(f"shard {index}/{total} fuera de rango")
    return tuple(u for i, u in enumerate(units) if i % total == index)


def run_point(
    out: Path, point: str, workers: int, units: Sequence[Unit] | None = None
) -> None:
    """Un punto de la grilla en ``workers`` procesos, de un solo hilo cada uno."""
    todas = units if units is not None else point_units(point)
    pendientes = [u for u in todas if not unit_path(out, u).exists()]
    if not pendientes:
        return
    entorno = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    procesos = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agents.cli",
                "calibrate",
                "worker",
                "--out",
                str(out),
                "--point",
                point,
                "--shard",
                f"{k}/{workers}",
            ],
            env=entorno,
        )
        for k in range(workers)
    ]
    fallos = [p.args for p in procesos if p.wait() != 0]
    if fallos:
        raise CalibrationError(
            f"fallaron {len(fallos)} de {workers} procesos en {point}. Lo ya "
            "entrenado queda guardado: relanzar el mismo comando continua"
        )


def load_point(out: Path, point: str) -> list[dict[str, Any]]:
    registros, faltan = [], []
    for unit in point_units(point):
        ruta = unit_path(out, unit)
        if ruta.exists():
            registros.append(json.loads(ruta.read_text(encoding="utf-8")))
        else:
            faltan.append(unit.key)
    if faltan:
        raise CalibrationError(
            f"{point}: faltan {len(faltan)} entrenamientos, p. ej. {faltan[0]}"
        )
    return registros


def judge_point(
    registros: Sequence[dict[str, Any]],
) -> tuple[LevelResult, MultiPathResult]:
    """Aplica el criterio de 4b del ADR 0005 sobre los 5 caminos de calibracion."""
    brazos: list[dict[str, Any]] = []
    drift_t: list[float] = []
    for camino in CALIBRATION_PATH_SEEDS:
        propios = {r["agent_seed"]: r for r in registros if r["path_seed"] == camino}
        ordenados = [propios[s] for s in STUDY_SEEDS]
        brazo = dict(ordenados[0]["arm"])
        brazo["runs"] = [r["arm"]["runs"][0] for r in ordenados]
        brazos.append(brazo)
        drift_t.append(float(ordenados[0]["drift_t"]))
    resultado = MultiPathResult.from_dict(
        {
            "label": "level_4b",
            "path_seeds": list(CALIBRATION_PATH_SEEDS),
            "agent_seeds": list(STUDY_SEEDS),
            "arms": brazos,
            "drift_t_by_path": drift_t,
            "ppo": registros[0]["arm"]["ppo"],
        }
    )
    return evaluate_level_4b(resultado, thresholds()), resultado


def summarize_point(point: str, registros: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Veredicto y lo que el ADR 0010 §5 pide reportar de un punto."""
    nivel, multicamino = judge_point(registros)
    corridas = [r["arm"]["runs"][0] for r in registros]
    etiquetas = list(range(len(corridas)))

    def dist(metrica: str, valores: list[float | None]) -> dict[str, object]:
        return summarize(metrica, etiquetas, valores).describe()

    return {
        "point": point,
        "timesteps": dict(GRID)[point],
        "verdict": nivel.verdict.value,
        "finding": nivel.finding,
        "decompositions": {
            k: v.describe() for k, v in multicamino.decompositions.items()
        },
        "pooled_over_paths_and_seeds": {
            m: dist(m, [c.get(m) for c in corridas])
            for m in (
                "time_invested",
                "action_std",
                "policy_sigma",
                "excess_log_growth_vs_always_long",
            )
        },
        "wall_seconds": dist("wall_seconds", [r["wall_seconds"] for r in registros]),
        "machine": registros[0]["machine"],
    }


def decide(
    out: Path,
    *,
    run: Callable[[str], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """ADR 0010 §4: en orden ascendente, el primer punto que pasa es ``T*``.

    ``run`` entrena el punto antes de juzgarlo; sin el, solo se juzga lo guardado.
    Ningun punto posterior al primero que pasa se entrena ni se mira.
    """
    puntos = []
    elegido: int | None = None
    for nombre, pasos in GRID:
        if run is not None:
            run(nombre)
        resumen = summarize_point(nombre, load_point(out, nombre))
        puntos.append(resumen)
        log(f"{nombre} ({pasos} pasos): {resumen['verdict']}. {resumen['finding']}")
        if resumen["verdict"] == Verdict.PASS.value:
            elegido = pasos
            break
    return {"selected_timesteps": elegido, "points": puntos}


def finalize(out: Path, resultado: dict[str, Any]) -> Path:
    """Escribe el resumen y el manifest con el SHA256 de cada entrenamiento.

    Los JSON de ``units/`` van al Release; al repositorio solo estos dos
    (regla de tamano de PR de CLAUDE.md).
    """
    usados = [
        unit_path(out, u) for p in resultado["points"] for u in point_units(p["point"])
    ]
    manifest = {
        ruta.name: hashlib.sha256(ruta.read_bytes()).hexdigest() for ruta in usados
    }
    write_atomic(out / "MANIFEST.json", {"adr_sha256": ADR_SHA256, "units": manifest})
    resumen = out / "summary.json"
    write_atomic(resumen, jsonable(resultado))
    return resumen


def projected_hours(
    seconds_per_step: float, timesteps: int, trainings: int, workers: int
) -> float:
    """Horas de pared si cada entrenamiento cuesta ``seconds_per_step`` por paso."""
    return seconds_per_step * timesteps * trainings / workers / 3600.0


def render(resultado: dict[str, Any]) -> str:
    """Texto corto para pegar en el hilo: veredicto por punto y sigma."""
    lineas = []
    for p in resultado["points"]:
        pooled = p["pooled_over_paths_and_seeds"]
        sigma = pooled["policy_sigma"]["median"]
        lineas.append(
            f"{p['point']} {p['timesteps']} pasos: {p['verdict']}. {p['finding']}. "
            f"sigma de la politica (mediana) "
            + ("n/a" if sigma is None else f"{sigma:.3f}")
            + f"; pared por entrenamiento (mediana) {p['wall_seconds']['median']:.0f} s"
        )
    elegido = resultado["selected_timesteps"]
    lineas.append(
        f"T* = {elegido} timesteps: sigue la confirmacion del ADR 0010 seccion 4."
        if elegido
        else "Ningun punto paso: NO se fija presupuesto. Un ADR nuevo decide la "
        "segunda ronda antes de correrla (ADR 0010 seccion 4)."
    )
    return "\n".join(lineas)
