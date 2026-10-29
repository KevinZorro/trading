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
import functools
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.experiment import ExperimentLog, jsonable
from agents.ppo import PPOConfig
from agents.protocol import (
    STUDY_SEEDS,
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
    level_4b_detectable_drift,
)

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
    if level == "level_4b":
        return level_4b_detectable_drift(n_bars, seed=seed)
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
    nivel_3_multicamino = [
        v for k, v in _load_multipath(carpeta).items() if k.startswith("level_3")
    ]
    multicamino = _load_multipath(carpeta)
    nivel_4a = multicamino.get("level_4")
    nivel_4b = multicamino.get("level_4b")

    reporte = assemble_protocol(
        ProtocolThresholds(),
        level_0=nivel_0,
        level_1=nivel_1 or None,
        level_2=nivel_2,
        level_2_reference=referencia,
        level_3=nivel_3 or None,
        level_3_multipath=nivel_3_multicamino or None,
        level_4a=nivel_4a,
        level_4b=nivel_4b,
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


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Tarea C3: la corrida sellada del ADR 0006. Toca el test una sola vez."""
    import subprocess

    from agents.ppo import train_ppo
    from agents.walkforward import WalkForwardConfig, run_walkforward

    config = WalkForwardConfig()
    ppo = config.ppo_config()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    reporte = run_walkforward(
        Path.cwd(),
        Path(args.out),
        lambda make_env, seed: train_ppo(make_env, ppo, seed=seed),
        config=config,
        run_commit=commit,
        resume=args.resume,
    )
    print(json.dumps(jsonable(reporte["main"]), indent=2))
    print(json.dumps(jsonable(reporte["prediction_adr_0004"]["verdict"])))
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    """ADR 0012: corrida sellada A vs B vs placebo. Toca el test una sola vez."""
    import subprocess

    from agents.ppo import train_ppo
    from agents.study_run import StudyRunConfig, run_study

    config = StudyRunConfig()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    entrenadas = run_study(
        Path.cwd(),
        Path(args.out),
        lambda _observation: lambda make_env, seed: train_ppo(
            make_env, config.ppo, seed=seed
        ),
        config=config,
        run_commit=commit,
        live_latency_p99_min=args.live_latency_p99_min,
        resume=args.resume,
    )
    print(f"entrenamientos nuevos: {entrenadas}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """ADR 0010: calibra el presupuesto de entrenamiento sobre 4b."""
    from agents import calibration as cal
    from agents import regularization as reg

    por_defecto = {
        "diagnose": "results/diagnostico",
        "ladder": "results/regularizacion",
        "ladder-report": "results/regularizacion",
        "confirm": "results/confirmacion",
        "confirm-report": "results/confirmacion",
        "gate": "results/puerta",
        "gate-report": "results/puerta",
        "gate-diagnose": "results/puerta",
    }.get(args.action, "results/calibracion")
    out = Path(args.out or por_defecto)
    if args.action in ("worker", "diagnose", "confirm", "confirm-report"):
        from agents import confirmation as conf
        from agents import diagnosis as dg
    if args.action in ("worker", "gate", "gate-report", "gate-diagnose"):
        from agents import gate

    if args.action == "worker":
        # Proceso hijo de ``run``: ya paso por la verificacion del sello.
        entrenar: Callable[[cal.Unit], dict[str, Any]] | None
        if args.point == dg.DIAGNOSIS_POINT:
            todas, entrenar = dg.diagnosis_units(), dg.train_unit
        elif args.point == conf.REDIAGNOSIS_POINT:
            todas = conf.rediagnosis_units()
            entrenar = conf.train_rediagnosis_unit
        elif gate.is_point(args.point):
            todas = gate.arm_units(args.point)
            entrenar = functools.partial(gate.train_unit, policy_dir=out / "policies")
        elif conf.is_point(args.point):
            todas, entrenar = conf.arm_units(args.point), conf.train_unit
        elif reg.is_rung(args.point):
            todas, entrenar = reg.rung_units(args.point), reg.train_unit
        else:
            todas, entrenar = cal.point_units(args.point), None
        unidades = cal.shard(todas, *(int(x) for x in args.shard.split("/")))
        cal.run_units(out, unidades, train=entrenar)
        return 0
    if args.action == "diagnose":
        # Diagnostico de por que 4b no pasa a ningun presupuesto. Sin sello: no
        # hay ADR todavia; la regla de lectura esta en el docstring de diagnosis.
        unidades = dg.diagnosis_units()
        cal.run_point(out, dg.DIAGNOSIS_POINT, args.workers, unidades)
        registros = [
            json.loads(cal.unit_path(out, u).read_text(encoding="utf-8"))
            for u in unidades
        ]
        resumen = dg.summarize_diagnosis(registros)
        cal.write_atomic(out / "summary.json", jsonable(resumen))
        print(json.dumps(jsonable(resumen), indent=2))
        return 0
    if args.action == "bench":
        unidad = cal.point_units("G1")[0]
        registro = cal.train_unit(unidad, timesteps=args.timesteps)
        por_paso = registro["wall_seconds"] / args.timesteps
        print(f"{args.timesteps} pasos en {registro['wall_seconds']:.0f} s")
        for nombre, pasos in cal.GRID:
            horas = cal.projected_hours(
                por_paso, pasos, len(cal.point_units(nombre)), args.workers
            )
            print(
                f"  {nombre} ({pasos} pasos): {horas:.1f} h con {args.workers} procesos"
            )
        for nombre, pasos in cal.GRID:
            horas = cal.projected_hours(
                por_paso, pasos, cal.CONFIRMATION_TRAININGS, args.workers
            )
            print(
                f"  confirmacion con {nombre}: {horas:.0f} h, {args.workers} procesos"
            )
        return 0
    if args.action in ("ladder", "ladder-report"):
        # ADR 0011: la escalera de regularizacion, sellada con su propio ADR.
        reg_adr = reg.verify_adr(Path.cwd())
        cal.acquire_seal(out, reg_adr, reg.config_digest())
        base = Path(args.baseline)
        correr = None
        if args.action == "ladder":

            def correr(nombre: str) -> None:
                cal.run_point(out, nombre, args.workers, reg.rung_units(nombre))

        escalera = reg.decide(out, base, run=correr)
        ruta = reg.finalize(out, escalera)
        print(reg.render(escalera))
        print(f"\nescrito {ruta}")
        return 0
    if args.action in ("gate", "gate-report", "gate-diagnose"):
        # ADR 0012 seccion 5: la puerta del Agente B, reanudable. Sello propio: el
        # de N8 es otro y lo escribe N8.
        adr12 = next(Path.cwd().glob("docs/adr/0012-*.md"))
        crudo = adr12.read_bytes().replace(b"\r\n", b"\n")
        cal.acquire_seal(out, hashlib.sha256(crudo).hexdigest(), gate.config_digest())
        if args.action == "gate":
            for b_puerta in gate.ARMS:
                cal.run_point(
                    out, b_puerta.point, args.workers, gate.arm_units(b_puerta.point)
                )
        if args.action == "gate-diagnose":
            # Re-diagnostico con el grupo noticias, sobre las politicas de B que
            # guardo la puerta (ADR 0012 seccion 5).
            from agents import gate_diagnosis

            for nombre, r in gate_diagnosis.run(out).items():
                print(f"{nombre}: {r['reading']}")
            return 0
        for nivel in gate.assess(out):
            print(f"[{nivel.verdict}] {nivel.label}\n  {nivel.finding}")
        return 0
    if args.action in ("confirm", "confirm-report"):
        # ADR 0011 seccion 5: protocolo completo con C*, reanudable.
        cal.acquire_seal(out, reg.verify_adr(Path.cwd()), conf.config_digest())
        rediagnostico = conf.rediagnosis_units()
        if args.action == "confirm":
            for brazo in conf.ARMS:
                cal.run_point(
                    out, brazo.point, args.workers, conf.arm_units(brazo.point)
                )
            cal.run_point(out, conf.REDIAGNOSIS_POINT, args.workers, rediagnostico)
        destino = conf.collect(out)
        conf.manifest(out)
        if all(cal.unit_path(out, u).exists() for u in rediagnostico):
            registros = cal.load_point(out, conf.REDIAGNOSIS_POINT, rediagnostico)
            resumen_dg = dg.summarize_diagnosis(registros)
            cal.write_atomic(out / "rediagnosis.json", jsonable(resumen_dg))
            print(f"Re-diagnostico con {conf.CONFIRMED_RUNG}: {resumen_dg['reading']}")
        procedencia = {
            "adr": str(reg.ADR_PATH),
            "adr_sha256": reg.ADR_SHA256,
            "rung": conf.CONFIRMED_RUNG,
        }
        return cmd_assemble(
            argparse.Namespace(out=str(destino), provenance=json.dumps(procedencia))
        )
    adr = cal.verify_adr(Path.cwd())
    cal.acquire_seal(out, adr)
    if args.action == "run":
        resultado = cal.decide(
            out, run=lambda punto: cal.run_point(out, punto, args.workers)
        )
    else:
        resultado = cal.decide(out)
    ruta = cal.finalize(out, resultado)
    print(cal.render(resultado))
    print(f"\nescrito {ruta}")
    return 0


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

    w = sub.add_parser(
        "walkforward", help="Tarea C3: walk-forward sellado sobre BTCUSDT diario"
    )
    w.add_argument("--out", default="results/c3")
    w.add_argument(
        "--resume",
        action="store_true",
        help="continua una corrida interrumpida con la misma configuracion sellada",
    )
    w.set_defaults(func=cmd_walkforward)

    st = sub.add_parser("study", help="ADR 0012: corrida sellada A vs B vs placebo")
    st.add_argument("action", choices=["run"])
    st.add_argument("--out", default="results/estudio")
    st.add_argument(
        "--live-latency-p99-min",
        type=float,
        required=True,
        help="p99 del retraso de los lotes en vivo, medido con los datos de L1",
    )
    st.add_argument(
        "--resume",
        action="store_true",
        help="continua una corrida interrumpida con el mismo sello",
    )
    st.set_defaults(func=cmd_study)

    c = sub.add_parser(
        "calibrate", help="ADR 0010: presupuesto de entrenamiento sobre 4b"
    )
    c.add_argument(
        "action",
        choices=[
            "bench",
            "run",
            "report",
            "worker",
            "diagnose",
            "ladder",
            "ladder-report",
            "confirm",
            "confirm-report",
            "gate",
            "gate-report",
            "gate-diagnose",
        ],
    )
    c.add_argument("--out", default=None)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--timesteps", type=int, default=20_000, help="solo para bench")
    c.add_argument("--point", default=None, help="interno, de worker")
    c.add_argument("--shard", default="0/1", help="interno, de worker")
    c.add_argument(
        "--baseline",
        default="results/calibracion",
        help="carpeta con los entrenamientos de G1, para el contraste pareado",
    )
    c.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    resultado: int = args.func(args)
    return resultado


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
