"""Corrida sellada A vs B vs placebo (ADR 0012): unidades, sello y reanudacion.

Sin torch: la politica se inyecta. Lee los datasets versionados (binance y gdelt),
que se verifican por SHA256; no tocan red.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import agents.study_run as sr
from agents.policy import ConstantWeightPolicy
from agents.study_run import (
    ARMS,
    StudyRunConfig,
    Unit,
    run_study,
    units,
    verify_adr,
)
from agents.walkforward import SealError, WalkForwardConfig

RAIZ = Path(__file__).resolve().parents[2]
UNA_SEMILLA = StudyRunConfig(walk=replace(StudyRunConfig().walk, seeds=(1,)))


def fabrica(vistas: list[int]) -> Any:
    """Anota cuantas columnas exogenas tiene la observacion de cada brazo."""

    def por_brazo(observation: Any) -> Any:
        vistas.append(len(observation.exog))
        return lambda _make_env, _seed: ConstantWeightPolicy(1.0)

    return por_brazo


def test_el_adr_en_disco_es_el_sellado() -> None:
    real = hashlib.sha256((RAIZ / sr.ADR_PATH).read_bytes()).hexdigest()
    assert real == sr.ADR_SHA256
    assert verify_adr(RAIZ) == real


def test_un_adr_distinto_aborta() -> None:
    with pytest.raises(SealError, match="cambio despues de sellarse"):
        verify_adr(RAIZ, expected="0" * 64)


def test_330_unidades_fold_por_fold_con_los_tres_brazos() -> None:
    todas = units(StudyRunConfig(), 3328)
    assert len(todas) == 3 * 11 * 10
    assert todas[0] == Unit("A", 0, StudyRunConfig().walk.seeds[0])
    # Un fold termina (sus tres brazos) antes de que empiece el siguiente.
    assert [u.fold for u in todas] == sorted(u.fold for u in todas)
    assert {u.arm for u in todas[:30]} == {"A", "B", "placebo"}


def test_la_configuracion_es_la_del_estudio_y_entra_al_digest() -> None:
    config = StudyRunConfig()
    assert config.ppo.total_timesteps == 120_000
    assert config.observation.lookback == 1
    assert config.news_latency_min == 60
    assert config.placebo_lag_days == 364
    assert "ppo" in config.describe()
    assert config.digest() == StudyRunConfig().digest()
    otra = replace(config, news_latency_min=30)
    assert otra.digest() != config.digest()


def test_la_latencia_en_vivo_mayor_que_la_usada_aborta_sin_sellar(
    tmp_path: Path,
) -> None:
    with pytest.raises(SealError, match="supera la usada"):
        run_study(
            RAIZ,
            tmp_path,
            fabrica([]),
            config=UNA_SEMILLA,
            run_commit="c",
            live_latency_p99_min=61.0,
        )
    assert not (tmp_path / "SEAL.json").exists()


def test_corrida_sellada_persistida_y_reanudable(tmp_path: Path) -> None:
    vistas: list[int] = []
    n = run_study(
        RAIZ,
        tmp_path,
        fabrica(vistas),
        config=UNA_SEMILLA,
        run_commit="abc",
        live_latency_p99_min=5.7,
    )
    assert n == 33  # 3 brazos x 11 folds x 1 semilla
    assert vistas == [0, 7, 7]  # A no ve noticias; B y placebo, las siete
    filas = [
        json.loads(x) for x in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert len(filas) == 33
    assert [f["arm"] for f in filas[:3]] == list(ARMS)
    assert {f["fold"] for f in filas} == set(range(11))
    sello = json.loads((tmp_path / "SEAL.json").read_text())
    assert sello["adr_sha256"] == sr.ADR_SHA256
    assert sello["commit"] == "abc"
    assert sello["latency_min"] == 60
    assert sello["live_latency_p99_min"] == 5.7
    assert sello["dataset_sha256"].startswith("0940f089")
    assert len(sello["finbert_files"]) == 5

    args: dict[str, Any] = {
        "config": UNA_SEMILLA,
        "run_commit": "abc",
        "live_latency_p99_min": 5.7,
    }
    with pytest.raises(SealError, match="el test ya se toco"):
        run_study(RAIZ, tmp_path, fabrica([]), **args)
    with pytest.raises(SealError, match="configuracion distinta"):
        run_study(
            RAIZ, tmp_path, fabrica([]), **{**args, "run_commit": "otro"}, resume=True
        )
    assert run_study(RAIZ, tmp_path, fabrica([]), **args, resume=True) == 0
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 33


def test_reanudar_salta_las_unidades_escritas(tmp_path: Path) -> None:
    args: dict[str, Any] = {
        "config": UNA_SEMILLA,
        "run_commit": "abc",
        "live_latency_p99_min": 5.7,
    }
    run_study(RAIZ, tmp_path, fabrica([]), **args)
    ruta = tmp_path / "records.jsonl"
    lineas = ruta.read_text().splitlines()
    ruta.write_text("\n".join(lineas[:-2]) + "\n")  # corte: faltan las ultimas dos
    assert run_study(RAIZ, tmp_path, fabrica([]), **args, resume=True) == 2
    assert ruta.read_text().splitlines()[:-2] == lineas[:-2]
    assert len(ruta.read_text().splitlines()) == 33


def test_walk_config_usa_el_agente_del_estudio_no_el_de_c3() -> None:
    assert StudyRunConfig().walk.total_timesteps == 120_000
    assert WalkForwardConfig().total_timesteps == 60_000  # C3 no se toca
