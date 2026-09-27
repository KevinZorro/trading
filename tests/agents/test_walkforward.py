"""Walk-forward sobre BTCUSDT diario (ADR 0006): geometria, regimen, sellado y
veredictos. Sin torch: la politica se inyecta.

Los tests que leen ``datasets/binance`` usan el dataset versionado, que la
carga verifica por SHA256; no tocan red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import agents.walkforward as wf
from agents.policy import ConstantWeightPolicy
from agents.walkforward import (
    FoldSeedRecord,
    RegimeClass,
    SealError,
    WalkForwardConfig,
    acquire_seal,
    classify_regime,
    evaluate_fold_seed,
    folds,
    load_dataset,
    main_verdict,
    prediction_verdict,
    run_walkforward,
    training_span,
    verify_adr,
)
from data.instruments import binance_spot_spec
from data.loaders import bars_from_frame
from data.synthetic import generate_gbm_sv
from eval.walkforward import Window, coverage

RAIZ = Path(__file__).resolve().parents[2]


def constante(peso: float = 1.0) -> Any:
    return lambda _make_env, _seed: ConstantWeightPolicy(peso)


@pytest.fixture(scope="module")
def dataset() -> Any:
    return load_dataset(RAIZ)


# ---------------------------------------------------------------------------
# Geometria sellada (ADR 0006, seccion 1)
# ---------------------------------------------------------------------------


def test_la_geometria_es_la_sellada(dataset: Any) -> None:
    serie, _, _ = dataset
    ventanas = folds(len(serie), WalkForwardConfig())
    cobertura = coverage(ventanas, len(serie))
    assert len(ventanas) == 11
    assert ventanas[0].test == (1000, 1200)
    assert ventanas[-1].test == (3000, 3200)
    assert cobertura["overlapping_test_bars"] == 0
    assert cobertura["unused_tail"] == 128


def test_2022_queda_cubierto_entero_por_los_tests(dataset: Any) -> None:
    """Obligatorio: un agente que solo se ve bien en ventanas alcistas no
    demostro nada. Toda barra que cierra entre el 2022-01-02 y el 2023-01-01
    (el ano 2022 con timestamp de cierre) cae en algun test."""
    serie, _, _ = dataset
    ts = pd.DatetimeIndex(serie.timestamp)
    de_2022 = np.flatnonzero(
        (ts >= pd.Timestamp("2022-01-02")) & (ts <= pd.Timestamp("2023-01-01"))
    )
    assert len(de_2022) == 365
    en_test: set[int] = set()
    for ventana in folds(len(serie), WalkForwardConfig()):
        en_test |= set(range(*ventana.test))
    assert set(de_2022.tolist()) <= en_test


def test_se_entrena_con_train_mas_validacion() -> None:
    """D2: las 1.000 barras, sin el test."""
    ventana = Window(index=0, train=(0, 800), validation=(800, 1000), test=(1000, 1200))
    assert training_span(ventana) == (0, 1000)


# ---------------------------------------------------------------------------
# Regimen (seccion 4.1)
# ---------------------------------------------------------------------------


def test_la_clasificacion_reproduce_la_tabla_sellada(dataset: Any) -> None:
    """Fija la clasificacion pre-registrada: si cambia el dato o la regla, este
    test cae antes que la corrida."""
    serie, _, _ = dataset
    clases = {
        v.index: classify_regime(serie, v)
        for v in folds(len(serie), WalkForwardConfig())
    }
    etiquetas = {k: c.label for k, c in clases.items()}
    assert {k for k, v in etiquetas.items() if v == "tras_cambio"} == {1, 4, 7}
    assert {k for k, v in etiquetas.items() if v == "sin_cambio"} == {2, 10}
    assert sum(v == "ambigua" for v in etiquetas.values()) == 6
    assert round(clases[4].r_ant, 2) == 1.70
    assert round(clases[4].t_rec, 2) == -1.36


def _serie_con_retornos(retornos: list[float]) -> Any:
    """Serie diaria con log-retornos dados (open de la primera barra = 100)."""
    cierre = 100.0 * np.exp(np.cumsum(retornos))
    apertura = np.concatenate([[100.0], cierre[:-1]])
    marco = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-02", periods=len(cierre), tz="UTC"),
            "open": apertura,
            "high": np.maximum(apertura, cierre) * 1.001,
            "low": np.minimum(apertura, cierre) * 0.999,
            "close": cierre,
            "volume": 1_000.0,
            "symbol": "BTCUSDT",
        }
    )
    return bars_from_frame(marco, instrument=binance_spot_spec(), freq="1D")


@pytest.mark.parametrize(
    ("anterior", "reciente", "esperado"),
    [
        (0.01, -0.02, "tras_cambio"),  # alcista y luego una caida nitida
        (0.01, 0.02, "sin_cambio"),
        (-0.01, 0.02, "tras_cambio"),
        (0.01, -0.0001, "ambigua"),  # cambio de signo sobre ruido
    ],
)
def test_regla_de_regimen_con_oraculo(
    anterior: float, reciente: float, esperado: str
) -> None:
    """Deriva constante mas un ruido alternante de +-1 %: el t reciente es
    ``200 * reciente / (0.01 * sqrt(200))`` y el oraculo se calcula de cabeza."""
    ruido = [0.01 if i % 2 else -0.01 for i in range(200)]
    retornos = [anterior] * 8 + [reciente + x for x in ruido]
    serie = _serie_con_retornos(retornos)
    ventana = Window(index=0, train=(0, 8), validation=(8, 208), test=(208, 208 + 1))
    clase = classify_regime(serie, ventana)
    assert clase.label == esperado


# ---------------------------------------------------------------------------
# Sellado (seccion 7)
# ---------------------------------------------------------------------------


def test_el_adr_en_disco_es_el_sellado() -> None:
    assert verify_adr(RAIZ) == wf.ADR_SHA256


def test_un_adr_distinto_no_corre() -> None:
    with pytest.raises(SealError, match="el criterio cambio despues de sellarse"):
        verify_adr(RAIZ, expected="0" * 64)


def test_el_test_no_se_toca_dos_veces(tmp_path: Path) -> None:
    sello = tmp_path / "SEAL.json"
    carga = {"adr_sha256": "a", "dataset_sha256": "b", "config": "c"}
    acquire_seal(sello, carga, resume=False)
    with pytest.raises(SealError, match="el test ya se toco"):
        acquire_seal(sello, carga, resume=False)
    acquire_seal(sello, carga, resume=True)  # misma configuracion: continua
    with pytest.raises(SealError, match="configuracion distinta"):
        acquire_seal(sello, {**carga, "config": "otra"}, resume=True)


# ---------------------------------------------------------------------------
# Un par (fold, semilla)
# ---------------------------------------------------------------------------


def _fold_sintetico(gap: float = 0.0) -> tuple[Any, Window]:
    serie = generate_gbm_sv(
        320, seed=3, instrument=binance_spot_spec(), overnight_gap_frac=gap
    )
    return serie, Window(
        index=0, train=(0, 150), validation=(150, 200), test=(200, 260)
    )


def test_el_escalador_se_ajusta_solo_con_el_entrenamiento(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serie, ventana = _fold_sintetico()
    vistas: list[Any] = []
    original = wf.scaler_for

    def espia(entrenamiento: Any, observacion: Any = None) -> Any:
        vistas.append(entrenamiento)
        return original(entrenamiento, observacion)

    monkeypatch.setattr(wf, "scaler_for", espia)
    evaluate_fold_seed(
        serie, np.zeros(len(serie)), ventana, 1, WalkForwardConfig(), constante()
    )
    (visto,) = vistas
    assert len(visto) == 200
    assert visto.timestamp[-1] == serie.timestamp[199]  # termina antes del test


def test_agente_y_baselines_deciden_por_primera_vez_en_el_inicio_del_test() -> None:
    serie, ventana = _fold_sintetico()
    registro = evaluate_fold_seed(
        serie, np.zeros(len(serie)), ventana, 1, WalkForwardConfig(), constante()
    )
    # 60 barras de test: 59 retornos diarios de exceso, desde la primera decision.
    assert len(registro.excess_daily) == 59
    assert registro.n_outliers == 0


def test_invertido_al_maximo_da_timing_casi_nulo() -> None:
    """Un peso constante de 1.0 es un buy-and-hold con exposicion ~0.98: su
    exceso contra el B&H de exposicion igualada es casi cero. La exposicion
    media es ``0.98 * 59/60``: en la barra de la primera decision la posicion
    todavia es cero, porque el fill llega en la barra siguiente."""
    serie, ventana = _fold_sintetico()
    registro = evaluate_fold_seed(
        serie, np.zeros(len(serie)), ventana, 1, WalkForwardConfig(), constante()
    )
    assert registro.exposure == pytest.approx(0.98 * 59 / 60, abs=0.002)
    assert abs(registro.t_mark) < 0.01


def test_una_decision_en_barra_marcada_es_outlier_y_no_se_filtra() -> None:
    """Seccion 5: se reporta aparte, pero el resultado principal la incluye.
    Con gap entre barras (Heston continuo no lo tiene), neutralizar el gap de
    los fills outlier mueve la version secundaria y deja quieta la principal."""
    serie, ventana = _fold_sintetico(gap=0.5)
    marcas = np.zeros(len(serie))
    marcas[ventana.test[0] : ventana.test[1]] = -1_000.0  # todo el test marcado
    con = evaluate_fold_seed(
        serie, marcas, ventana, 1, WalkForwardConfig(), constante()
    )
    sin = evaluate_fold_seed(
        serie, np.zeros(len(serie)), ventana, 1, WalkForwardConfig(), constante()
    )
    assert con.n_outliers >= 1
    assert con.e_mark == sin.e_mark  # el principal no cambia
    assert con.e_mark_sin_outliers != con.e_mark


# ---------------------------------------------------------------------------
# Veredictos (secciones 2 y 4.3), con oraculos a mano
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x_mark", "x_liq", "dsr", "esperado"),
    [
        ([0.1, 0.2, 0.3], [0.1, 0.1, 0.2], [0.96, 0.97, 0.99], "SUPERA"),
        ([0.1, 0.2, 0.3], [-0.1, -0.1, 0.2], [0.96, 0.97, 0.99], "NO SUPERA"),
        ([0.1, 0.2, 0.3], [0.1, 0.1, 0.2], [0.90, 0.94, 0.99], "NO SUPERA"),
        ([-0.1, 0.0, 0.3], [0.1, 0.1, 0.2], [0.96, 0.97, 0.99], "NO SUPERA"),
        ([0.1, 0.2, 0.3], [0.1, 0.1, 0.2], [float("nan")] * 3, "NO SUPERA"),
    ],
)
def test_veredicto_principal(
    x_mark: list[float], x_liq: list[float], dsr: list[float], esperado: str
) -> None:
    assert main_verdict(x_mark, x_liq, dsr) == esperado


def _t(mediana: float, ancho: float) -> list[float]:
    """Cinco semillas con mediana e IQR dados: [m-a, m-a/2, m, m+a/2, m+a]
    tiene IQR exactamente ``ancho``."""
    return [
        mediana - ancho,
        mediana - ancho / 2,
        mediana,
        mediana + ancho / 2,
        mediana + ancho,
    ]


ETIQUETAS = {
    1: "tras_cambio",
    4: "tras_cambio",
    7: "tras_cambio",
    2: "sin_cambio",
    10: "sin_cambio",
    0: "ambigua",
}


@pytest.mark.parametrize(
    ("tras", "sin", "esperado"),
    [
        (-0.30, 0.00, "CONSISTENTE"),  # G = -0.30 < -F = -0.10
        (-0.05, 0.00, "NO CONCLUYENTE"),  # -0.10 <= G < 0
        (0.00, 0.00, "REFUTADA"),  # G = 0
        (0.20, 0.00, "REFUTADA"),
    ],
)
def test_veredicto_de_la_prediccion(tras: float, sin: float, esperado: str) -> None:
    t = {k: _t(tras if v == "tras_cambio" else sin, 0.10) for k, v in ETIQUETAS.items()}
    salida = prediction_verdict(t, ETIQUETAS)
    assert salida["verdict"] == esperado
    assert salida["F"] == pytest.approx(0.10)


def test_con_menos_de_dos_folds_por_clase_no_es_contrastable() -> None:
    etiquetas = {1: "tras_cambio", 2: "sin_cambio", 10: "sin_cambio"}
    t = {k: _t(-1.0, 0.1) for k in etiquetas}
    assert prediction_verdict(t, etiquetas)["verdict"] == "NO CONTRASTABLE"


def test_confirmada_no_existe_como_veredicto() -> None:
    """El maximo es CONSISTENTE: este diseno no puede confirmar (N = 1)."""
    t = {
        k: _t(-100.0 if v == "tras_cambio" else 100.0, 0.0)
        for k, v in ETIQUETAS.items()
    }
    assert prediction_verdict(t, ETIQUETAS)["verdict"] == "CONSISTENTE"


# ---------------------------------------------------------------------------
# Corrida completa sobre el dataset real, con politica inyectada
# ---------------------------------------------------------------------------


def test_corrida_completa_sellada_persistida_y_reanudable(tmp_path: Path) -> None:
    """Sin marcador slow a proposito: tarda ~5 s y es el unico test que ejercita
    ``run_walkforward`` y ``assemble`` completos en el job principal."""
    config = WalkForwardConfig(seeds=(1, 2))
    reporte = run_walkforward(RAIZ, tmp_path, constante(), config=config)

    registros = (tmp_path / "records.jsonl").read_text().splitlines()
    assert len(registros) == 22  # 11 folds x 2 semillas, uno por linea
    assert FoldSeedRecord.from_dict(json.loads(registros[0])).fold == 0
    assert (
        json.loads((tmp_path / "SEAL.json").read_text())["adr_sha256"] == wf.ADR_SHA256
    )
    assert reporte["main"]["verdict"] in {"SUPERA", "NO SUPERA"}
    assert reporte["main"]["n_outliers"] == 0  # seccion 5: ninguno esperado
    assert reporte["provenance"]["dataset_sha256"].startswith("0940f089")
    clases = [RegimeClass(**c) for c in reporte["prediction_adr_0004"]["classes"]]
    assert len(clases) == 11
    subsidiaria = reporte["prediction_adr_0004"]["subsidiary_3"]
    assert set(subsidiaria["per_fold"]) == {1, 4, 7}
    assert isinstance(subsidiaria["sostenida"], bool)
    assert reporte["provenance"]["adr_seal_commit"] == "fcc10ea"

    with pytest.raises(SealError, match="el test ya se toco"):
        run_walkforward(RAIZ, tmp_path, constante(), config=config)
    run_walkforward(RAIZ, tmp_path, constante(), config=config, resume=True)
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 22


def test_el_digest_del_sello_cubre_los_hiperparametros_de_ppo() -> None:
    """Cambiar un hiperparametro de PPO despues de sellar se detecta: entra al
    hash que se compara al reanudar."""
    base = WalkForwardConfig()
    assert base.digest() == WalkForwardConfig().digest()  # determinista
    assert base.digest() != WalkForwardConfig(total_timesteps=240_000).digest()
    assert base.describe()["ppo"]["ent_coef"] == 0.0
    assert base.ppo_config().total_timesteps == 60_000  # opcion i, como el protocolo
