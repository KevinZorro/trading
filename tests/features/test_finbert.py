"""FinBERT: deduplicacion, pin por hash y puntaje, con un modelo doble.

El modelo real no se carga en el CI (grupo ``nlp``, 440 MB): el test marcado ``nlp``
lo corre quien tenga ``FINBERT_DIR``. Los oraculos de aqui son a mano.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from data.errors import DataValidationError
from data.schema import FloatArray
from features import finbert
from features.finbert import (
    DedupWindow,
    FinBertScorer,
    clean_title,
    normalize_title,
    score_articles,
    verify_model_files,
)

H = np.timedelta64(1, "h")
T0 = np.datetime64("2020-03-01T00:00:00", "ns")


class Doble:
    """Puntaje conocido: pos = len/100, neg = 0.1. Cuenta lo que le piden."""

    def __init__(self) -> None:
        self.pedidos: list[str] = []

    def score(self, titles: Sequence[str]) -> FloatArray:
        self.pedidos += list(titles)
        return np.array([[len(t) / 100, 0.1] for t in titles])


def test_clean_y_normalize_a_mano() -> None:
    assert (
        clean_title("  Bitcoin &amp; Ether&#39;s   rally ") == "Bitcoin & Ether's rally"
    )
    assert normalize_title("Bitcoin &amp; Ether's  RALLY!") == "bitcoin ethers rally"
    # NFKC: la ligadura y el espacio duro no separan titulos iguales.
    assert normalize_title("ﬁnance\u00a0news") == normalize_title("finance news")


def test_dedup_conserva_la_primera_y_mide_contra_la_ultima_conservada() -> None:
    d = DedupWindow()
    assert d.keep(T0, "BTC sube")
    assert not d.keep(T0, "btc sube!")  # mismo lote, misma clave
    assert not d.keep(T0 + 23 * H, "BTC sube")  # 23 h < 24 h
    # 25 h desde la ultima conservada (T0): vuelve a conservarse. Medida contra la
    # ultima vista (T0+23h) seria 2 h y se descartaria.
    assert d.keep(T0 + 25 * H, "BTC sube")


def test_dedup_el_limite_de_24_h_exactas_se_conserva() -> None:
    d = DedupWindow()
    d.keep(T0, "x")
    assert d.keep(T0 + 24 * H, "x")


def test_dedup_rechaza_rotulos_desordenados() -> None:
    d = DedupWindow()
    d.keep(T0 + H, "x")
    with pytest.raises(DataValidationError, match="desordenados"):
        d.keep(T0, "y")


def test_restore_da_las_mismas_decisiones_que_haber_seguido_de_corrido() -> None:
    rng = np.random.default_rng(3)
    eventos = [
        (T0 + int(h) * H, f"titulo {int(k)}")
        for h, k in zip(
            np.sort(rng.integers(0, 24 * 10, 400)),
            rng.integers(0, 15, 400),
            strict=True,
        )
    ]
    completo = DedupWindow()
    decisiones = [completo.keep(t, s) for t, s in eventos]
    corte = 250
    primera = DedupWindow()
    previas = [primera.keep(t, s) for t, s in eventos[:corte]]
    conservados = [
        (t, finbert.title_key(s))
        for (t, s), k in zip(eventos[:corte], previas, strict=True)
        if k
    ]
    segunda = DedupWindow()
    segunda.restore(conservados)
    resto = [segunda.keep(t, s) for t, s in eventos[corte:]]
    assert previas + resto == decisiones


@pytest.mark.leakage
def test_la_decision_de_una_aparicion_no_depende_de_las_posteriores() -> None:
    eventos = [(T0 + i * H, "mismo titulo") for i in (0, 5, 24, 30, 49)]
    a, b = DedupWindow(), DedupWindow()
    corto = [a.keep(t, s) for t, s in eventos[:3]]
    largo = [b.keep(t, s) for t, s in eventos]
    assert largo[:3] == corto == [True, False, True]
    assert largo[3:] == [False, True]  # 30 h: 6 h tras T0+24; 49 h: 25 h tras T0+24


def test_score_articles_ordena_deduplica_y_cachea() -> None:
    doble = Doble()
    cache: dict[str, tuple[float, float]] = {}
    arts = [
        (T0, "a1", "dos"),
        (T0, "a2", "Dos."),  # duplicado del anterior
        (T0 + H, "a3", "tres tres"),
        (T0 + 30 * H, "a4", "dos"),  # pasadas 24 h: se conserva, ya esta en cache
    ]
    r = score_articles(arts, doble, DedupWindow(), cache)
    assert [a.record_id for a in r] == ["a1", "a3", "a4"]
    assert doble.pedidos == ["dos", "tres tres"]  # cada titulo exacto, una vez
    assert r[0].score == pytest.approx(0.03 - 0.1)
    assert r[2].score == r[0].score
    # Segunda llamada con la cache: no pide nada al modelo.
    score_articles([(T0 + 60 * H, "a5", "dos")], doble, DedupWindow(), cache)
    assert doble.pedidos == ["dos", "tres tres"]


def test_verify_model_files_rechaza_falta_y_alteracion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    (tmp_path / "a.txt").write_bytes(b"pesos")
    monkeypatch.setattr(
        finbert, "MODEL_FILES", {"a.txt": hashlib.sha256(b"pesos").hexdigest()}
    )
    verify_model_files(tmp_path)  # coincide
    (tmp_path / "a.txt").write_bytes(b"pesos alterados")
    with pytest.raises(DataValidationError, match="SHA256"):
        verify_model_files(tmp_path)
    (tmp_path / "a.txt").unlink()
    with pytest.raises(DataValidationError, match="falta a.txt"):
        verify_model_files(tmp_path)


def test_el_pin_lista_los_cinco_archivos_con_sha256() -> None:
    assert set(finbert.MODEL_FILES) == {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    assert all(len(h) == 64 for h in finbert.MODEL_FILES.values())
    assert len(finbert.MODEL_REVISION) == 40


@pytest.mark.nlp
def test_el_modelo_real_separa_positivo_de_negativo() -> None:
    directorio = os.environ.get("FINBERT_DIR")
    if not directorio:
        pytest.skip("FINBERT_DIR no definido")
    pytest.importorskip("transformers")
    scorer = FinBertScorer(Path(directorio))
    p = scorer.score(
        [
            "Company reports record profit and raises its dividend",
            "Company files for bankruptcy after massive losses",
        ]
    )
    assert p.shape == (2, 2)
    assert p[0, 0] > 0.8 > 0.2 > p[0, 1]  # positivo
    assert p[1, 1] > 0.8 > 0.2 > p[1, 0]  # negativo
    # Determinista: el orden de entrada no cambia el puntaje de un titulo.
    inverso = scorer.score(
        [
            "Company files for bankruptcy after massive losses",
            "Company reports record profit and raises its dividend",
        ]
    )
    np.testing.assert_allclose(inverso[::-1], p, atol=1e-5)
