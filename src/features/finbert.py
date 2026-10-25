"""Puntaje de FinBERT sobre los titulares de GDELT (Etapa 4, PR N2b; ADR 0012, 3.1).

``finbert_mean`` es la media de P(positivo) - P(negativo) de los titulares de una barra.
Este modulo da los tres ladrillos de esa columna, sin red y sin reloj:

- :func:`normalize_title` y :class:`DedupWindow`: la deduplicacion del ADR 0012, 2.4.
- :class:`FinBertScorer`: el modelo pinneado por revision y por SHA256 de cada archivo.
- :func:`score_articles`: puntua en orden y devuelve un registro por articulo.

Anti-lookahead. El modelo es ``ProsusAI/finbert`` (BERT base de 2018 ajustado con
noticias financieras de 2008 a 2010): no sabe nada del periodo de test. Los pesos se
verifican contra el SHA256 de este archivo antes de cargarse; otro modelo "porque es
mejor" es otra configuracion y cuenta en ``n_trials``. La deduplicacion conserva la
**primera** aparicion de un titulo y descarta las posteriores: una aparicion solo
depende de los lotes anteriores o del mismo.

ASSUMPTION: el titulo se puntua despues de ``html.unescape`` y de colapsar espacios; el
GKG trae entidades HTML (``&amp;``, ``&#39;``) que el modelo nunca vio. El grabador
las guarda tal cual (``data.gdelt``) y limpiarlas es de quien puntua.
ASSUMPTION: la ventana de deduplicacion se mide contra la ultima aparicion
**conservada** del titulo: se descarta si pasaron menos de 24 h (``<``) desde ella, y
vuelve a conservarse a las 24 h exactas. El ADR dice "dentro de las 24 h previas,
quedandose con la primera aparicion" y no distingue entre medir contra la ultima
aparicion conservada o contra la ultima vista; la segunda silenciaria para siempre un
titulo que se repite cada 23 h.
ASSUMPTION: los titulos se truncan a ``MAX_LENGTH`` tokens (el p99 mide ~45).
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from data.errors import DataValidationError
from data.schema import FloatArray

MODEL_ID = "ProsusAI/finbert"
# Revision exacta del repositorio del modelo y huella de cada archivo que se carga.
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MODEL_FILES: Mapping[str, str] = {
    "config.json": "f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
    "model.safetensors": (
        "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00"
    ),
    "special_tokens_map.json": (
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
    ),
    "tokenizer_config.json": (
        "ffc7913f6084b138dad68de5a0c2f0ee25ac983709260901e01f53be5796b9d9"
    ),
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MAX_LENGTH = 128
MINI_BATCH = 32  # fijo: la composicion del lote de inferencia cambia el relleno
DEDUP_WINDOW = np.timedelta64(24, "h")
_ESPACIOS = re.compile(r"\s+")
_PUNTUACION = re.compile(r"[^\w\s]")


def clean_title(title: str) -> str:
    """Lo que ve el modelo: sin entidades HTML y con los espacios colapsados."""
    return _ESPACIOS.sub(" ", html.unescape(title)).strip()


def normalize_title(title: str) -> str:
    """Clave de deduplicacion (ADR 0012, 2.4): minusculas, sin puntuacion ni dobles
    espacios. NFKC primero, para que ``ﬁ`` y ``fi`` o un espacio duro no separen
    dos titulos iguales."""
    texto = unicodedata.normalize("NFKC", clean_title(title)).lower()
    return _ESPACIOS.sub(" ", _PUNTUACION.sub("", texto)).strip()


def title_key(title: str) -> str:
    """Huella corta del titulo normalizado: lo que se guarda para reanudar."""
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()[:16]


class DedupWindow:
    """Conserva la primera aparicion de cada titulo normalizado en 24 h.

    Los rotulos deben llegar en orden no decreciente: la decision sobre un titulo usa
    solo apariciones anteriores o iguales, nunca posteriores.
    """

    def __init__(self, window: np.timedelta64 = DEDUP_WINDOW) -> None:
        self._window = window
        self._ultima: dict[str, np.datetime64] = {}
        self._rotulo: np.datetime64 | None = None

    def keep(self, label: np.datetime64, title: str) -> bool:
        if self._rotulo is not None and label < self._rotulo:
            raise DataValidationError(
                f"rotulos desordenados en la deduplicacion: {label} < {self._rotulo}"
            )
        self._rotulo = label
        clave = title_key(title)
        previa = self._ultima.get(clave)
        if previa is not None and label - previa < self._window:
            return False
        self._ultima[clave] = label
        return True

    def restore(self, kept: Iterable[tuple[np.datetime64, str]]) -> None:
        """Estado al reanudar: ``(rotulo, clave)`` de lo ya conservado, en orden.

        Hace falta la ultima aparicion **conservada**, no la vista: rehacerla desde el
        crudo con una ventana recortada daria un estado distinto (ver ASSUMPTION).
        """
        for rotulo, clave in kept:
            self._ultima[clave] = rotulo
            self._rotulo = rotulo


class Scorer(Protocol):
    def score(self, titles: Sequence[str]) -> FloatArray:
        """``(n, 2)``: P(positivo) y P(negativo) de cada titulo, en ese orden."""
        ...


def verify_model_files(directory: Path) -> None:
    """Falla si falta un archivo del modelo o su SHA256 no es el pinneado."""
    for nombre, esperado in MODEL_FILES.items():
        ruta = directory / nombre
        if not ruta.is_file():
            raise DataValidationError(f"falta {nombre} en {directory}")
        digest = hashlib.sha256()
        with ruta.open("rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                digest.update(bloque)
        if digest.hexdigest() != esperado:
            raise DataValidationError(
                f"{nombre}: SHA256 {digest.hexdigest()} != pinneado {esperado}"
            )


class FinBertScorer:
    """``ProsusAI/finbert`` en CPU, modo evaluacion, fp32, sin cuantizar.

    Necesita el grupo ``nlp``. Los indices de positivo y negativo salen de
    ``id2label`` de la configuracion, no de un orden supuesto.
    """

    def __init__(self, directory: Path) -> None:
        verify_model_files(directory)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch: Any = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(directory)
        self._model: Any = AutoModelForSequenceClassification.from_pretrained(
            directory
        ).eval()
        etiquetas = {v: int(k) for k, v in self._model.config.id2label.items()}
        self._columnas = [etiquetas["positive"], etiquetas["negative"]]

    def score(self, titles: Sequence[str]) -> FloatArray:
        limpios = [clean_title(t) for t in titles]
        # Ordenar por largo reduce el relleno; el orden de salida se restituye.
        orden = sorted(range(len(limpios)), key=lambda i: len(limpios[i]))
        salida = np.zeros((len(limpios), 2), dtype=np.float64)
        with self._torch.inference_mode():
            for i in range(0, len(orden), MINI_BATCH):
                idx = orden[i : i + MINI_BATCH]
                enc = self._tokenizer(
                    [limpios[j] for j in idx],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                p = self._model(**enc).logits.softmax(-1).double().numpy()
                salida[idx] = p[:, self._columnas]
        return salida


@dataclass(frozen=True)
class ScoredArticle:
    label: np.datetime64
    record_id: str
    key: str  # title_key del titulo; permite reanudar la deduplicacion
    pos: float
    neg: float

    @property
    def score(self) -> float:
        return self.pos - self.neg


def score_articles(
    articles: Iterable[tuple[np.datetime64, str, str]],
    scorer: Scorer,
    dedup: DedupWindow,
    cache: dict[str, tuple[float, float]] | None = None,
) -> list[ScoredArticle]:
    """``(rotulo, record_id, titulo)`` en orden -> un puntaje por articulo conservado.

    ``cache`` guarda el puntaje por titulo exacto: un titulo repetido pasado el plazo de
    deduplicacion recibe el mismo puntaje sin volver a correr el modelo.
    """
    cache = {} if cache is None else cache
    conservados = [a for a in articles if dedup.keep(a[0], a[2])]
    nuevos = sorted({a[2] for a in conservados if a[2] not in cache})
    if nuevos:
        for titulo, (pos, neg) in zip(nuevos, scorer.score(nuevos), strict=True):
            cache[titulo] = (float(pos), float(neg))
    return [
        ScoredArticle(lab, rid, title_key(t), *cache[t]) for lab, rid, t in conservados
    ]
