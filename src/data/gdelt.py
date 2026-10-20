"""Parser puro del GKG 2.0 de GDELT (Etapa 4). Sin red, sin reloj, sin disco.

Lo usan el grabador en vivo (``live.news``) y, mas adelante, la descarga
historica: los dos leen el mismo formato con el mismo codigo, para que el
historico y lo grabado en vivo sean comparables fila a fila.

Un lote es un archivo ``AAAAMMDDHHMMSS.gkg.csv.zip`` cada 15 minutos, separado por
tabuladores, 27 columnas (GKG 2.1). Las que importan aqui:

- 0 ``GKGRECORDID``, 3 ``SourceCommonName`` (dominio), 4 ``DocumentIdentifier`` (URL)
- 8 ``V2EnhancedThemes``: ``TEMA,offset;TEMA,offset;...``
- 15 ``V2Tone``: siete numeros separados por coma
- 26 ``Extras``: contiene ``<PAGE_TITLE>titulo</PAGE_TITLE>``

ASSUMPTION: el tema se busca por **token exacto** en V2EnhancedThemes. Un
``in`` sobre la fila entera aceptaria cualquier campo que contenga el texto.
ASSUMPTION: el titulo se guarda tal cual viene (con entidades HTML si las trae);
limpiarlo es decision de quien lo puntue, no de quien lo graba.
El texto del articulo no existe en el GKG ni se guarda (derechos de autor).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

THEME_BITCOIN = "ECON_BITCOIN"
N_COLUMNAS = 27
N_TONO = 7  # tono, positivo, negativo, polaridad, actividad, auto-referencia, palabras
LOTE_MINUTOS = 15
GKG_BASE_URL = "https://data.gdeltproject.org/gdeltv2"

_COL_ID, _COL_DOMINIO, _COL_URL, _COL_TEMAS, _COL_TONO, _COL_EXTRAS = 0, 3, 4, 8, 15, 26
_TITULO = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


@dataclass(frozen=True)
class GkgArticle:
    record_id: str
    domain: str
    url: str
    tone: tuple[float, ...]  # los siete campos de V2Tone
    title: str | None


@dataclass(frozen=True)
class GkgBatch:
    rows_total: int
    rows_malformed: int  # no tienen 27 columnas; cuentan en el total
    articles: tuple[GkgArticle, ...]  # las filas con el tema, en orden de archivo


def batch_label_str(label: np.datetime64) -> str:
    """``2026-10-20T01:15`` -> ``20261020011500``."""
    s = str(np.datetime64(label, "s"))
    return s.replace("-", "").replace(":", "").replace("T", "")


def gkg_url(label: np.datetime64, base_url: str = GKG_BASE_URL) -> str:
    return f"{base_url}/{batch_label_str(label)}.gkg.csv.zip"


def floor_label(instante: np.datetime64) -> np.datetime64:
    """Rotulo de lote (multiplo de 15 min UTC) menor o igual a ``instante``."""
    ns = np.datetime64(instante, "ns").astype("int64")
    paso = int(
        np.timedelta64(LOTE_MINUTOS, "m").astype("timedelta64[ns]").astype("int64")
    )
    return np.datetime64(int(ns // paso * paso), "ns")


def _tono(texto: str) -> tuple[float, ...] | None:
    try:
        valores = tuple(float(x) for x in texto.split(","))
    except ValueError:
        return None
    return valores if len(valores) == N_TONO else None


def parse_gkg(lineas: Iterable[str], theme: str = THEME_BITCOIN) -> GkgBatch:
    total = malformadas = 0
    articulos: list[GkgArticle] = []
    for linea in lineas:
        linea = linea.rstrip("\r\n")
        if not linea:
            continue
        total += 1
        c = linea.split("\t")
        if len(c) != N_COLUMNAS:
            malformadas += 1
            continue
        if theme not in (t.split(",", 1)[0] for t in c[_COL_TEMAS].split(";")):
            continue
        tono = _tono(c[_COL_TONO])
        if tono is None:  # tema presente pero tono ilegible: no se inventa uno
            malformadas += 1
            continue
        titulo = _TITULO.search(c[_COL_EXTRAS])
        articulos.append(
            GkgArticle(
                c[_COL_ID],
                c[_COL_DOMINIO],
                c[_COL_URL],
                tono,
                titulo.group(1) if titulo else None,
            )
        )
    return GkgBatch(total, malformadas, tuple(articulos))
