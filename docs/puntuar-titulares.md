# Puntaje de titulares con FinBERT (Etapa 4, PR N2b)

`scripts/puntuar_titulares.py` puntúa con `ProsusAI/finbert` los titulares del crudo de
GDELT y agrega por lote de 15 min. Alimenta `finbert_mean` de `features/news.py`
(ADR 0012, 3 y 3.1).

## Uso

Necesita el grupo `nlp` (`uv sync --group nlp --group rl`; torch viene de `rl`).

```bash
PYTHONPATH=src python scripts/puntuar_titulares.py \
    --raw RUTA/gdelt-raw --model RUTA/finbert --out RUTA/finbert-salida \
    --agregados datasets/gdelt --descargar
```

- `--descargar` baja los cinco archivos del modelo en la revisión pinneada y
  `FinBertScorer` verifica el SHA256 de cada uno antes de cargarlo.
- Reanudable: repetir el comando sigue desde el último rótulo completo, con la
  deduplicación restaurada desde lo ya escrito. Una caída a mitad de línea se repara sola.
- Salida: `finbert-articles.jsonl` (nivel artículo, al Release) y
  `datasets/gdelt/finbert-batches.csv.gz` con su manifest (al repositorio).
  Se lee con `features.finbert_dataset.read_finbert_aggregates` y se une a los agregados
  de GDELT con `attach_finbert`.

## Medido (2026-10-25/26, contenedor de la nube, 4 núcleos)

| | |
|---|---|
| Titulares con título en el crudo | 1 974 145 |
| Conservados tras la deduplicación de 24 h | 1 412 307 |
| Lotes con al menos un titular puntuado | 216 243 |
| Velocidad | 25 a 78 titulares/s según la carga de la máquina |
| Barras diarias con `finbert_mean` | 2 547 de 3 711; las 1 164 restantes (hasta 2019-09-22) tienen `no_title` = 1 |

`finbert_mean` es `NaN` exactamente donde `no_title` = 1, sin excepciones.

## Límites que conviene saber

- **No es bit-reproducible entre composiciones de lote de inferencia:** reanudar a mitad
  da diferencias de ~3e-6 en P(positivo). El artefacto sellado es el archivo de artículos
  con su SHA256 (en el manifest), no su regeneración.
- La deduplicación mide contra la última aparición **conservada** (ver `features/finbert.py`).
- Los titulares se puntúan tras `html.unescape`, con `max_length` = 128 tokens.

## Release

Los artefactos completos no van al repositorio (`CLAUDE.md`, tamaño de PR). Van al
Release `n2b-articulos`:

| Archivo | Qué es |
|---|---|
| `finbert-articles.jsonl.gz` | puntajes por artículo (SHA256 en el manifest, descomprimido) |
| `gdelt-raw.tar.gz` | el crudo de GDELT con títulos (`MANIFEST.json` con el SHA256 de cada archivo) |
