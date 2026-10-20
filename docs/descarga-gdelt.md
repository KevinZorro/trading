# Descarga del histórico de GDELT (Etapa 4, PR N2)

`scripts/descargar_gdelt.py` baja el GKG 2.0 de GDELT y escribe los mismos
`news-AAAA-MM-DD.jsonl` que el grabador en vivo, así `data.news.load_news` lee histórico
y vivo con el mismo código. Se queda con los artículos `ECON_BITCOIN`; el archivo crudo
se verifica contra el MD5 del `masterfilelist.txt` y se descarta.

## Uso

```bash
PYTHONPATH=src python scripts/descargar_gdelt.py --out datasets/gdelt/raw \
    --start 2016-08-01 --end 2017-01-01 --workers 8
```

- `--start` inclusivo y `--end` exclusivo, UTC. Reanudable: repetir el mismo comando salta
  los rótulos ya escritos. Sale con código 1 si algún lote agotó los reintentos (no queda
  escrito; volver a correr lo reintenta).
- Un rótulo que el masterfilelist no lista, o que da 404, queda `status: "missing"`.
- Cada corrida lee el masterfilelist entero (~128 MB, ~5 s) para filtrar su rango.

## Medido (2026-10-20, contenedor de la nube, 4 núcleos)

| Prueba | Resultado |
|---|---|
| 1 día (96 lotes, ~600-680 MB) con 8, 16 y 32 hilos en **un** proceso | ~21 MB/s agregados, igual con 8 que con 32 |
| Dos procesos a la vez (un día cada uno) | ~21 MB/s cada uno, ~42 MB/s en total |
| Un archivo suelto | 8 MB/s, el parseo toma 0,15 s |

Más hilos dentro de un proceso no ayudan; más procesos sí. A ~21 MB/s por proceso, los
~2,6 TB de 2016-08 a 2026-09 serían ~35 h con un proceso y ~9 h con cuatro (uno por
rango de fechas, cada uno con su `--out`; no comparten directorio).

## Dos hechos del dato que cambian el diseño

1. **`PAGE_TITLE` no existe antes de ~2019-09-23.** Muestreados 14 lotes de 2016-08 a
   2020-02: ninguna fila trae título hasta 2019-09-22; desde 2019-09-23 todas. El tono
   (`V2Tone`) sí existe en todo el rango. Sin título no hay FinBERT ni deduplicación por
   título.
2. **La sindicación existe también sin título.** 2017-09-01 19:00: cinco diarios de
   Hearst (`greenwichtime`, `newstimes`, `thehour`, `stamfordadvocate`, `ctpost`) con el
   mismo tono 1,49253731343284. `dedupe_first_seen` no las detecta: sin título no hay clave.

## Deduplicación

`data.gdelt_history.dedupe_first_seen` quita copias del mismo título normalizado (NFKC,
minúsculas, solo alfanuméricos) vistas hace menos de 24 h, y conserva la **primera**. No
se aplica al escribir: los archivos conservan todo.
