# Despliegue del grabador de noticias GDELT (Etapa 4, PR L1)

El grabador (`python -m live.news`) baja cada lote de 15 minutos del GKG de GDELT, se
queda con los artículos del tema `ECON_BITCOIN` y agrega una línea por lote a
`/var/lib/news/news-AAAA-MM-DD.jsonl`. Es un servicio **aparte** (`noticias`) con su
propia imagen: levantarlo no toca `shadow` ni `watchdog`. La sombra tiene un reloj de
30 días en marcha y **no se reinicia**.

Red: solo HTTPS a `data.gdeltproject.org`. No usa `.env`, ni claves, ni Telegram.
Memoria: tope de 256 MB en el compose (el proceso ronda los 110 MB al importar y procesa
un lote por vez en streaming).

## Qué queda grabado

Una línea JSON por lote (la primera línea de cada archivo es la cabecera). Campos:

- `label`: rótulo del lote (UTC). **No es** `pub_time`: la regla anti-lookahead
  (rótulo + latencia de 30 min) la aplica la alineación de la Etapa 4.
- `received_at`: hora de recepción propia. `last_404_at`: última vez que se pidió y no
  existía. La disponibilidad real está entre los dos (`last_404_at` es `null` si el
  primer intento ya encontró el lote: entonces solo hay cota superior).
- `last_modified`, `md5`, `bytes`: del archivo de origen; el MD5 es comparable con el de
  `masterfilelist.txt` y `lastupdate.txt`.
- `rows_total`, `rows_malformed`, `rows_theme` y `articles` (id, dominio, URL, 7 campos de
  `V2Tone`, título). El texto del artículo no existe en el GKG ni se guarda.
- `status`: `ok` o `missing`. Un lote que no aparece en 2 h queda `missing` con
  `attempts` y, si hubo, `last_error`. GDELT tiene huecos propios (el lote 01:30 del
  2026-10-20 no existe).

## Desplegar (solo el servicio nuevo)

Dentro de la VM (después del ssh), primera vez:

```bash
sudo install -d -o 10001 -g 10001 -m 750 /var/lib/news
cd /opt/trading
git pull --ff-only origin main
export GIT_SHA=$(git rev-parse --short HEAD)
docker compose build noticias
docker compose up -d --no-deps noticias
```

`--no-deps` y el nombre del servicio garantizan que Compose no recree `shadow` ni
`watchdog`. **Nunca** `docker compose up -d` sin nombre ni `docker compose build` sin
nombre mientras corra el reloj de la sombra.

Comprobar que la sombra no se movió y que el grabador anda:

```bash
docker compose ps
docker compose logs --tail 20 noticias
docker stats --no-stream
cat /var/lib/news/heartbeat.json
ls -la /var/lib/news
```

`shadow` debe seguir `Up` con la misma antigüedad de antes. Tras ~20 minutos tiene que
haber un `news-AAAA-MM-DD.jsonl` con la cabecera y al menos un lote `ok`.

## Operación

- Reinicia solo (`restart: unless-stopped`) y retoma después del último rótulo grabado;
  tras una caída larga recupera hasta 7 días de lotes (`NEWS_MAX_CATCHUP_BATCHES`).
  En una recuperación `received_at` es la hora de la descarga tardía, no la de
  disponibilidad: para medir latencia se usan solo los lotes con `last_404_at`.
- `docker compose ps` marca `unhealthy` si el heartbeat tiene más de 45 min.
- Sin alertas de Telegram todavía (queda como mejora): hay que mirar `docker compose ps`.
- Copia de los datos: `rsync -a azureuser@IP:/var/lib/news/ ./noticias-datos/`.

Variables (todas opcionales salvo `NEWS_DATA_DIR`, que el compose ya fija): `NEWS_THEME`,
`NEWS_BASE_URL`, `NEWS_POLL_SECONDS` (60), `NEWS_LEAD_MINUTES` (15),
`NEWS_GIVE_UP_MINUTES` (120), `NEWS_TIMEOUT_S`, `NEWS_MAX_CATCHUP_BATCHES`,
`NEWS_HEARTBEAT_MAX_AGE_MINUTES`.
