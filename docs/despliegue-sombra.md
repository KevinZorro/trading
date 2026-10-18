# Despliegue de la sombra en un VPS

Guía para Kevin. Todo lo de aquí lo hace una persona: crear el servidor, poner los
secretos y levantar los contenedores. Ver `docs/adr/0007-modo-sombra-y-testnet.md`.

La predicción pre-registrada queda **NO EVALUABLE con menos de 25 de 30 snapshots
válidos de las 00:00:30 UTC**. Cada día sin proceso vivo a esa hora cuenta.

## Qué corre

| Servicio | Comando | Datos | Función |
|---|---|---|---|
| `shadow` | `python -m live.shadow` | lectura y escritura | toma los snapshots |
| `watchdog` | `python -m live.watchdog` | solo lectura | avisa por Telegram |

- Los dos reinician solos (`restart: unless-stopped`) tras una caída o un reinicio
  del servidor.
- Un `shadow` **colgado** (vivo pero sin escribir) se mata a sí mismo con código 70
  a los `2 × SHADOW_SNAPSHOT_MINUTES + 5` = 35 minutos sin progreso y Docker lo
  levanta. Docker Compose no reinicia por un `HEALTHCHECK` en `unhealthy`; por eso
  la terminación la hace el propio proceso.
- Los datos viven en el host (`/var/lib/shadow`), no en el contenedor: reconstruir
  la imagen no los toca.
- El watchdog avisa al fallar, recuerda cada 6 h y avisa al recuperarse. Además avisa
  a las 00:10 UTC si falta el snapshot diario válido del día.

## 1. Servidor (lo haces tú)

Cualquier VPS Linux con 1 vCPU, 1 GB de RAM y 10 GB de disco alcanza (~4 MB/día de
libro). Elige una región **fuera de EE. UU.**: Binance restringe esas IP.
Ubuntu 24.04 LTS o Debian 12.

```bash
ssh root@IP_DEL_VPS
apt-get update && apt-get -y upgrade
apt-get install -y ca-certificates curl git chrony
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker chrony
```

### Reloj (NTP)

El snapshot de las 00:00:30 depende de la hora del servidor. Con el reloj corrido,
el "comparable directo con el backtest" deja de serlo. Verifica **una vez
instalado y después cada semana**:

```bash
chronyc tracking          # "System time" debe ser < 0.1 segundos
timedatectl               # "System clock synchronized: yes"
# Contra Binance (diferencia en ms; debe ser < 1000):
echo $(( $(curl -s https://data-api.binance.vision/api/v3/time | python3 -c 'import sys,json;print(json.load(sys.stdin)["serverTime"])') - $(date +%s%3N) ))
```

Si `synchronized` dice `no`, la sombra no debe correr: arregla el NTP antes. Esta
verificación es manual a propósito: automatizarla pide un endpoint nuevo en
`MarketData` con su respuesta grabada, y queda para un PR propio si hace falta.

## 2. Bot de Telegram (lo haces tú)

Sigue la sección «Crear el bot de Telegram para las alertas» del `README.md`
(token con @BotFather y `chat_id`). El token solo va en el `.env` del servidor.

## 3. Código y secretos (lo haces tú)

```bash
git clone https://github.com/KevinZorro/trading /opt/trading
cd /opt/trading
cp .env.example .env
chmod 600 .env
nano .env     # completa SHADOW_TELEGRAM_TOKEN y SHADOW_TELEGRAM_CHAT_ID
install -d -o 10001 -g 10001 -m 750 /var/lib/shadow
```

`.env` está en `.gitignore`, y nunca se sube al repositorio. `SHADOW_DATA_DIR` puede
quedar vacía: el compose la fija en `/data` dentro del contenedor.

## 4. Levantar

```bash
cd /opt/trading
GIT_SHA=$(git rev-parse HEAD) docker compose up -d --build
docker compose ps
docker compose logs -f shadow       # Ctrl+C para salir
cat /var/lib/shadow/heartbeat.json
```

Esperado: `shadow` y `watchdog` en `Up`, un `heartbeat.json` que se actualiza cada
15 minutos y, tras el primer snapshot, `snapshots-AAAA-MM-DD.jsonl` con la cabecera
(`adr_sha256`, `git_sha`).

## 5. Probar que la alerta llega (lo haces tú, una vez)

```bash
set -a; . ./.env; set +a
curl -s "https://api.telegram.org/bot${SHADOW_TELEGRAM_TOKEN}/sendMessage" \
  -d chat_id="${SHADOW_TELEGRAM_CHAT_ID}" -d text="prueba de alerta"
```

Para probar el watchdog completo: `docker compose stop shadow` y espera ~50 minutos;
debe llegar `ALERTA sombra: heartbeat atrasado mas de 45 min`. Luego
`docker compose start shadow`: llega "Sombra recuperada" (el hueco queda en los
datos y cuenta contra el 25/30; por eso esta prueba se hace **antes** de la
ventana de 30 días, no durante).

## Operación

- **Actualizar**: `git pull && GIT_SHA=$(git rev-parse HEAD) docker compose up -d --build`.
  Cambiar el código en medio de los 30 días queda registrado en `git_sha` de cada
  cabecera diaria.
- **Copia de seguridad** desde tu PC, una vez al día:
  `rsync -a root@IP_DEL_VPS:/var/lib/shadow/ ./sombra-datos/`
- **Lo que este despliegue no cubre**: si el servidor entero cae, el watchdog cae con
  él y Telegram no avisa. Revisa el estado de Telegram al menos una vez al día: la
  ausencia del recordatorio no es una señal.
