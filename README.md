# trading

Estudio comparativo de agentes de aprendizaje por refuerzo aplicados a trading: dos
agentes idénticos salvo por su espacio de estado, uno con señales de noticias y otro sin
ellas, evaluados en tres regímenes de riesgo y con capital variable.

Las reglas del proyecto (principios no negociables, invariantes consolidados y flujo de
trabajo) están en [`CLAUDE.md`](CLAUDE.md). Manda ese archivo.

## Puesta en marcha

```bash
uv sync                     # dependencias pinneadas desde uv.lock
uv run pytest               # suite completa
uv run pre-commit install   # hooks locales (ruff, mypy, anti-leakage, gitleaks)
```

### Escaneo de secretos (gitleaks)

El hook de pre-commit usa el binario de gitleaks del `PATH`; no hace falta Go. La
versión es **8.21.2**, la misma que pinnea el CI. Se descarga del release oficial y se
verifica su SHA256 antes de usarlo.

**Windows (PowerShell):**

```powershell
$v = "8.21.2"
$zip = "gitleaks_${v}_windows_x64.zip"
Invoke-WebRequest "https://github.com/gitleaks/gitleaks/releases/download/v$v/$zip" -OutFile $zip
# Tiene que imprimir True. Si imprime False, no sigas: el archivo no es el publicado.
(Get-FileHash $zip -Algorithm SHA256).Hash -eq "F238C85E5F47E18FAC779CE71EE11091CF70A0A8FB4415F165EFBA2800EEF133"
$dest = "$env:LOCALAPPDATA\Programs\gitleaks"
New-Item -ItemType Directory -Force $dest | Out-Null
Expand-Archive $zip -DestinationPath $dest -Force
# Agrega la carpeta al PATH del usuario (abre una terminal nueva despues).
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$dest", "User")
```

En una terminal nueva, `gitleaks version` debe imprimir `8.21.2`, y
`uv run pre-commit run gitleaks-system --all-files` debe terminar en `Passed`.

**Linux / macOS:** el mismo release, archivo `gitleaks_8.21.2_<os>_<arch>.tar.gz`,
con su SHA256 en `gitleaks_8.21.2_checksums.txt`; se descomprime `gitleaks` en un
directorio del `PATH`.

Sin el binario, el hook falla con "gitleaks: command not found" en vez de saltearse.
Usar `--no-verify` para esquivarlo deja la verificación solo en el CI, que escanea
toda la historia en cada push, pero para entonces el secreto ya está publicado.

## Las cuatro puertas

Las mismas en local y en CI. Si una falla localmente, va a fallar en el PR.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src                                    # modo strict
uv run pytest -m "not slow" --cov --cov-fail-under=85
```

## Anti-leakage

Los tests que verifican que no hay lookahead bias son la garantía central del estudio y
corren en un job propio del CI:

```bash
uv run pytest -m leakage -v
```

La suite entera corre **sin acceso a red**: `tests/conftest.py` bloquea los sockets y
falla con `NetworkAccessError`. Los datos se cargan de archivos locales o se generan con
`data.synthetic` a partir de una semilla fija.

## El agente PPO y su grupo opcional

`torch` y `stable-baselines3` viven en el grupo `rl`, que **`uv sync --frozen` no
instala**: la rueda de torch de PyPI arrastra el stack de CUDA (más de 3 GB) y sacaría
al job principal de CI de su presupuesto de 5 minutos en cada PR. El núcleo de
`agents/` —el protocolo de validación, el reporte por semillas, el walk-forward— no
depende de torch y se testea con políticas deterministas inyectadas.

```bash
uv sync --frozen --group rl          # torch + stable-baselines3 + sb3-contrib
```

### Qué cubre CI y qué no

La costura entre el entorno y SB3 **sí** está cubierta en cada PR, sin torch:
`tests/agents/test_ppo_contract.py` prueba `SB3Policy` contra un doble que implementa
la misma firma de `predict`. Verifica que la observación llegue al modelo tal como la
produjo el entorno, que la evaluación sea siempre determinista, que el adaptador no
altere la acción —el recorte lo hace el entorno y queda registrado— y que el reward
que el agente maximiza sea el retorno neto del equity del motor.

El riesgo de un doble es que se aleje del original, así que hay dos tests marcados
`rl` que comparan su firma contra la de SB3 y verifican que un modelo real devuelva
lo que el doble promete. Esos, y el smoke test que entrena de verdad, **no corren en
CI**: se saltean donde el grupo no está instalado.

Localmente:

```bash
uv sync --frozen --group rl
uv run pytest -m rl -v                       # contrato contra SB3 real + smoke de PPO
uv run pytest tests/agents/test_ppo_smoke.py -v
```

No se agregó un job de CI que instale torch de CPU porque el índice
`download.pytorch.org` no era alcanzable desde el entorno donde se desarrolló esto, y
un job que no se pudo verificar es peor que ninguno. Queda declarado en
`docs/adr/0003` en vez de disimulado.

### Correr el protocolo de validación

```bash
python -m agents.cli arm --level level_0 --bars 4000 --out results
python -m agents.cli assemble --out results          # aplica los criterios
```

`assemble` devuelve código distinto de cero si el protocolo se detuvo en algún nivel.

### Walk-forward sobre BTCUSDT diario (Tarea C3)

Implementa el ADR 0006, que está sellado. **Toca el test una sola vez.**

```bash
PYTHONPATH=src uv run --group rl python -m agents.cli walkforward --out results/c3
```

- No corre si el SHA256 del ADR 0006 en disco no coincide con el sellado
  (`agents.walkforward.ADR_SHA256`).
- Crea `results/c3/SEAL.json` antes del primer fold y aborta si ya existe.
- Cada par (fold, semilla) se agrega a `records.jsonl` al terminar, no al final.
- Si el proceso se cae, `--resume` continúa con la misma configuración sellada y no
  reevalúa ningún par ya escrito. Con otra configuración aborta.
- `records.jsonl` es el artefacto completo y va al GitHub Release. Al repositorio van
  `SEAL.json` y `summary.json`, que lleva el SHA256 de `records.jsonl`.

## Modo sombra (Etapa 3.5)

Diseño, seguridad y métrica pre-registrada en `docs/adr/0007-modo-sombra-y-testnet.md`.
La sombra lee datos públicos de mainnet y **no envía órdenes**; el código que envía
órdenes solo acepta hosts de testnet (`live.guard`). El proceso, el
despliegue y el Agente A congelado ya están en `main`; ver la matriz al final.

### Configuración

Todo por variables de entorno con prefijo `SHADOW_`. Copia `.env.example` a `.env`
(que está en `.gitignore`) y complétalo. **Nunca pongas un token en un archivo
versionado.**

| Variable | Por defecto | |
|---|---|---|
| `SHADOW_DATA_DIR` | — | Obligatoria. Donde se persisten los snapshots |
| `SHADOW_SYMBOL` | `BTCUSDT` | |
| `SHADOW_ORDER_SIZES_USDT` | `10,100,1000` | Tamaños de las órdenes hipotéticas |
| `SHADOW_SNAPSHOT_MINUTES` | `15` | Debe dividir a 60 |
| `SHADOW_DEPTH_LIMIT` | `1000` | Niveles del libro por snapshot |
| `SHADOW_HEARTBEAT_MAX_AGE_MINUTES` | `45` | Minutos sin escribir antes de alertar |
| `SHADOW_MARKET_DATA_URL` | `https://data-api.binance.vision` | Solo HTTPS |
| `SHADOW_TELEGRAM_TOKEN` | vacío | Secreto. Va junto con el chat_id |
| `SHADOW_TELEGRAM_CHAT_ID` | vacío | |
| `SHADOW_DEADMAN_URL` | vacío | Secreto. Ping de healthchecks.io cuando todo está sano |

Sin las dos variables de Telegram la sombra corre igual, pero sin alertas.
Despliegue en VPS con Docker, watchdog y NTP: `docs/despliegue-sombra.md`.

### Crear el bot de Telegram para las alertas

1. En Telegram, abre una conversación con **@BotFather** (verifica la marca azul de
   cuenta oficial) y envía `/newbot`. Elige un nombre y un usuario que termine en
   `bot`.
2. BotFather responde con el **token**, de la forma `123456789:AA...`. Es una
   credencial: quien lo tenga puede escribir como tu bot. Guárdalo directamente en
   `.env` como `SHADOW_TELEGRAM_TOKEN=...`, y no lo pegues en issues, PRs ni chats.
3. Abre una conversación con tu bot nuevo y envíale cualquier mensaje. Un bot no
   puede escribirte hasta que tú le escribas primero.
4. Obtén tu `chat_id`. Desde una terminal, con el token en la variable de entorno
   (así no queda en el historial):

   ```bash
   set -a; . ./.env; set +a
   curl -s "https://api.telegram.org/bot${SHADOW_TELEGRAM_TOKEN}/getUpdates"
   ```

   En PowerShell:

   ```powershell
   $t = (Select-String -Path .env -Pattern '^SHADOW_TELEGRAM_TOKEN=(.*)$').Matches[0].Groups[1].Value
   Invoke-RestMethod "https://api.telegram.org/bot$t/getUpdates" | ConvertTo-Json -Depth 6
   ```

   El número en `result[0].message.chat.id` es el `chat_id`. Guárdalo como
   `SHADOW_TELEGRAM_CHAT_ID=...` en `.env`.
5. Si el token se filtra, en @BotFather usa `/revoke` para invalidarlo y generar
   otro.

## Estado y matriz de trazabilidad

Etapas 1 (simulador, baselines y métricas), 2 (entorno Gymnasium) y 3 (Agente A, protocolo
de validación y Tarea C) cerradas. La Etapa 3.5 (infraestructura en vivo mínima) está
cerrada desde 2026-10-19; la sombra de costos sigue corriendo su ventana de 30 días. La
Etapa 4 está en curso. Las reglas y la lista de
etapas están en [`CLAUDE.md`](CLAUDE.md); aquí se rastrea cada hito hasta su PR y su
evidencia en el repositorio.

Estado al 2026-10-22. Los números de PR son de GitHub; "mergeado" quiere decir que
está en `main`. Cuando cambia un hito, se actualiza esta tabla en el mismo PR.

### Etapas 1 a 3: simulador, entorno y Agente A

| Hito | Origen | PR | Estado | Evidencia |
|---|---|---|---|---|
| CI con job propio de anti-leakage | `CLAUDE.md` (invariantes de CI) | #1 | mergeado | `.github/workflows/ci.yml`, marcador `leakage` |
| Métricas sobre ambas series de equity | Etapa 1 | #2, #3 | mergeado | `src/eval/metrics.py`, `tests/eval/` |
| Arquitectura para ejecución real (`ExecutionVenue`, `Clock`, `client_order_id`, `risk/`) | ADR 0001, Etapa 1 | #4 | mergeado | `src/sim/venue.py`, `src/sim/clock.py`, `src/sim/ids.py`, `src/risk/layer.py`, `tests/sim/test_venue.py` |
| Entorno Gymnasium con anti-leakage | Etapa 2 | #5 | mergeado | `src/envs/`, `tests/envs/test_leakage.py` |
| Fixtures sintéticos con óptimo calculable | ADR 0002 | #6 | mergeado | `src/data/fixtures.py`, `tests/data/test_fixtures_optimum.py` |
| Agente A (PPO) y protocolo de validación | ADR 0003 | #7 | mergeado | `src/agents/protocol.py`, `src/agents/ppo.py`, `tests/agents/test_protocol.py` |
| Varianza de mercado separada de la de entrenamiento | ADR 0004 (Tarea A) | #8 | mergeado | `src/eval/distribution.py`, `tests/eval/test_distribution.py` |
| Contrato del adaptador de SB3, sin torch | ADR 0003 (Tarea B) | #9 | mergeado | `tests/agents/test_ppo_contract.py` |
| Nivel 4 partido en 4a y 4b, reportado como par | ADR 0005 (Tarea D) | #10, #12 | mergeado | `docs/adr/0005-nivel-4-partido-en-dos-hipotesis.md`, `results/multipath_level_4b.json` |
| Nivel 3 con 10 caminos | ADR 0004 (Tarea E) | #11 | mergeado | `results/multipath_level_3_mlp.json`, `results/multipath_level_3_lstm.json` |
| Nuevo orden de etapas y regla de tamaño de PR | `CLAUDE.md` | #13 | mergeado | sección "Etapas" y "Tamaño de PR" de `CLAUDE.md` |
| Dataset BTCUSDT con manifest y validación | Tarea C1 | #14, #15 | mergeado | `datasets/`, `src/data/binance.py`, `src/data/manifest.py` |
| Walk-forward sellado sobre BTCUSDT diario | ADR 0006 (C2) | #16, #17, #18 | mergeado | `docs/adr/0006-walk-forward-sobre-btcusdt-diario.md`, `src/agents/walkforward.py`, `src/eval/walkforward.py` |
| Resultados de la Tarea C3: no supera a buy-and-hold, N = 1 | ADR 0006 (C3) | #19, #21 | mergeado | `results/c3/REPORTE.md`, `results/c3/SEAL.json`, `results/c3/MANIFEST.json` |
| Un delta numéricamente cero no es una orden | `CLAUDE.md` (anti-patrones) | #20 | mergeado | `src/sim/sizing.py`, `tests/sim/test_sizing.py` |

### Etapa 3.5: sombra sobre mainnet y testnet

La partición por PR es la de la sección 7 del
[ADR 0007](docs/adr/0007-modo-sombra-y-testnet.md) (pre-registro, no se edita). La
sombra valida costos; testnet valida mecánica, nunca costos.

| Hito | Origen | PR | Estado | Evidencia |
|---|---|---|---|---|
| PR 1: ADR 0007, `ShadowConfig`, guard de testnet y logs | ADR 0007 §7 | #22 | mergeado | `docs/adr/0007-modo-sombra-y-testnet.md`, `src/live/config.py`, `src/live/guard.py`, `tests/live/test_guard.py` |
| CLAUDE.md partido por módulo | `CLAUDE.md` | #23 | mergeado | `src/*/CLAUDE.md`, `scripts/verificar_particion_claude_md.py` |
| PR 2: `MarketData` REST de solo lectura y grabación de respuestas | ADR 0007 §7 | #24 | mergeado | `src/live/marketdata.py`, `tests/live/respuestas/`, `tests/live/test_marketdata.py` |
| `min_notional` medido y `InstrumentSpec` desde `exchangeInfo` | ADR 0008 | #25 | mergeado | `docs/adr/0008-instrument-spec-desde-exchangeinfo.md` |
| PR 3: `walk_book` y `model_cost` puras | ADR 0007 §7 | #26 | mergeado | `src/live/shadow_cost.py`, `tests/live/test_shadow_cost.py`, `tests/live/test_shadow_cost_leakage.py` |
| Hook de git-guard versionado | flujo de trabajo | #27 | mergeado | `.claude/hooks/git-guard.sh` |
| PR 4: proceso de la sombra de costos | ADR 0007 §4 y §7 | #28 | mergeado | `src/live/shadow.py`, `src/live/schedule.py`, `src/live/store.py`, `tests/live/test_shadow.py` |
| PR 6: despliegue con Docker, watchdog y autoterminación | ADR 0007 §7 | #29 | mergeado | `Dockerfile`, `docker-compose.yml`, `docs/despliegue-sombra.md`, `src/live/watchdog.py`, `tests/live/test_despliegue.py` |
| PR 5a: `ShadowVenue` y `FillStore` | ADR 0007 §7 | #30 | mergeado | `src/live/venue.py`, `src/live/fillstore.py`, `tests/live/test_venue.py`, `tests/live/test_venue_leakage.py` |
| PR 5b: Agente A congelado, una política por semilla | ADR 0009 | #31, #34, #35 | mergeado | `docs/adr/0009-agente-a-congelado-todas-las-semillas.md`, `src/live/frozen.py`, `frozen/agente_a/`, `tests/live/test_frozen_versionado.py` |
| PR 7a: capa HTTP firmada de testnet y grabador | ADR 0007 §7 | #32, #37, #38 | mergeado | `src/live/testnet_http.py`, `scripts/grabar_respuestas_testnet.py`, `tests/live/respuestas/testnet/` |
| PR 5c: Agente A congelado conectado a la sombra | ADR 0009 | #33 | mergeado | `src/live/agent_step.py`, `src/live/shadow_agents.py`, `tests/live/test_shadow_agents.py` |
| Imagen Docker con torch solo CPU | ADR 0009 | #36 | mergeado | `Dockerfile`, `tests/live/test_despliegue.py` |
| PR 7b: `BinanceTestnetVenue`, idempotente por `client_order_id` | ADR 0007 §7 | #39 | mergeado | `src/live/testnet_venue.py`, `tests/live/test_testnet_venue.py`, `tests/live/test_testnet_venue_leakage.py` |
| Grabador de errores de testnet (-1013, -2010, -1022) | PR 7a | #40 | mergeado | `scripts/grabar_respuestas_testnet.py`, `tests/live/test_grabador_testnet.py` |
| PR 8a: grabador del WebSocket de testnet y guard `wss` | ADR 0007 §7 | #41 | mergeado | `scripts/grabar_ws_testnet.py`, `tests/live/test_grabador_ws_testnet.py`, `tests/live/respuestas/testnet_ws/` |
| Grabaciones reales de -1013, -2010 y -1022 | PR 7a | #42 | mergeado | `tests/live/respuestas/testnet/err_*.json` |
| Venue de testnet maneja -1013 y -2010 como rechazo con motivo; -1022 se propaga | PR 7b | #45 | mergeado | `src/live/testnet_venue.py`, `tests/live/test_testnet_venue.py` |
| PR 8b: WebSocket como observador (comisión real, fills tras caída) | ADR 0007 §7 | #44 | mergeado | `src/live/testnet_ws.py`; lo grabado en 8a: `tests/live/respuestas/testnet_ws/` |
| Persistir el estado de `RiskLayer` ante reinicio | deuda de PR 5b | sin PR | pendiente | `src/risk/layer.py` no persiste estado |

### Sombra de costos en mainnet

| Hito | Estado |
|---|---|
| Despliegue | Hecho el 2026-10-19 en una VM del proveedor de nube; el proceso corre en Docker (`docs/despliegue-sombra.md`). |
| Reloj de 30 días | **En marcha.** Primer snapshot diario válido: 2026-10-20 00:00:30 UTC, con los diez Agentes A cargados sin errores (semillas 11, 23, 37, 41, 59, 67, 73, 89, 97 y 101). Ventana: 2026-10-20 a ~2026-11-18. |
| Evaluación de la predicción | **Abierta.** Sin evaluar hasta completar la ventana. |
| Veredicto | Menos de 25 de los 30 snapshots válidos significa **NO EVALUABLE** (ADR 0007 §6). La predicción sobre el spread está pre-registrada ahí y no se edita. |

### Etapa 4 y prerrequisitos de la Etapa 5

El plan de la etapa vive fuera del repositorio, en los archivos del proyecto
(`etapa4-noticias/PLAN.md`): fuente GDELT GKG 2.0 crudo, tema `ECON_BITCOIN`, tono de
GDELT + FinBERT, `pub_time` = rótulo del lote + 30 min. La pista L es el grabador en
vivo y la pista P, el presupuesto de entrenamiento. Aquí se rastrean los PRs.

| Hito | Origen | PR | Estado | Evidencia |
|---|---|---|---|---|
| L1: grabador en vivo de lotes GDELT, servicio aparte en la VM | Plan Etapa 4, pista L | #47 | mergeado | `src/data/gdelt.py`, `src/live/news.py`, `docs/despliegue-noticias.md`, `tests/live/test_news.py`, `tests/live/test_news_leakage.py` |
| L1b: sondeo paralelo de lotes pendientes (un lote tardío no retiene a los siguientes) | Hallazgo de la VM, 2026-10-20 | #58 | mergeado | `src/live/news.py`, `tests/live/test_news.py` |
| N1: esquema, validación y carga de noticias con la regla de alineación | Plan Etapa 4, pista N | #49 | mergeado | `src/data/news.py`, `tests/data/test_news.py` |
| N2: descarga histórica de GDELT al formato del grabador, reanudable y con MD5 | Plan Etapa 4, pista N | #51 | mergeado (la descarga completa sigue en curso) | `src/data/gdelt_history.py`, `scripts/descargar_gdelt.py`, `docs/descarga-gdelt.md`, `tests/data/test_gdelt_history.py`, `tests/data/test_gdelt_history_leakage.py` |
| P1: grilla del presupuesto de entrenamiento sobre 4b, declarada antes de correrla | `src/agents/CLAUDE.md` (deuda antes de la Etapa 5), ADR 0005 | #48 | mergeado | `docs/adr/0010-presupuesto-de-entrenamiento-sobre-4b.md` |
| P2: calibración y re-corrida del protocolo con el presupuesto elegido | ADR 0010 | #50, #52, #53 | script mergeado (la corrida y la confirmación siguen en curso; #52 importa torch antes que numpy en Windows y #53 tolera CRLF en el SHA256 del ADR) | `src/agents/calibration.py`, `docs/calibracion-presupuesto.md`, `tests/agents/test_calibration.py` |
| P2b: diagnóstico de por qué 4b no pasa a ningún presupuesto (`calibrate diagnose`) | ADR 0010 §4 | #57 | mergeado; corrido (ningún grupo solo explica la rotación) | `src/agents/diagnosis.py`, `tests/agents/test_diagnosis.py`, `results/calibracion/summary.json` |
| P3: segunda ronda de calibración, escalera de regularización declarada antes de correrla | ADR 0010 §4, diagnóstico de #57 | #60 | mergeado; escalera elegida (R1 a R3, sin penalización); sin correr | `docs/adr/0011-segunda-ronda-de-calibracion-regularizacion.md` |
| P3b: script de la escalera R1 a R3 (`calibrate ladder`), pareado contra G1 | ADR 0011 | #61 | mergeado; corrido: R1 y R2 fallan, **R3 pasa 4b** (`C*` = R3) | `src/agents/regularization.py`, `docs/calibracion-presupuesto.md`, `tests/agents/test_regularization.py` |
| P3c: confirmación de `C*` = R3 con el protocolo completo, reanudable, y re-diagnóstico | ADR 0011 §5 y §6 | #62 | mergeado; **corrido: protocolo completo en verde con R3** (4b y el par PASS por primera vez). `C*` = R3 a 120 000 timesteps queda fijo para el estudio | `src/agents/confirmation.py`, `results/regularizacion/summary.json`, `docs/calibracion-presupuesto.md`, `tests/agents/test_confirmation.py` |
| P3d: registro del resultado de la confirmación (resumen, deuda de la Etapa 5 saldada) | ADR 0011 §5 regla 3 | (este PR) | en revisión | `results/confirmacion/summary.json`, `docs/calibracion-presupuesto.md`, `src/agents/CLAUDE.md`, `CLAUDE.md` |

### Etapas siguientes

| Etapa | Estado |
|---|---|
| 4. Pipeline de noticias con validación point-in-time | en curso: mergeados L1 (#47), ADR 0010 (#48), N1 (#49), P2 (#50, #52, #53), P2b (#57) y N2 (#51). Ver la tabla de la Etapa 4 |
| 5. Agente B y comparación controlada contra A | sin empezar |
| 6. Barrido de regímenes y capital, `PortfolioSimulator` multi-activo | sin empezar |
| 7. Paper trading en vivo (3 a 6 meses antes de capital real) | sin empezar; hereda los datos de la 3.5 |
