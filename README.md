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
órdenes solo acepta hosts de testnet (`live.guard`). En este PR solo existe la
configuración: el proceso llega en los PRs siguientes.

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

Sin las dos variables de Telegram la sombra corre igual, pero sin alertas.

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

## Estado

Etapas 1 (simulador, baselines y métricas) y 2 (entorno Gymnasium) cerradas.
Ver la sección "Etapas" de `CLAUDE.md`.
