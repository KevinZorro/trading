# CLAUDE.md — Proyecto: RL Trading con Señales de Noticias

## Qué es esto

Estudio comparativo de agentes de aprendizaje por refuerzo aplicados a trading.
Se comparan **dos agentes idénticos** salvo por su espacio de estado:

- **Agente A (baseline):** solo features de precio/volumen/técnicos.
- **Agente B (news-aware):** los mismos features + señales derivadas de noticias y macro.

Se evalúan en **3 regímenes de riesgo** (bajo / medio / alto) y con **capital variable**
para determinar a partir de qué monto los costos de transacción destruyen cualquier ventaja.

El objetivo es un resultado honesto sobre si las noticias aportan alfa, no maximizar una
métrica de backtest. Un resultado negativo bien medido es un resultado válido y esperado.

## Objetivo a largo plazo: ejecución real

Este sistema debe poder operar con dinero real en el futuro. No se construye infraestructura
de trading en vivo todavía, pero **toda decisión de diseño preserva la paridad
backtest-live**: el mismo código de estrategia y de gestión de riesgo debe poder correr
contra el simulador y contra un broker real, sin ramas condicionales por entorno.

Consecuencias vigentes desde ya:

- Ninguna llamada directa a `datetime.now()`. El tiempo entra por un `Clock` inyectable.
- Los identificadores de orden los genera el emisor (`client_order_id` único), no el venue.
  Sin esto, un timeout de red en vivo duplica posiciones al reintentar.
- La estrategia nunca asume que su memoria del estado es la verdad. La posición autoritativa
  la reporta el venue, y el arranque siempre reconcilia contra él.
- La capa de riesgo vive fuera del agente y se aplica idénticamente en simulación y en vivo.
  Un agente RL ante un estado fuera de distribución puede decidir cualquier cosa; el límite
  de riesgo es lo que separa una mala racha de una pérdida total.
- Nada de símbolos, rutas o parámetros hardcodeados: todo por configuración.

Gate obligatorio antes de cualquier capital real: **paper trading en vivo de 3 a 6 meses**,
comparando ejecución real contra lo que el simulador predijo para las mismas señales. Si el
slippage real difiere materialmente del simulado, todos los resultados del estudio se releen
con ese factor.

## Principios no negociables

1. **Sin lookahead bias.** Ningún dato disponible en `t` puede depender de información
   publicada después de `t`. Aplica a precios, noticias, fundamentales y a los modelos de
   NLP usados para extraer sentimiento.
2. **Point-in-time timestamps.** Las noticias usan hora de publicación original, nunca de
   última edición. Si la fuente no distingue, se documenta como limitación.
3. **Costos dentro de la recompensa.** Comisiones, spread y slippage se restan en el reward,
   no en post-procesamiento.
4. **Ejecución nunca en la misma barra que generó la señal.** Decisión con close de `t`,
   ejecución al open de `t+1`.
5. **Nunca reportar el mejor seed, y nunca confundir las dos varianzas.** Toda métrica
   se reporta como distribución sobre >= 10 semillas: mediana, p25, p75, min, max. Pero
   diez semillas sobre **un** camino miden varianza de *entrenamiento* —cuánto cambia el
   resultado al reentrenar sobre la misma serie—, que no es varianza de *mercado*.
   - **Sobre fixtures sintéticos, N caminos independientes × M semillas**, con las dos
     varianzas reportadas por separado (`eval.distribution.decompose_variance`). Los
     caminos se muestrean cambiando la semilla del proceso; las semillas del agente son
     otro eje y no lo sustituyen. Un contraste contra cero usa el error estándar **entre
     caminos**: hay N observaciones independientes, no N×M.
   - **Sobre datos reales, N = 1 por construcción** —la historia es un solo camino— y se
     declara como limitación en cada reporte. Esa asimetría es la razón de fondo por la
     que el walk-forward fuera de muestra es la única defensa que queda ahí: no se puede
     medir el error de muestreo, solo acotarlo con más ventanas.

   Medido: en el nivel 4 el exceso del agente fue +0.164 con las diez semillas de acuerdo
   entre sí, sobre un camino donde el baseline rindió −0.013. El desvío de ese mismo
   baseline entre 20 caminos independientes es **0.781**. El "hallazgo" era cinco veces
   más chico que la dispersión del sorteo, y ninguna cantidad de semillas lo habría
   revelado.
6. **Todo experimento incluye baselines**: buy-and-hold, aleatorio, cruce de medias.
7. **Reproducibilidad total.** Seeds fijos, dependencias pinneadas, config serializada junto
   a cada resultado.

## Convención temporal de las barras

Decidida al versionar el primer dataset real (Tarea C1). Rige para todo el repositorio.

- **El `timestamp` de una barra es su cierre** (`close_time`), no su apertura. Todo lo
  que la fila contiene —OHLC, volumen— se conoce recién en `timestamp`, así que la fila
  entera es point-in-time. Binance rotula por la apertura: `data.binance` escribe
  `timestamp = open_time + intervalo` y conserva `open_time` como columna aparte. Los
  generadores sintéticos y los fixtures arrancan en `21:00Z` (cierre de la sesión de
  EE.UU.) y `WeekdayCalendar` fecha cada sesión por su hora de cierre: la misma
  convención.
- **Regla de alineación de la Etapa 4.** Una noticia es usable en la decisión de la
  barra `t` **si y solo si `pub_time <= close_time[t]`**, con `close_time[t] =
  timestamp[t]`. La decisión se ejecuta en el open de `t+1`. `pub_time` es la hora de
  publicación original (principio 2); si se modela latencia de procesamiento, se suma
  a `pub_time` antes de comparar, nunca se resta a `close_time`.
- **El motor colapsa la barra en un instante.** `Simulator` avanza el reloj a
  `timestamp[t]` al abrir la barra `t`, y con esa misma marca registra el fill al open
  de `t` y la decisión al close de `t`. Con la convención de cierre, `timestamp_decision`
  es exacto y **`timestamp_fill` llega una barra tarde**: el fill real ocurre en
  `close_time[t-1]` en un mercado continuo, o en la apertura de la sesión en uno con
  gap. La contabilidad es por índice y no se afecta; lo que se afecta es comparar horas
  de fill contra un broker (Etapa 7), que debe usar la hora que reporta el broker.
- **El día de riesgo queda corrido una barra en intradía.** `risk.layer._dia` agrupa por
  la fecha UTC de `timestamp`; con la convención de cierre, la barra que cierra a las
  00:00 cae en el día siguiente al que pertenece. Con barras diarias no cambia nada (una
  barra por día). Se corrige antes de correr la capa de riesgo sobre barras horarias.

## Invariantes de calidad y CI

Consolidados al montar el pipeline. No los relajes para que algo pase.

- **La suite corre sin red.** `tests/conftest.py` bloquea `socket.connect`,
  `connect_ex` y `create_connection`, y lanza `NetworkAccessError`. Un test que
  necesite datos los versiona o los genera con `data.synthetic` y una semilla fija.
  Un test que descarga de una API pasa hoy y en seis meses falla, o -peor- sigue
  pasando con datos distintos y vuelve irreproducible un resultado del estudio.
- **Los tests de anti-leakage llevan `@pytest.mark.leakage`** (a nivel de módulo con
  `pytestmark`) y corren en un job propio del CI. Cuando la Etapa 2 agregue tests de
  anti-leakage sobre `envs/`, márcalos: el job verifica que recogió al menos uno, así
  que un marcador mal escrito falla en vez de quedar en verde sin correr nada.
- **Tests caros: `@pytest.mark.slow`**, van a un job aparte. El job principal se
  mantiene por debajo de los 5 minutos.
- **Cuatro puertas obligatorias**, idénticas en local (pre-commit) y en CI:
  `ruff check`, `ruff format --check`, `mypy --strict src`, y cobertura >= 85%.
  Matriz de Python 3.11 y 3.12.
- **`pytest.raises` siempre con `match`.** Un `raises(ValueError)` a secas queda en
  verde aunque la falla venga de un motivo distinto del que el test pretendía cubrir.
- **Nada de `np.ndarray` sin parámetros en firmas.** Usa `FloatArray` y `TimeArray`
  de `data.schema`: `np.ndarray` pelado hace que mypy trate como `Any` todo lo que
  sale de la serie, que es precisamente lo que hay que tipar.
- **Los `Protocol` declaran `name` como property de solo lectura**, no como
  `name: str`. Las implementaciones son dataclasses frozen y un atributo de clase
  exige que sea asignable; con `name: str` ningún modelo de costo satisfacía
  formalmente su propio Protocol.
- **`uv.lock` se versiona** y el CI corre con `uv sync --frozen` más `uv lock --check`.
  El bytecode compilado no se versiona (`.gitignore`).
- `PYTHONHASHSEED=0` en CI: sin eso el orden de iteración de sets varía entre
  corridas y un test puede pasar o fallar según el día.

## Seguridad de la ejecución real (Etapa 3.5 en adelante)

No negociable. Se aplica desde el primer PR que toque red.

- **Mainnet es de solo lectura por construcción.** El código que envía órdenes solo
  puede apuntar a URLs de testnet: un guard lanza error si una URL de envío de órdenes
  no es de testnet, y hay test que lo fija. No es una convención, es una imposibilidad.
- **Credenciales por variables de entorno o `.env`, nunca en el repo.** `.env` está en
  `.gitignore`, y el escaneo de secretos (gitleaks) corre en pre-commit y en CI.
- **Las API keys de mainnet, cuando existan, se crean SIN permiso de retiro.** Queda
  escrito en el ADR de la 3.5.
- **Los tests que tocan red llevan marcador y quedan fuera del CI.** El guard de red de
  `tests/conftest.py` se mantiene. La lógica se testea contra respuestas grabadas y
  versionadas.
- **Persistencia incremental** (append) en todo proceso de larga duración. Nunca solo
  al final: un proceso que muere a las tres semanas no puede llevarse las tres semanas.

## Anti-patrones prohibidos

- Normalizar con estadísticas calculadas sobre todo el dataset. Se ajusta solo en train.
- Precios crudos como feature. Usar retornos logarítmicos.
- Train/test split simple. Se usa walk-forward con ventanas rodantes.
- `ffill` a través de la frontera train/test.
- Ejecutar al precio medio ignorando el spread.
- Iterar hiperparámetros contra el test. El test se toca una vez, al final.
- Universos construidos con la lista actual de constituyentes (survivorship bias).
- Baselines que se autocensuran pre-redondeando a cero: hace indistinguible "no quiso" de
  "no pudo". Se envía la orden y el rechazo queda en el log.

## Stack

Python 3.11+, `uv`, `gymnasium`, `stable-baselines3` (PPO/SAC), `polars`/`pandas`,
`pytest`, `hydra` o dataclasses + YAML, MLflow o wandb.

## Estructura

```
src/
  data/       # instruments, schema, validation, calendars, adjustments, loaders,
              # synthetic, fixtures (validacion con optimo conocido)
  sim/        # engine, costs, orders, portfolio, view, venue, clock, ids, gate, sizing
  eval/       # metrics, report, trades, walkforward, distribution (semillas + DSR)
  risk/       # RiskLayer independiente del agente
  features/   # technical (RSI, MACD, ATR, Bollinger), scaler fit-en-train
  envs/       # trading_env, observation, rewards (wrapper delgado, sin lógica propia)
  agents/     # baselines, policy, runner, ppo (grupo opcional), protocol, cli
  live/       # sombra sobre mainnet (solo lectura) y testnet: config, guard, logs
  configs/
tests/
docs/adr/
notebooks/    # solo exploración
```

**Invariantes por módulo.** Los de cada módulo viven en su propio `CLAUDE.md`:
`src/data/`, `src/sim/`, `src/risk/`, `src/eval/`, `src/envs/` y `src/agents/`.
Se cargan solos al leer un archivo de esa carpeta. **Antes de cambiar un módulo,
o sus tests en `tests/<módulo>/`, lee su `CLAUDE.md`**: los tests no están bajo
`src/` y no lo cargan por sí solos. Se partieron con
`scripts/verificar_particion_claude_md.py`, que comprueba que ninguna regla se
perdió ni quedó duplicada.

## Etapas

1. **Simulador + baselines + `eval/`.** Cerrada. `data/`, `sim/` y `eval/`, CI en verde.
   Refactor de arquitectura para ejecución real aplicado sobre esta base: `ExecutionVenue`,
   `Clock`, `client_order_id` y `risk/`. 330 tests.
2. **Entorno Gymnasium** como wrapper delgado, con tests de anti-leakage. Cerrada.
   `envs/`, `features/` y `sim/sizing.py`; `drive()` como costura. 478 tests.
3. **Agente A** (solo precio), un activo, un régimen. ¿Supera buy-and-hold neto de costos?
   Fixtures sintéticos con señal conocida y óptimo calculable (`data/fixtures.py`) y
   PPO con el protocolo de validación (`agents/`). Protocolo cerrado: 0–2 PASS, 3
   MEASURED (memoriza, 10/10 caminos), 4a PASS, 4b FAIL, par PARTIAL. **Tarea C
   cerrada** (ADR 0006, `results/c3/REPORTE.md`): NO SUPERA a buy-and-hold, predicción
   del ADR 0004 NO CONCLUYENTE, con política subentrenada y N = 1. Corrió con el sizer
   anterior a la regla del cero numérico: no es comparable con corridas posteriores.
3.5. **Infraestructura en vivo mínima y Agente A en sombra.** Dos validaciones con
   propósitos distintos, que no se mezclan:
   - **Sombra sobre mainnet → valida COSTOS.** Solo lectura, sin API key, sin órdenes.
     Recorre la profundidad L2 real para órdenes hipotéticas de varios tamaños y
     compara contra lo que predice el modelo de costos del simulador en ese mismo
     momento. Corre además el Agente A sobre barras diarias reales. Métrica principal:
     distribución de `costo_real / costo_simulado` por tamaño de orden.
   - **Testnet → valida MECÁNICA, nunca costos.** Su libro tiene liquidez artificial y
     precios que pueden divergir de mainnet. `BinanceTestnetVenue` implementa
     `ExecutionVenue`: envío y cancelación, `client_order_id` idempotente ante
     reintentos reales, reconciliación al arranque, recuperación tras caída del
     proceso y del WebSocket.

   Motivo del adelanto: "¿el simulador dice la verdad sobre los costos?" no depende de
   las noticias. Adelantarla convierte tiempo de calendario en evidencia.
4. **Pipeline de noticias** con validación point-in-time estricta. Se desarrolla **en
   paralelo** con la 3.5 mientras ésta acumula datos.
   **Antes de la Etapa 5:** calibrar el presupuesto de entrenamiento sobre el fixture
   4b (ver *Invariantes del Agente A*), con la grilla declarada en un ADR antes de
   correrla.
5. **Agente B** y comparación controlada contra A.
6. **Barrido** de regímenes y capital. `PortfolioSimulator` multi-activo.
7. **Paper trading** en vivo. No arranca de cero: hereda los meses de datos de la 3.5.

## Métricas

Retorno total, CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover, win rate,
profit factor, y costos desglosados como métrica de primer nivel. Sobre **ambas** series de
equity, con la brecha entre ellas como métrica explícita (mide fricción no realizada).

Para significancia ante comparaciones múltiples: **Deflated Sharpe Ratio**
(Bailey & López de Prado), no p-values ingenuos.

## Flujo de trabajo

Para cada feature, sin excepción:

1. Rama desde `main`: `feat/<nombre>` o `fix/<nombre>`
2. Implementación con sus tests
3. Verificación local en verde
4. Commits convencionales (`feat:`, `fix:`, `test:`, `chore:`)
5. Push y PR (con `gh` CLI o la API de GitHub)
6. Descripción del PR: qué resuelve, decisiones de diseño, `ASSUMPTION`s introducidos,
   qué tests lo cubren
7. Esperar CI verde
8. **Reportar el PR. No mergear sin aprobación humana.**

### Tamaño de PR

El revisor tiene unas 2 horas por semana. **Cada PR se revisa en 30–40 minutos**:

- Como guía, **no más de ~500 líneas de código de producción** (`src/`) por PR. Los tests
  van aparte de esa cuenta, pero también se leen: que no la dupliquen sin motivo.
- **Los artefactos de resultados no se commitean completos**: solo resúmenes y
  manifests. El manifest lleva el SHA256 de cada artefacto completo, para que un
  resumen sea verificable contra los datos que lo produjeron. Los JSON que ya están en
  `results/` se conservan como registro histórico —`assemble` los lee y borrarlos
  haría irreproducible el veredicto desde un clon—; la regla aplica hacia adelante.
- Los **datasets** no son artefactos de resultados: se versionan con su manifest
  (SHA256, fuente, fecha, script reproductor), como exige la reproducibilidad.
- **Si una tarea no cabe, se parte y la partición se propone antes de empezar.**

## Estilo de trabajo esperado

- Tests junto con la lógica, especialmente en `sim/`, `data/` y `risk/`.
- Los tests de anti-leakage son tan importantes como los funcionales.
- Tests con oráculos calculados a mano, nunca con la propia implementación como referencia.
- Código explícito y aburrido sobre abstracciones ingeniosas.
- Supuestos sobre el mercado documentados con comentarios `ASSUMPTION:`.
- **Si una instrucción introduce leakage, un backtest optimista o una premisa falsa, dilo
  antes de implementarla.** En la Etapa 1 se corrigió una afirmación errónea sobre el
  drawdown de la serie de liquidación en vez de forzar el test para que pasara. Ese es el
  comportamiento correcto.
