# ¿Aportan alfa las noticias a un agente de aprendizaje por refuerzo? Un estudio pre-registrado sobre BTCUSDT con costos reales y paridad backtest-vivo

**Autor:** Kevin David Zorro Hernández
**Repositorio:** <https://github.com/KevinZorro/trading>
**Versión:** 1.0, 2026-11-10. Sección 6.6 (sombra sobre mainnet) **provisional**: la
ventana de 30 días cierra el 2026-11-18 y se evalúa una sola vez (ADR 0007 §6).
**Tipo de resultado:** negativo, pre-registrado.

---

## Abstract (English)

We test whether news-derived signals add alpha to a deep reinforcement learning (RL)
trading agent on daily BTCUSDT. Two Proximal Policy Optimization (PPO) agents are
identical except for their observation space: Agent A sees 15 price, volume and
technical features; Agent B sees the same 15 plus 7 news features built from the GDELT
Global Knowledge Graph (2.74 M articles, 2016–2026) and FinBERT headline sentiment. A
third arm, a *placebo*, receives the same news features shifted by 364 days. Design,
features, latency rule, decision criterion and number of trials were pre-registered
in an Architecture Decision Record and sealed by hash before any training on real
data. Transaction costs (Corwin–Schultz spread, 10 bps commission, square-root
impact) are charged inside the reward; decisions use the close of day *t* and fill at
the open of *t+1*. Evaluation is walk-forward over 11 out-of-sample folds
(2020-05 to 2026-05) with 10 seeds per arm (330 trainings) and the test set was
touched exactly once. Before the real run, the training budget was calibrated on
synthetic fixtures with a known optimum, and a positive control showed that the
pipeline lets the agent learn an exogenous signal when one exists (capture 0.925,
paired *t* = 14.8). **Result: the pre-registered verdict is "no contribution (flat)".**
B beats A and the placebo in median (+0.185 and +0.326 log excess) but the Deflated
Sharpe Ratio is 0.146 and 0.200 against a 0.95 threshold. A perturbation diagnosis
shows B does not use the news inputs (action change 0.011 when news is frozen, below
the 0.02 rule, versus 0.105 for technicals). All three arms underperform buy-and-hold
by a wide margin, mostly from low market exposure in a market that rose ×8. The
capital threshold at which costs destroy an edge could not be estimated because no
edge was found. The same strategy and risk code runs against the simulator, a
read-only mainnet cost shadow and a Binance testnet venue with idempotent client
order IDs; the shadow's pre-registered prediction on spread costs is pending.

## Resumen

Se mide si las señales de noticias aportan alfa a un agente de aprendizaje por
refuerzo (RL) que opera BTCUSDT en barras diarias. Dos agentes PPO idénticos difieren
solo en su espacio de estado: el **Agente A** ve 15 columnas de precio, volumen y
técnicos; el **Agente B** ve las mismas más 7 columnas de noticias derivadas de GDELT
GKG 2.0 (2,74 M artículos, 2016-2026) y del sentimiento de titulares con FinBERT. Un
tercer brazo, el **placebo**, recibe las mismas 7 columnas desplazadas 364 días. El
diseño, las features, la regla de latencia, el criterio de decisión y el número de
pruebas se **pre-registraron** en un ADR sellado por SHA256 antes de entrenar sobre
datos reales. Los costos (spread de Corwin-Schultz, comisión de 10 pb e impacto raíz
cuadrada) se cobran **dentro de la recompensa**; la decisión usa el cierre de `t` y se
ejecuta en la apertura de `t+1`. La evaluación es **walk-forward** sobre 11 ventanas
fuera de muestra (2020-05 a 2026-05) con **10 semillas** por brazo (330
entrenamientos), y el test se tocó **una sola vez**. Antes, el presupuesto de
entrenamiento se calibró sobre fixtures sintéticos con óptimo conocido, y un control
positivo mostró que la tubería deja aprender una señal exógena cuando existe (capture
0,925; `t` pareado 14,8). **Resultado: el veredicto pre-registrado es NO APORTA
(plano).** B supera en mediana a A y al placebo (+0,185 y +0,326 de exceso
logarítmico), pero el Deflated Sharpe Ratio es 0,146 y 0,200 contra un umbral de 0,95.
Un diagnóstico por perturbación muestra que **B no usa las noticias** (fijarlas cambia
la acción 0,011, bajo la regla de 0,02, contra 0,105 de los técnicos). Los tres brazos
pierden con holgura contra buy-and-hold, sobre todo por baja exposición en un mercado
que subió ×8. El capital a partir del cual los costos destruyen la ventaja **no se
pudo estimar porque no hubo ventaja**. El mismo código de estrategia y de riesgo corre
contra el simulador, una sombra de solo lectura sobre mainnet y un venue de testnet de
Binance con `client_order_id` idempotente; la predicción pre-registrada de la sombra
sobre el spread está pendiente.

**Palabras clave:** aprendizaje por refuerzo, PPO, trading algorítmico, análisis de
sentimiento, GDELT, FinBERT, pre-registro, walk-forward, Deflated Sharpe Ratio,
resultado negativo, reproducibilidad.

---

## 1. Introducción

### 1.1 Motivación

La literatura de RL aplicado a trading reporta con frecuencia retornos de backtest que
no sobreviven a tres controles básicos: costos realistas, evaluación fuera de muestra
y corrección por el número de configuraciones probadas (Bailey et al., 2014; López de
Prado, 2018). En RL se suma una cuarta fuente de error: la varianza entre semillas de
entrenamiento suele ser del mismo orden que el efecto reportado (Henderson et al.,
2018; Agarwal et al., 2021). La pregunta de si las noticias aportan información útil
para decidir es antigua (Tetlock, 2007), pero rara vez se contesta con un diseño que
separe "el agente ve noticias" de "el agente ve más columnas".

### 1.2 Preguntas de investigación

- **P1.** ¿Un agente PPO con features de noticias (B) supera a uno idéntico sin ellas
  (A) fuera de muestra, neto de costos?
- **P2.** ¿Esa diferencia se debe al contenido de las noticias y no al hecho de tener
  más entradas? (control: placebo con noticias de `t − 364` días).
- **P3.** ¿Algún brazo supera a buy-and-hold neto de costos?
- **P4.** ¿A partir de qué capital los costos de transacción destruyen la ventaja?
- **P5 (ingeniería).** ¿Se puede construir el sistema de modo que el mismo código de
  estrategia y de riesgo corra en simulación y contra un broker real, y verificar los
  costos del simulador contra el mercado real?

### 1.3 Contribuciones

1. Un **diseño pre-registrado** de comparación A / B / placebo con criterio de
   decisión, número de pruebas y regla de latencia fijados por hash antes de entrenar.
2. Una **cadena de validación** del agente sobre fixtures sintéticos con óptimo
   calculable, con varianza de mercado y de entrenamiento separadas, y un control
   positivo y uno negativo para la tubería de noticias.
3. Un **resultado negativo bien medido**: NO APORTA, con el diagnóstico de por qué
   (el agente no usa la entrada) y las limitaciones que acotan su alcance.
4. Una **arquitectura con paridad backtest-vivo** (`ExecutionVenue`, `Clock`
   inyectable, `client_order_id` del emisor, capa de riesgo fuera del agente) y una
   sombra sobre mainnet que contrasta los costos simulados con el libro L2 real.

---

## 2. Trabajos relacionados

- **RL para trading.** Moody y Saffell (2001) introducen el RL directo para trading;
  Deng et al. (2016) y Jiang et al. (2017) lo extienden con redes profundas; FinRL
  (Liu et al., 2020) ofrece entornos de referencia. Este trabajo usa PPO (Schulman et
  al., 2017) en la implementación de Stable-Baselines3 (Raffin et al., 2021) sobre un
  entorno Gymnasium (Towers et al., 2024).
- **Noticias y sentimiento.** Tetlock (2007) mide el efecto del tono de la prensa
  sobre los retornos. GDELT (Leetaru y Schrodt, 2013) da cobertura global de medios
  con tono precalculado; FinBERT (Araci, 2019) clasifica el sentimiento de textos
  financieros.
- **Sobreajuste de backtests.** Bailey et al. (2014) muestran que probar muchas
  configuraciones garantiza un Sharpe alto por azar; el Deflated Sharpe Ratio (Bailey
  y López de Prado, 2014) lo corrige por número de pruebas, asimetría y curtosis.
- **Estadística en RL profundo.** Henderson et al. (2018) y Agarwal et al. (2021)
  documentan la sensibilidad a semillas y recomiendan reportar distribuciones e
  intervalos, no el mejor resultado.
- **Costos.** El spread se estima con Corwin y Schultz (2012) a partir de máximos y
  mínimos diarios; el impacto sigue la ley de raíz cuadrada (Tóth et al., 2011).
- **Pre-registro.** Nosek et al. (2018) proponen fijar hipótesis y análisis antes de
  ver los datos para separar confirmación de exploración.

---

## 3. Métodos

### 3.1 Datos

- **Precios.** BTCUSDT diario de Binance, versionado con manifest (SHA256, fuente,
  fecha, script). El `timestamp` de cada barra es su **cierre**, así que la fila entera
  es *point-in-time*.
- **Noticias.** GDELT GKG 2.0 en archivos crudos, tema `ECON_BITCOIN`, del 2016-08-01
  al 2026-09-28: 2 740 114 artículos en 356 160 lotes de 15 minutos (2 % de lotes
  inexistentes). La descarga histórica usa el mismo formato que el grabador en vivo,
  así que histórico y vivo se leen con el mismo código.
- **Sentimiento.** FinBERT (`ProsusAI/finbert`, pinneado por revisión y SHA256) sobre
  1 412 307 titulares deduplicados en 24 h. GDELT no trae título antes de ~2019-09-23;
  donde no hay título, FinBERT queda vacío y una columna indicadora lo marca.

### 3.2 Regla anti-lookahead

Una noticia es usable en la decisión de la barra `t` si y solo si
`rótulo_del_lote + latencia ≤ close_time[t]`. La latencia se fijó en **60 minutos**
tras medir en vivo lotes que llegaron ~45 min tarde (p99 en vivo al sellar: 48,4 min).
La regla vive en un único punto del código (`NewsDataset.usable`) y está cubierta por
tests de anti-leakage que corren en un job propio del CI. La deduplicación de títulos
se queda con la **primera** aparición: quedarse con otra sería mirar adelante.

### 3.3 Espacio de estado

| Brazo | Observación |
|---|---|
| A | 15 columnas: retornos logarítmicos, volumen, RSI, MACD, ATR, Bollinger, posición |
| B | A + 7 columnas de noticias: proporción de cobertura, tono GDELT, FinBERT, indicadores de "sin título" y de cobertura suficiente |
| Placebo | A + las mismas 7 columnas tomadas de `t − 364` días (conserva el día de la semana) |

Los escaladores se ajustan **solo en el tramo de entrenamiento** de cada ventana. Nunca
se usan precios crudos.

### 3.4 Simulador y costos

- Decisión con el cierre de `t`, ejecución en la **apertura de `t+1`**.
- Costos dentro de la recompensa: medio spread de Corwin-Schultz (tope 200 pb),
  comisión de 10 pb e impacto raíz cuadrada. La recompensa es el retorno neto del
  equity del motor.
- Reglas de la plataforma reales (`min_notional`, paso de cantidad) leídas de
  `exchangeInfo`. Una orden rechazada se envía igual y el rechazo queda en el log: un
  baseline que pre-redondea a cero haría indistinguible "no quiso" de "no pudo".
- Capital inicial: 100 USDT.

### 3.5 Agente

PPO (Stable-Baselines3), red MLP 32×32, `lookback` 1, tasa de aprendizaje 1e-4, 5
épocas, `gamma` 0,9, **120 000 timesteps** por entrenamiento (configuración "R3").
Evaluación determinista.

### 3.6 Calibración previa sobre fixtures sintéticos

Antes de tocar datos reales, el agente pasó un **protocolo de validación** sobre
series sintéticas con óptimo calculable (ADR 0002 a 0005), con N caminos
independientes × M semillas y las dos varianzas reportadas por separado:

- Los niveles 0 a 2 pasan; el nivel 3 se mide (el agente memoriza en 10 de 10
  caminos); el nivel 4 se partió en dos hipótesis (4a, 4b).
- El presupuesto de entrenamiento se calibró sobre 4b con una grilla declarada antes
  de correrla (ADR 0010): de 120 k a 960 k timesteps, **ninguno pasa**. El diagnóstico
  mostró que el presupuesto no era la palanca.
- Una escalera de regularización, también declarada antes (ADR 0011), encontró que
  **R3 pasa 4b**, y la confirmación con el protocolo completo quedó en verde.

Este paso evitó un error concreto: en el nivel 4 un exceso de +0,164 con diez
semillas de acuerdo resultó ser cinco veces menor que el desvío del baseline entre 20
caminos independientes (0,781). Ninguna cantidad de semillas sobre un solo camino lo
habría revelado.

### 3.7 Puerta de la tubería de noticias (controles positivo y negativo)

Antes de la corrida real, dos fixtures con un canal exógeno (ADR 0012 §5), 10 caminos
× 10 semillas por brazo, 400 entrenamientos:

- **5a, control positivo** (la señal vive en el tono de noticias): B captura 0,925 del
  óptimo (umbral 0,50), A ciego −0,067; diferencia pareada +0,98, `t` = 14,84 contra
  un crítico de 2,262 (N = 10 caminos). **PASS.**
- **5b, control negativo** (7 columnas de ruido): B no gana (`t` = −0,77). El piso de
  ruido del diagnóstico por perturbación es 0,025.

La puerta importa para leer el resultado real: **cuando la señal existe y es fuerte,
B la aprende**. Un NO APORTA en datos reales no se debe a una tubería rota.

### 3.8 Diseño experimental

- **Walk-forward**: 11 ventanas de 1 000 barras de entrenamiento y 200 de test
  (2020-05 a 2026-05), sin `ffill` a través de la frontera.
- **Semillas**: 10 por brazo y ventana (11, 23, 37, 41, 59, 67, 73, 89, 97, 101): 330
  entrenamientos.
- **Sello**: `SEAL.json` registra el commit, el SHA256 del ADR 0012, el del dataset y
  la latencia antes del primer entrenamiento; el runner aborta si algo difiere. Cada
  unidad (brazo, ventana, semilla) se escribe al terminar (persistencia incremental) y
  la corrida es reanudable sin reevaluar nada.
- **Baselines**: buy-and-hold, aleatorio y cruce de medias.

### 3.9 Criterio de decisión pre-registrado (K3)

Con `X` = exceso logarítmico sobre buy-and-hold, `D_A = X(B) − X(A)` y
`D_P = X(B) − X(placebo)`, medianas sobre semillas, **APORTA** exige las seis
condiciones: `D_A > 0` y `D_P > 0` en la serie *mark* y en la de liquidación, y
**DSR(B − A) ≥ 0,95** y **DSR(B − placebo) ≥ 0,95** con `n_trials` = 10. Las demás
combinaciones tienen etiqueta fija (plano, efecto de forma, etc.).

### 3.10 Diagnóstico por perturbación (descriptivo, pre-registrado)

Sobre las políticas guardadas de B, cada grupo de entradas se fija en su media y se
mide el cambio de la acción. Regla del ADR: si fijar las noticias cambia la acción
menos de 0,02, B no las usa.

---

## 4. Infraestructura y paridad backtest-vivo

El objetivo de largo plazo es operar con dinero real, así que el diseño preserva la
paridad desde el principio (ADR 0001):

- **`ExecutionVenue`** como única interfaz de ejecución: la implementan el simulador,
  la sombra sobre mainnet (`ShadowVenue`) y el venue de testnet
  (`BinanceTestnetVenue`). La estrategia y la capa de riesgo no saben contra cuál
  corren.
- **`Clock` inyectable**: ninguna llamada a `datetime.now()`.
- **`client_order_id` generado por el emisor**: un reintento tras un timeout no
  duplica posiciones. Hallazgo con grabaciones reales: Binance **no** rechaza un
  `newClientOrderId` repetido sobre una orden de mercado ya llena, así que la
  idempotencia la garantiza el código propio.
- **Reconciliación al arranque** contra la posición que reporta el venue; tras una
  caída del WebSocket se reconcilia por `myTrades` y se deduplica por `trade_id`.
- **Capa de riesgo fuera del agente**, idéntica en simulación y en vivo.
- **Mainnet de solo lectura por construcción**: un guard lanza error si una URL de
  envío de órdenes no es de testnet, y hay test que lo fija. Escaneo de secretos con
  gitleaks en pre-commit y CI.
- **Sombra desplegada** en una VM con Docker, watchdog que mata el proceso tras 35
  minutos sin progreso, alertas por Telegram y *dead man's switch* externo. Cada 15
  minutos recorre el libro L2 real (1 000 niveles) para órdenes hipotéticas de 10, 100
  y 1 000 USDT y compara contra lo que predice el modelo de costos del simulador en
  ese mismo instante. Corre además los diez Agentes A congelados, uno por semilla.
- **Grabador de noticias en vivo** en la misma VM, con el mismo formato que el
  histórico.

**Calidad.** 1 520 tests, todos sin red (un `conftest` bloquea los sockets); los de
anti-leakage y los lentos en jobs propios; cuatro puertas idénticas en local y en CI
(`ruff check`, `ruff format --check`, `mypy --strict`, cobertura ≥ 85 %), matriz de
Python 3.11 y 3.12, `uv.lock` versionado, `PYTHONHASHSEED=0`. Respuestas de Binance
grabadas y versionadas, nunca inventadas.

---

## 5. Resultados

### 5.1 Resultado principal (P1 y P2)

Medianas sobre 10 semillas, 11 ventanas concatenadas:

| Condición K3 | Valor | Cumple |
|---|---|---|
| `D_A` mark (liquidación) | +0,185 (+0,197); p25 −0,135, p75 +0,228; 7 de 10 semillas > 0 | sí |
| `D_P` mark (liquidación) | +0,326 (+0,329); p25 +0,058, p75 +0,513; 7 de 10 > 0 | sí |
| DSR(B − A), `n_trials` = 10 | **0,146** (máx 0,585) | **no** |
| DSR(B − placebo), `n_trials` = 10 | **0,200** (máx 0,904) | **no** |

**Veredicto: NO APORTA (plano).** Ni siquiera sin corrección por pruebas múltiples
(`n_trials` = 1) la mediana cruza 0,95 (0,573 y 0,804). `D_A` va de −0,55 a +0,67
entre semillas.

### 5.2 ¿Usa B las noticias?

| Grupo fijado en su media | Cambio de la acción (mediana, 22 políticas) |
|---|---|
| técnicos | 0,105 |
| posición | 0,031 |
| **noticias** | **0,011** (p75 0,017; máx 0,049) |
| retornos | 0,004 |
| resultado de órdenes | 0,003 |

Fijar las noticias cambia la acción **0,0106 < 0,02**: **B no usa las noticias**. Cinco
de las 22 políticas son constantes. Las ventanas 0 a 3 reaccionan algo más (hasta
0,048), un indicio débil frente al piso de ruido de 0,025 medido en 5b.

### 5.3 Contra buy-and-hold (P3)

| Brazo | `X` mark (rango entre semillas) | `X` con exposición igualada |
|---|---|---|
| A | −1,53 (−1,75 a −1,20) | −0,26 |
| B | −1,37 (−1,74 a −1,08) | +0,22 (8 de 10 semillas > 0) |
| Placebo | −1,76 (−2,37 a −1,12) | −0,05 |

**Ningún brazo supera a buy-and-hold.** El veredicto secundario de A es NO SUPERA (DSR
0,017). En el periodo de test buy-and-hold crece ×8 (+2,08 log) y los agentes están
poco invertidos. La cifra de B con exposición igualada (+0,22) es el indicio más
favorable a las noticias, pero **no estaba en el criterio y no está corregida por
pruebas múltiples**: no se lee como hallazgo.

### 5.4 Capital umbral (P4)

**No se pudo responder**, porque no hay una ventaja que los costos puedan destruir. El
estudio corre a 100 USDT y el barrido de capital (Etapa 6) no se corrió: dar un umbral
sería inventarlo. Lo que sí se midió: a 100 USDT, el 42,6 % de las 6 263 órdenes de A
se rechazó por `MIN_NOTIONAL`, así que el piso de la plataforma limita al agente antes
que el spread. En la corrida anterior del Agente A (C3), quitar spread y slippage
mejoraba unos +0,7 log y no cambiaba el veredicto: la brecha contra buy-and-hold es
sobre todo de exposición, no de costos.

### 5.5 Por ventana

Las diferencias entre brazos se concentran en la ventana 1 (2020-11 a 2021-06: B −0,24
contra A −0,56) y en las de mucho movimiento; en la mitad de las ventanas los tres
brazos son indistinguibles. El apagón de 17 días de GDELT en 2025 no explica nada (la
ventana 9 da el mismo exceso con y sin esos días).

### 5.6 Sombra sobre mainnet (P5) — **PROVISIONAL**

> La ventana de 30 días va del 2026-10-20 (primer snapshot diario válido, 00:00:30 UTC)
> al 2026-11-18. La predicción se evalúa **una sola vez** al cerrar, con el evaluador
> ya escrito y probado (`src/eval/shadow.py`, PR #84), y no se reabre. **Este
> documento no contiene resultados de la sombra.**

Predicción pre-registrada (ADR 0007 §6), fijada antes de existir datos:

- **P1 (spread):** la mediana de `costo_spread_real / costo_spread_modelo` es
  ≤ 10^−2,5 ≈ 3,2e-3. Es decir: el spread de Corwin-Schultz del simulador
  **sobreestima** el spread real de BTCUSDT en al menos dos órdenes y media de
  magnitud.
- **P2:** para cada tamaño (10, 100, 1 000 USDT) y lado, el costo real medio sin
  comisión está más cerca de cero que del modelo principal.
- Veredicto: SOSTENIDA / PARCIAL / REFUTADA / NO EVALUABLE (menos de 25 de 30 diarios
  válidos).

Resultado: **pendiente (~2026-11-18).** Consecuencia prevista: si los costos reales
difieren materialmente de los simulados, las secciones 5.1 a 5.4 se releen con ese
factor. Si se sostiene P1, el escenario principal fue conservador con los costos y el
"solo comisión" era el más cercano a la realidad; eso **no** rescata a ningún brazo,
porque en C3 los agentes tampoco superaban a buy-and-hold con solo comisión.

---

## 6. Discusión

**Qué dice el resultado.** Con este agente, estas siete columnas y barras diarias,
añadir noticias no produce una mejora distinguible del ruido. El diagnóstico explica
por qué: la política casi no reacciona a esas entradas. La lectura correcta no es "las
noticias de GDELT no tienen alfa" sino "**este agente no aprendió a extraerlo**". La
puerta 5a descarta que la causa sea una tubería rota: con una señal fuerte, B la
aprende.

**Por qué el diseño importa más que el número.** Sin el placebo, un `D_A` de +0,185
en mediana se habría podido presentar como evidencia de que las noticias ayudan. Sin
el DSR, siete de diez semillas positivas parecen mucho. Sin la separación de
varianzas, un exceso de +0,164 con diez semillas de acuerdo habría pasado por
hallazgo. Cada control se fijó antes de ver el dato que podía desmentirlo.

**Potencia.** El efecto observado es pequeño (Sharpe diario de B − A ≈ 0,004). Con un
solo camino histórico, confirmar un efecto de ese tamaño exigiría órdenes de magnitud
más barras de las que existen. El estudio no tenía potencia para un efecto chico, y
se dice: lo que sí descarta es un efecto grande con esta configuración.

**Ingeniería.** La parte de infraestructura se construyó para que el paso a dinero
real no requiera reescribir la estrategia. El hallazgo sobre `newClientOrderId` es un
ejemplo de por qué se validó contra respuestas reales grabadas y no contra la
documentación.

## 7. Limitaciones

1. **N = 1 camino real.** Diez semillas miden varianza de entrenamiento, no de
   mercado; el error de muestreo no se puede medir, solo acotar con más ventanas.
2. **Barras diarias.** No se midió operar noticias intradía, que en cripto se
   descuentan en minutos u horas.
3. **Fuente de prensa general.** GDELT mide medios generales, no conversación cripto.
   La hora es la de detección de GDELT, no la de publicación original (sesgo
   conservador).
4. **Agente pequeño.** MLP 32×32 con alta entropía de política en entrenamiento
   (σ ≈ 0,93). Un LSTM rindió algo mejor en la calibración y no se probó en el estudio.
5. **Sin agente placebo guardado** para el diagnóstico: la comparación es por
   reemplazo de entradas. El umbral 0,02 y el piso 0,025 vienen de fixtures
   sintéticos.
6. **Cobertura de GDELT.** 4,4 % de las barras de test de B sin cobertura suficiente
   (27 % en la ventana 0).
7. **A se evaluó dos veces** sobre el mismo test (C3 y R3); se declara en el conteo de
   pruebas.
8. **Un solo activo, un régimen de riesgo, un capital.** El barrido de regímenes y
   capital no se corrió.
9. **Sombra:** no mide la comisión efectiva de una cuenta y no hay L2 histórico para
   verificar el libro de 2020 a 2026.

## 8. Conclusión

El estudio responde con un negativo bien medido: en las condiciones pre-registradas,
las noticias de GDELT no mejoran a un agente PPO sobre BTCUSDT diario, el agente no
usa esa entrada, y ningún brazo supera a buy-and-hold neto de costos. No se arriesga
capital real: falta el gate de 3 a 6 meses de paper trading contrastado contra el
simulador, y no hay una estrategia que lo justifique. Una segunda ronda con otras
features o arquitectura exige un pre-registro nuevo (borrador en el ADR 0013) y cuenta
como prueba adicional en el DSR.

---

## 9. Declaraciones

**Pre-registro.** Los diseños se fijaron en ADRs antes de correr: ADR 0006
(walk-forward del Agente A), ADR 0007 (sombra y predicción sobre el spread), ADR 0010 y
0011 (calibración), ADR 0012 (Agente B). Cada corrida verifica el SHA256 de su ADR y
aborta si difiere. Ningún ADR pre-registrado se editó después de ver resultados; las
correcciones van en adendas.

**Resultado negativo.** Se reporta tal cual. No se buscó otra configuración para
"rescatar" el resultado dentro de este estudio.

**Disponibilidad de código y datos.** Código, ADRs, resúmenes y manifests en el
repositorio. Los artefactos completos (registros por unidad y políticas entrenadas)
están en GitHub Releases (`c3-results`, `n9-resultados`), verificables por SHA256
contra `results/*/MANIFEST.json`. Los datos de GDELT son públicos; el dataset agregado
se versiona con su manifest.

**Conflictos de interés.** Ninguno. No hay capital real comprometido.

---

## 10. Checklist de reproducibilidad

Basado en la *ML Reproducibility Checklist* (Pineau et al., 2021) y la checklist de
NeurIPS.

| Ítem | Estado | Dónde |
|---|---|---|
| Hipótesis y criterio declarados antes de los datos | Sí | ADR 0012 §6, `SEAL.json` |
| Descripción del modelo y del algoritmo | Sí | §3.5, `src/agents/study.py` (`STUDY_PPO`) |
| Hiperparámetros y cómo se eligieron | Sí, por calibración sobre fixtures, nunca contra el test | ADR 0010, 0011; `results/confirmacion/` |
| Número de semillas y cómo se reporta | 10; mediana, p25, p75, mín, máx | `results/estudio/REPORTE.md` |
| Varianza de mercado separada de la de entrenamiento | Sí en fixtures (N caminos × M semillas); N = 1 declarado en datos reales | `eval.distribution.decompose_variance`, ADR 0004 |
| Corrección por pruebas múltiples | DSR con `n_trials` declarado | §3.9 |
| Partición de datos | Walk-forward, 11 ventanas, sin solapamiento de test | ADR 0006, 0012 |
| Test tocado una vez | Sí, garantizado por sello | `SEAL.json` |
| Anti-lookahead verificado por tests | Sí, job de CI propio (`-m leakage`) | `tests/**/test_*leakage*.py` |
| Costos en la recompensa | Sí | `src/sim/costs.py` |
| Baselines | Buy-and-hold, aleatorio, cruce de medias | `src/agents/baselines.py` |
| Controles positivo y negativo | Sí (5a, 5b) | `results/puerta/` |
| Dependencias pinneadas | `uv.lock`, CI con `uv sync --frozen` | raíz del repo |
| Entorno de cómputo | CPU; la corrida real ~10 h en 3 procesos | ADR 0012 §10 |
| Código para reproducir las cifras | `python -m agents.cli study assemble --out results/estudio` | README |
| Artefactos completos con hash | Releases + `MANIFEST.json` | `results/*/` |
| Resultados negativos reportados | Sí | este documento |

---

## Referencias

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. y Bellemare, M. G. (2021).
  Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS 34*.
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language
  models. *arXiv:1908.10063*.
- Bailey, D. H., Borwein, J. M., López de Prado, M. y Zhu, Q. J. (2014).
  Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting
  on out-of-sample performance. *Notices of the AMS, 61*(5), 458-471.
- Bailey, D. H. y López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for
  selection bias, backtest overfitting and non-normality. *Journal of Portfolio
  Management, 40*(5), 94-107.
- Corwin, S. A. y Schultz, P. (2012). A simple way to estimate bid-ask spreads from
  daily high and low prices. *Journal of Finance, 67*(2), 719-760.
- Deng, Y., Bao, F., Kong, Y., Ren, Z. y Dai, Q. (2016). Deep direct reinforcement
  learning for financial signal representation and trading. *IEEE Transactions on
  Neural Networks and Learning Systems, 28*(3), 653-664.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. y Meger, D. (2018).
  Deep reinforcement learning that matters. *AAAI 32*.
- Jiang, Z., Xu, D. y Liang, J. (2017). A deep reinforcement learning framework for
  the financial portfolio management problem. *arXiv:1706.10059*.
- Leetaru, K. y Schrodt, P. A. (2013). GDELT: Global data on events, location, and
  tone, 1979-2012. *ISA Annual Convention*.
- Liu, X.-Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B. y Wang, C. D. (2020).
  FinRL: A deep reinforcement learning library for automated stock trading in
  quantitative finance. *NeurIPS Deep RL Workshop*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Moody, J. y Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE
  Transactions on Neural Networks, 12*(4), 875-889.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C. y Mellor, D. T. (2018). The
  preregistration revolution. *PNAS, 115*(11), 2600-2606.
- Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc,
  F., Fox, E. y Larochelle, H. (2021). Improving reproducibility in machine learning
  research. *JMLR, 22*(164), 1-20.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. y Dormann, N. (2021).
  Stable-Baselines3: Reliable reinforcement learning implementations. *JMLR,
  22*(268), 1-8.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. y Klimov, O. (2017). Proximal
  policy optimization algorithms. *arXiv:1707.06347*.
- Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in
  the stock market. *Journal of Finance, 62*(3), 1139-1168.
- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. y
  Bouchaud, J.-P. (2011). Anomalous price impact and the critical nature of
  liquidity in financial markets. *Physical Review X, 1*(2), 021006.
- Towers, M. et al. (2024). Gymnasium: A standard interface for reinforcement
  learning environments. *arXiv:2407.17032*.

## Anexo A. Trazabilidad

Cada cifra de este documento sale de un archivo versionado:

| Sección | Fuente |
|---|---|
| 3.6 | `results/protocol.json`, `results/multipath_level_*.json`, `results/calibracion/`, `results/regularizacion/`, `results/confirmacion/`, ADR 0004, 0010, 0011 |
| 3.7 | `results/puerta/REPORTE.md` |
| 5.1, 5.3, 5.5 | `results/estudio/REPORTE.md`, `results/estudio/summary.json` |
| 5.2 | `results/estudio/REDIAGNOSTICO.md`, `results/estudio/rediagnosis.json` |
| 5.4 | `results/estudio/REPORTE.md`, `results/c3/REPORTE.md` |
| 5.6 | ADR 0007 §6; resultado en `results/sombra/` cuando exista |
| 4 | ADR 0001, 0007, 0008, 0009; `src/live/`; matriz del README |

Matriz completa de hitos y PRs: sección "Estado y matriz de trazabilidad" del
[README](../../README.md).
