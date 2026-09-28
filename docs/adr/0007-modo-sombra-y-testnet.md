# ADR 0007 — Modo sombra sobre mainnet y testnet (Etapa 3.5)

- **Estado:** propuesto en el PR 1 de la Etapa 3.5. Partición, orden e interfaz
  aprobados antes de escribirlo. Los cuatro puntos de la sección 5 están aprobados,
  con el agregado del punto 4. La predicción de la sección 6 la pidió el revisor.
- **Fecha:** 2026-09-28
- **A la fecha de este ADR la sombra no tomó ningún snapshot.** No existe el código
  que los toma: llega en los PRs 2 a 4. La predicción de la sección 6 se fija sin
  haber visto ni un libro de órdenes. El proceso de la sombra (PR 4) escribe el
  SHA256 de este archivo en el encabezado de su primer archivo de datos, así que el
  orden entre el pre-registro y los datos queda verificable además de por el historial
  de git.

---

## 1. Dos validaciones que no se mezclan

| | Sombra sobre mainnet | Testnet |
|---|---|---|
| **Valida** | **Costos**: ¿el simulador dice la verdad sobre spread y slippage? | **Mecánica**: envío, cancelación, idempotencia, reconciliación, recuperación |
| **No valida** | Mecánica de envío: no envía nada | Costos: el libro de testnet tiene liquidez artificial y precios que pueden divergir de mainnet |
| **Credenciales** | Ninguna. Solo endpoints públicos de datos | API key de testnet, por variable de entorno |
| **Órdenes** | Imposibles por construcción | Solo a hosts de testnet (sección 2) |

Una cifra de costo medida en testnet no entra a ningún reporte de costos. Un fill de la
sombra no prueba que el adaptador de broker funcione.

## 2. Seguridad

- **Mainnet es de solo lectura por construcción.** Todo código que envía órdenes pasa
  la URL por `live.guard.assert_testnet_order_url` antes de abrir la conexión. El guard
  parsea la URL y compara el **host exacto** contra `TESTNET_ORDER_HOSTS`
  (`testnet.binance.vision`). Exige HTTPS y rechaza credenciales embebidas y puertos
  explícitos. Un `in` o un `endswith` sobre el string dejaría pasar
  `https://testnet.binance.vision.evil.com` o `https://testnet.binance.vision@api.binance.com`;
  hay un test para cada uno. Ampliar `TESTNET_ORDER_HOSTS` es una decisión de
  seguridad: se revisa como tal, en un PR propio.
- **La sombra no usa API key.** Lee `https://data-api.binance.vision`, el host de datos
  públicos de mercado, que no acepta órdenes ni endpoints de cuenta.
- **Credenciales por variables de entorno o `.env`**, nunca en el repo. `.env` está en
  `.gitignore` y gitleaks corre en pre-commit y en CI. `ShadowConfig` no incluye el
  token de Telegram en `repr` ni en `describe()`, que es lo que se serializa junto a
  cada corrida, y `JsonFormatter` redacta todo campo de log cuyo nombre contenga
  `token`, `secret`, `password`, `api_key`, `apikey` o `signature`.
- **Las API keys de mainnet, cuando algún día existan, se crean SIN permiso de
  retiro**, con restricción por IP al VPS que las usa. Una clave filtrada con permiso
  de retiro es una pérdida total; sin él, el daño máximo lo acota la capa de riesgo.
- **Los tests que tocan red llevan el marcador `network`** y quedan fuera del CI
  (`-m "not slow and not network"`). El guard de `tests/conftest.py` se mantiene y
  solo se levanta con `RUN_NETWORK_TESTS=1`. La lógica se testea contra respuestas
  grabadas y versionadas; el PR 2 trae el script para grabarlas.
- **Persistencia incremental.** La sombra escribe con append y `fsync` en cada
  snapshot, en un archivo por día UTC. Un proceso que muere a las tres semanas no se
  lleva las tres semanas.

## 3. Métrica pre-registrada: `costo_real / costo_modelo`

Por cada snapshot, cada tamaño `S ∈ {10, 100, 1000}` USDT y cada lado (compra y
venta):

**Real** (`walk_book`): se recorre el libro L2 del snapshot consumiendo `S` USDT de
nocional.

- `mid = (mejor bid + mejor ask) / 2`
- `half_spread_rel = |mejor precio del lado − mid| / mid`
- `depth_impact_rel = |vwap − mejor precio del lado| / mid`
- `real_rel = half_spread_rel + depth_impact_rel`
- Si los niveles pedidos (`depth_limit = 1000`) no alcanzan para `S`, el registro
  lleva `exhausted = true` y no entra a los cocientes; se cuenta aparte.

**Modelo** (`model_cost`): lo que habría cobrado el simulador de C3 por la misma
orden, **con el código de `sim.costs`, no una copia**, y los parámetros de la sección
1.1 del ADR 0006.

- `half_spread_rel`: `CorwinSchultzSpread(max_bps=200)` sobre las **dos últimas
  barras diarias cerradas**. Es lo mismo que usa el motor: `SpreadContext` llega hasta
  la barra de decisión `t` inclusive (`Simulator._spread_context`).
- `slippage_rel`: `SqrtSlippage(k=0.1)` con participación
  `= (S / mid) / volumen de la última barra diaria cerrada` (sección 5, punto 3).
- `model_rel = half_spread_rel + slippage_rel`

**Cocientes**, ambos **sin comisión**:

- `ratio_total = real_rel / model_rel`
- `ratio_spread = real.half_spread_rel / model.half_spread_rel`
- Si el denominador es 0, el cociente es `None`, no `inf` ni `0`, y se cuenta. Sobre
  los días del test de C3, Corwin-Schultz trunca a cero en el 34 % de las barras: no
  es un caso raro.

La comisión queda fuera porque no la mide el libro. Es la del tarifario (10 bps
taker), idéntica en ambos lados del cociente; incluirla empujaría el cociente hacia 1
y escondería justo lo que se quiere medir.

**Reporte.** Por tamaño y lado: distribución de `ratio_total` (mediana, p25, p75,
min, max), el conteo de `None` y de `exhausted`, los dos componentes por separado y
los costos **absolutos en bps**, reales y del modelo. El absoluto es lo que se usará
para recalibrar el simulador; el cociente solo dice cuánto se equivocaba.

## 4. Cadencia y fuente

- **Un snapshot cada 15 minutos**, alineado a la hora (`:00`, `:15`, `:30`, `:45`
  UTC), más **uno dedicado a las 00:00:30 UTC**.
- El de las 00:00:30 es el **comparable directo con el backtest**: el simulador
  ejecuta al open de `t+1`, que es 00:00 UTC. Los 30 segundos dan tiempo a que la
  barra diaria `t` cierre y se publique, así que el modelo ve exactamente las barras
  que habría visto el backtest. La liquidez en el rollover del día puede no ser la
  del promedio, y por eso se reporta por separado.
- **REST, no WebSocket**, para la profundidad. Una foto cada 15 minutos no necesita
  mantener un libro local sincronizado. El REST de `depth` no trae hora: el registro
  lleva la hora del `Clock` inyectado al recibir la respuesta, y `lastUpdateId`.

## 5. Los cuatro puntos

1. **El cociente de spread va a salir diminuto, y eso ya se sabe.** El tick de
   BTCUSDT es 0,01 sobre un precio de decenas de miles: con el libro a un tick, el
   medio spread real es del orden de 1e-7 relativo (0,001 bps). Corwin-Schultz diario
   lo estima en decenas de bps. Por eso el reporte lleva el costo real absoluto por
   tamaño, que es lo que sirve para recalibrar. La sección 6 corrige la magnitud que
   se había propuesto para ese cociente.
2. **Momento de la medición**: el snapshot de las 00:00:30 (sección 4).
3. **Volumen para el slippage.** El simulador usa el volumen de la barra de
   ejecución, que en vivo todavía no existe. La sombra usa el de la **última barra
   diaria cerrada**. `ASSUMPTION`: la diferencia es irrelevante a estos tamaños. Con
   1000 USDT sobre un volumen diario de cientos de millones a miles de millones de
   USDT, la participación es del orden de 1e-6 y el término de slippage es menor a
   0,1 bps en cualquiera de las dos versiones.
4. **Qué política corre en sombra.** Un Agente A congelado:
   - **Se entrena con el código actual**, posterior al fix del sizer (#20, regla del
     cero numérico), **no con el de C3**. La configuración de PPO es la de C3
     (opción i, 60 000 timesteps) y el entrenamiento usa las 1000 barras diarias más
     recientes al momento de congelarlo.
   - **Una semilla, fijada ahora: 11** (`STUDY_SEEDS[0]`). No se entrenan varias y se
     elige una: eso sería reportar el mejor seed.
   - Se guarda con el SHA256 de los pesos, el commit y la configuración.
   - **No es comparable con C3.** El fix del sizer cambió la observación del agente
     (`last_order_rejected` ya no se enciende por residuo), la ventana de
     entrenamiento es otra y es una sola semilla.
   - Está **subentrenado** y se declara así. Su función es producir decisiones reales
     para medir mecánica y costos, no rendimiento. Se reemplaza cuando el presupuesto
     de entrenamiento se calibre sobre el fixture 4b, antes de la Etapa 5.
   - Opera un ledger sombra de 100 USDT, el capital de C3 y el que se planea operar.

## 6. Predicción pre-registrada sobre el spread

**Enunciado del revisor:** si el cociente de spread sale del orden de 1e-3, el
escenario principal de C3 sobreestimó los costos y el de solo comisión era el más
cercano a la realidad.

**Corrección de magnitud antes de operacionalizar.** El "orden de 1e-3" salió de la
propuesta de la Etapa 3.5 y es un error de aritmética. Con el dataset versionado:

- Mediana del medio spread de Corwin-Schultz en los días del test de C3: **31 bps**
  (último año: 20 bps).
- Medio spread real con el libro a un tick: `0,005 / precio`, mediana **0,001 bps**.
- Cociente esperado si el libro está a un tick: **≈ 2e-5**, no 1e-3.

Leído como banda de dos lados (`[10^-3,5, 10^-2,5]`), el enunciado quedaría
**refutado por el resultado más probable**, que es uno donde su conclusión vale con
más fuerza. Se operacionaliza entonces como cota de un solo lado: **del orden de 1e-3
o menor**. Esta corrección se fija acá, antes de que existan datos. Si el revisor
prefiere la banda de dos lados, se cambia en este PR y no después.

**Datos.** Los snapshots de las 00:00:30 de los **primeros 30 días UTC** desde el
primero válido. Los de 15 minutos se reportan con las mismas cifras y no deciden. La
predicción se evalúa **una vez**, al cumplirse los 30 días; la sombra sigue juntando
datos después y eso no reabre el veredicto.

**Condición P1 (spread).** La mediana de `ratio_spread` sobre los snapshots con
`model.half_spread_rel > 0` es **≤ 10^-2,5 ≈ 3,2e-3**. Como el medio spread no depende
del tamaño ni del lado, hay un valor por snapshot.

**Implicación P2 (se mide, no se asume).** Que el escenario de solo comisión sea "el
más cercano a la realidad" es una afirmación sobre el costo total sin comisión, no
solo sobre el spread. El impacto de profundidad a 1000 USDT podría romperla aunque P1
valga. Por eso se contrasta aparte: para cada tamaño y lado, el costo real medio sin
comisión en bps (`real_rel`) está más cerca del de solo comisión (0) que del modelo
principal (`model_rel`):

    media(real_rel) < media(model_rel) / 2

Las medias van sobre **todos** los snapshots de las 00:00:30, incluidos los que tienen
`model_rel = 0`, porque el P&L paga la media y no la mediana.

**Veredicto:**

| Veredicto | Condición |
|---|---|
| **SOSTENIDA** | P1, y P2 en las 6 celdas (3 tamaños × 2 lados) |
| **PARCIAL** | P1, pero P2 falla en alguna celda. El hallazgo nombra las celdas |
| **REFUTADA** | No se cumple P1 |
| **NO EVALUABLE** | Menos de 25 de los 30 snapshots de las 00:00:30 válidos, o menos de 10 con `model.half_spread_rel > 0` |

**Alcance.** La predicción se mide hoy, y su conclusión es sobre C3, cuyo test cubre
2020-05 a 2026-05. El salto requiere que el libro haya estado a pocos ticks también en
ese período; no requiere la misma profundidad. No hay L2 histórico para verificarlo,
y se declara como limitación en el reporte. Tampoco se mide la comisión: la sombra no
opera y no ve la tarifa efectiva de una cuenta. Eso queda para la Etapa 7.

## 7. Partición de la Etapa 3.5

Primero va lo que acumula datos con el tiempo. Testnet no bloquea a la sombra.

| PR | Contenido |
|---|---|
| 1 | Este ADR. `live/`: `ShadowConfig` por variables de entorno, guard de URLs de testnet y logs JSON. Guía del bot de Telegram |
| 2 | `MarketData` REST de solo lectura (profundidad, klines cerradas, filtros → `InstrumentSpec`) y script para grabar respuestas |
| 3 | `walk_book` y `model_cost` como funciones puras. Reusan `sim.costs` |
| 4 | Proceso sombra de costos: `Clock` inyectado, cadencia de la sección 4, append con `fsync`, heartbeat |
| 5 | `ShadowVenue` (implementa `ExecutionVenue`, llena contra el libro sin enviar nada) y el Agente A de la sección 5 |
| 6 | Dockerfile, compose con reinicio, healthcheck, watchdog con alerta por Telegram y guía de VPS |
| 7 | `BinanceTestnetVenue` REST firmado, idempotente por `client_order_id` |
| 8 | WebSocket de fills de testnet, reconexión y recuperación tras caída |

## 8. Qué no concluye la sombra

- **Latencia y cola.** `walk_book` recorre una foto del libro. Una orden real llega
  unos milisegundos después y el libro pudo haber cambiado. Es un piso del costo de
  una orden de mercado, no el costo.
- **Tamaño.** A 10–1000 USDT la sombra no dice nada sobre el impacto de órdenes
  grandes. Ese barrido es la Etapa 6, y su calibración necesitaría otra fuente.
- **Un solo símbolo y un solo venue.**

## Consecuencias

- El simulador sigue corriendo con los supuestos del ADR 0006 hasta que la sombra
  produzca 30 días. Recalibrar antes sería ajustar el modelo con una muestra que se
  eligió por conveniencia.
- La recalibración del modelo de costos, si hace falta, es un ADR aparte y re-corre lo
  que dependa de él. No se aplica retroactivamente a C3, que queda como está: su
  sensibilidad de solo comisión ya acota el efecto.
