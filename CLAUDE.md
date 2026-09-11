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

## Invariantes consolidados en la Etapa 1

No los redefinas. Impórtalos o respétalos.

- `allow_short=False`. El flag existe en la firma y lanza `NotImplementedError`.
- Espacio de acción del agente en `[0, 1]`. Usa la constante `LONG_ONLY_ACTION_RANGE`
  del motor; no la redeclares en el env.
- Fraccionales, tick, lote, mínimos y comisiones son propiedad de `InstrumentSpec` por
  instrumento, no configuración global.
- En cripto el binding constraint suele ser `min_notional`, no `min_order_qty`.
  `check_tradable` los evalúa por separado y reporta cuál mordió.
- `check_tradable` **nunca redimensiona**. Rechaza y registra el motivo.
- `round_qty` siempre hacia cero.
- Contabilidad con `positions: dict[str, float]` aunque hoy sea un solo símbolo.
- Remanente de fill parcial se cancela, no se arrastra. La posición que ve el agente es
  siempre la realmente llenada, nunca la intencionada.
- Orden emitida en la última barra: `EXPIRED`.
- Dos series de equity paralelas:
  - `equity_mark` = cash + posición a close. Métrica primaria.
  - `equity_liquidation` = cash + producto neto de cerrar la posición en esa barra,
    con spread, slippage y comisión de salida. Contrafactual puro: no toca cash, posición
    ni el log. El slippage de salida usa el volumen **de esa barra**.
  - Con `allow_short=False`, aserción de que `equity_liquidation <= equity_mark`. Si se
    invierte, hay bug de signo.
- **`gap`, `spread_cost`, `slippage_cost` y `commission` van siempre en columnas separadas
  del log. Jamás agregados en una sola métrica de "costos".** Medido en la Etapa 1: en el
  tier alto el gap fue favorable y compensó un cuarto de los costos, invirtiendo el orden
  aparente de los regímenes.
- Spread y slippage nunca mejoran el precio de referencia. Hay test explícito.
- Precios sin ajustar para ejecución; factores de ajuste en columnas separadas.
  `total_return_index` es point-in-time; `backward_adjusted_close` exige
  `allow_lookahead=True`.
- Corwin-Schultz para estimar spread desde high-low, no el rango crudo. Devuelve spread
  relativo; el costo de cruzar es la mitad.
- `cash_rate` configurable, 0.0 por defecto.
- El generador sintético de Heston produce caminos continuos y por tanto gap cero.
  `overnight_gap_frac` existe para hacer testeable ese efecto.

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

## Invariantes de `eval/`

Consolidados al cerrar la Etapa 1. Las métricas son funciones puras sobre arrays,
no métodos de `SimResult`: las mismas tienen que poder aplicarse a un benchmark
externo o a la salida de un broker en la Etapa 7.

- **El gap nunca se suma a los costos.** `CostReport.total_costs_paid` es exactamente
  spread + slippage + comisión. El gap tiene línea propia, con signo, y solo aparece
  sumado en `implementation_shortfall`, que se llama por su nombre.
- **`equity_liquidation <= equity_mark` NO implica que su drawdown sea mayor.** Si la
  posición es grande en el pico y chica en el valle, la fricción deprime el pico más
  que el valle y el drawdown de la serie de liquidación resulta *menor*. Es
  contraintuitivo y hay un test que lo fija. No escribas la aserción al revés.
- **Unidad de trade: round-trip flat-to-flat.** De posición cero a cero, con todas las
  compras y ventas intermedias adentro. **La posición abierta al final no entra al win
  rate**: se reporta aparte. Contarla mezcla P&L realizado con no realizado y hace que
  una estrategia que no cierra sus perdedoras se vea mejor de lo que es.
- **Sin trades cerrados, `win_rate` y `profit_factor` son `None`, no `0.0`.** Un cero se
  lee como "perdió todos los trades", que es distinto de "no hizo ninguno". En un estudio
  cuya hipótesis nula es que el agente no opera, esa diferencia es el resultado.
- **La tasa libre de riesgo por defecto es el `cash_rate` de la corrida**, no cero. El
  motor devenga interés sobre el cash ocioso; descontar cero le regala alfa a una
  estrategia que pasa la mitad del tiempo fuera del mercado cobrando esa misma tasa.
  `periodic_rate` usa la misma fórmula que `SimConfig.rate_per_bar` y hay un test que
  las compara.
- **"Volatilidad cero" se evalúa con tolerancia relativa (`DISPERSION_NULA_REL`), no
  contra cero exacto.** Una serie que crece exactamente 10% por barra no produce
  retornos idénticos en float64; sin el umbral el Sharpe sale del orden de 1e15 en vez
  de infinito, y ese número gana cualquier ranking de semillas sin significar nada.
- **Sharpe y Sortino sobre retornos simples**, no logarítmicos. El Sharpe se define sobre
  aritméticos; calcularlo sobre logarítmicos lo subestima de forma creciente con la
  volatilidad.
- **Sortino: segundo momento parcial inferior sobre todas las observaciones**, no solo
  sobre las negativas. Dividir entre la cantidad de negativas infla el ratio de las
  estrategias que rara vez pierden, que son las que hay que mirar con más desconfianza.
- **Una serie que toca cero se trunca en esa barra** y el reporte expone `ruined_at`.
  Después de cero no hay retorno definido y todo lo posterior sale `inf` o `nan`.
- **`annualization_reliable = years >= 1`.** El CAGR y el Sharpe se anualizan siempre
  (en walk-forward todas las ventanas son cortas y hay que anualizarlas igual para
  poder compararlas), pero la extrapolación queda marcada, no disimulada.
- Una serie de `n` observaciones tiene `n - 1` períodos. Un año de barras diarias son
  253 observaciones, no 252.

## Invariantes de ejecución real

Consolidados en el refactor de arquitectura. Ver `docs/adr/0001-arquitectura-para-ejecucion-real.md`
para el porqué de cada uno y el inventario de lo que falta para operar en vivo.

- **`ExecutionVenue` es la interfaz contra la que se opera**, en simulación y en vivo:
  `submit_order`, `cancel_order`, `get_positions`, `get_fills`, `reconcile`. `Simulator`
  es una implementación; el adaptador de broker de la Etapa 7 será otra. El runner y la
  capa de riesgo hablan con el protocolo y no saben cuál tienen enfrente.
- **`submit_order` acusa recepción, no ejecución.** Devuelve `OrderAck`, nunca `Fill`.
  La validación va partida en dos: **admisión** al enviar (identificador presente, no
  duplicado, venue activo) y **ejecución** al llenar (volumen, mínimos, cash). No se
  pueden adelantar: en el momento del envío, los datos de la barra de ejecución todavía
  no existen. Validarlos ahí sería lookahead.
- **La superficie pública del motor está fijada por test** en una lista blanca de seis
  nombres. Un séptimo método público rompe `test_el_motor_no_expone_step` y obliga a
  justificarlo. Ningún método público mueve el cursor `_t`; hay tests que lo verifican.
- **El tiempo entra por el `Clock` inyectable.** `SystemClock` es el único punto de
  `src/` que consulta el reloj de pared, y hay un test que recorre el árbol y falla si
  aparece un `datetime.now()` en otro módulo. `advance_to` prohíbe retroceder dentro de
  una corrida; `reset_to` es la única forma de rebobinar, y existe porque una instancia
  de `Simulator` se corre más de una vez.
- **El `client_order_id` lo genera el emisor**, y `submit_order` rechaza toda orden que
  llegue sin él. Reenviar un identificador ya visto devuelve el mismo acuse con
  `is_duplicate=True` y **no ejecuta de nuevo**. El generador es inyectable porque
  reproducibilidad y unicidad entre reinicios no se satisfacen con uno solo:
  `SequentialIds` (determinista) en simulación, `PrefixedSequentialIds` en vivo.
- **La `RiskLayer` rechaza, nunca redimensiona.** Misma regla que `check_tradable`.
  Recortar una orden al límite hace indistinguible "el agente pidió esto" de "el límite
  lo recortó hasta acá". `GateDecision` no tiene dónde devolver una orden modificada.
- **Los vetos de riesgo van en `SimResult.gate_rejections`, lista separada de los
  rechazos del venue.** "El venue no pudo llenarla" y "nuestra capa no la dejó salir"
  son diagnósticos distintos; agregarlos es el mismo error que agregar el gap a los costos.
- **El kill switch permite cerrar, no abrir.** Bloquear todo deja la posición abierta
  justo cuando algo salió mal. Solo pasan las órdenes que reducen exposición. La
  reposición es manual: si se repusiera solo sería una pausa, no un kill switch.
- **El ancla del límite de pérdida diaria es el equity con que cerró el día anterior**,
  no el primero del día actual. Con barras diarias —donde una barra es un día— anclar a
  la propia barra compara el equity contra sí mismo y el límite no muerde nunca.
  Es pérdida diaria, no drawdown intradiario: el ancla es la apertura, no el máximo.
- **`risk` depende de `sim`, nunca al revés.** La costura es el `Protocol`
  `sim.gate.OrderGate`. El simulador no sabe que existe una política de riesgo, igual
  que no lo sabe un broker real.

## Invariantes del entorno Gymnasium

Consolidados en la Etapa 2. El env es un **wrapper delgado**: rutea, no calcula.

- **`Simulator.drive()` es la costura de inversión de control.** Generador que cede en
  cada punto de decisión y recibe la orden por `send()`. Se admite como séptimo método
  público porque **`send` fusiona avanzar y decidir en una operación atómica**: para
  llegar a `t+1` hay que entregar la decisión de `t`, así que no se puede adelantar el
  cursor, mirar y volver. `next(gen)` equivale a `send(None)`, o sea "decidí no operar".
- **`run` y `drive` son dos bucles públicos, no dos implementaciones.** Los cuatro pasos
  por barra (`_start`, `_open_bar`, `_dispatch`, `_finish`) son la única copia de la
  contabilidad. Si agregas un tercer bucle, consume esos helpers.
- **La garantía anti-leakage del env vive en `ObservationBuilder.build`, que recibe solo
  una `Decision`.** El generador no puede impedir que el env tenga la `BarSeries` —la
  necesita para construir el simulador—, así que la garantía es estructural en el
  constructor de la observación. Hay un test que compara dos series idénticas hasta `t`
  y distintas después, y exige observaciones bit-idénticas.
- **Dimensionar es ejecución: `sim.sizing.TargetWeightSizer`, no el env.** El env y
  cualquier estrategia directa usan el mismo sizer. Si el env tuviera el suyo, el
  backtest dejaría de corresponder al simulador validado.
- **El sizer no pre-redondea a cero ni recorta contra el cash.** `round_qty` redondea
  hacia cero: redondear ahí convertiría un delta chico en `qty=0` y la orden
  desaparecería del log. Se manda sin redondear y el venue rechaza con su motivo.
  `deadband=0.0` por defecto; una banda muerta se configura explícita y se serializa.
- **`safety` define qué significa la acción, no censura órdenes.** Un peso de 1.0 es "lo
  más invertido que se puede estar sin conocer el precio de ejecución" (98% por defecto).
  Estar exactamente all-in exigiría lookahead.
- **El recorte de la acción a `[0,1]` se registra** en `ClipEvent` y en `info`. Un agente
  entrenado sobre un rango que el env recorta en silencio aprende sobre un mundo que no
  existe. Una acción no finita es un error, no un recorte.
- **La observación distingue "no quise" de "no pude"** con tres columnas separadas:
  `last_order_blocked` (veto de riesgo), `last_order_rejected` (rechazo del venue) y
  `last_fill_ratio`. La posición observada es la **realmente llenada**, del ledger.
- **La normalización se ajusta solo con train** (`fit_scaler_on_train` recorre
  `MarketView` sobre la porción de train, así que la serie de test no está en el objeto)
  y el escalador se serializa con el env. No se reajusta nunca, ni en test ni en vivo.
- **El reward se calcula sobre `equity_mark`**, que ya pagó comisión, spread y slippage:
  los costos están dentro por construcción, no restados después. No se usa
  `equity_liquidation` porque cobraría la fricción de salida en cada barra cuando en la
  realidad se paga una vez; la brecha entre ambas series va como feature para que el
  agente la vea.
- **El env no puentea la `RiskLayer`**: la pasa a `drive()` para que se aplique en el
  mismo punto que en vivo. Se recibe como **fábrica**, no como instancia: la capa acumula
  estado por episodio y reusarla arrastraría el kill switch de uno al siguiente.
- **El episodio arranca después del calentamiento** de los indicadores, enviando `None`
  en cada barra previa. Esas barras quedan en el `SimResult` sin fills, que es la verdad.
- **Gymnasium**: `terminated=True` solo por ruina (equity <= 0); agotarse los datos es
  `truncated=True`. La acción del último paso **expira sin ejecutarse** y da reward 0.

### Sobre el sintético de Heston como test de sanidad

`generate_gbm_sv` es GBM con volatilidad estocástica: `E[r_{t+1} | F_t] = mu·dt`
constante e independiente de la historia. **La dirección es impredecible por
construcción.** Un agente que no supera a buy-and-hold en retorno total sobre Heston
no tiene un bug: está en lo correcto, porque estar fuera del mercado cuesta drift y
además paga costos. Lo único explotable es el drift (que buy-and-hold captura entero) y
la volatilidad, que sí es mean-reverting y predecible.

El criterio de sanidad válido sobre Heston es más débil: **el agente debe converger a
estar casi totalmente invertido y no debe rotar**. Para un test de "¿puede aprender una
señal?" hace falta un fixture con estructura direccional deliberada (AR(1) con
reversión, alternancia de régimen), etiquetado como fixture y nunca como dataset de
investigación. Queda pendiente para la Etapa 3.

## Invariantes de los fixtures de validación (Etapa 3)

Ver `docs/adr/0002-fixtures-sinteticos-de-validacion.md`. Cinco niveles en
`data/fixtures.py`, cada uno aislando un fallo distinto.

- **Son infraestructura de validación, no datos de investigación.** Ningún
  resultado del estudio se reporta sobre ellos. La etiqueta no vive solo en el
  docstring: `series.source` empieza con `fixture:` y `Fixture` lo exige, así que
  viaja dentro de `SimResult.config` y cualquier corrida sobre un fixture es
  identificable en el tracking.
- **La señal va embebida en el precio, nunca en una columna aparte.** La
  observación del entorno es cerrada; una columna exógena sería invisible para el
  agente. El estado predictivo es el último log-retorno, o sea la feature
  `log_return_{lookback-1}`. Consecuencia: los niveles 0 y 1 validan el
  **pipeline**, no el descubrimiento de una señal escondida.
- **`SignalSpec` se parametriza por momentos estacionarios** (`drift`,
  `sigma_target`), no por `mu` y `sigma`. Así `R² = beta²` exactamente y la
  distribución marginal no se mueve al barrer el SNR ni al invertir el régimen.
  Sin eso, SNR y régimen de volatilidad quedarían confundidos.
- **Gap cero y sin redondeo a tick**, e instrumento sin mínimos ni lote. El techo
  tiene que ser exacto y alcanzable; un rechazo por `MIN_NOTIONAL` haría que el
  agente no lo alcanzara por un motivo ajeno a aprender. Estos fixtures **no**
  ejercitan gap ni restricciones de venue: eso es la Etapa 1 y el barrido de
  capital de la Etapa 6.
- **El camino intra-barra existe y no es decorativo.** Sin él
  `high = max(open, close)` y Corwin-Schultz devuelve cero: el nivel 2 correría
  con el spread apagado en silencio.
- **El techo se expone como curvas de equity, no como un número**, para que
  `eval.metrics` se aplique tal cual. `informed` es la barra a superar;
  `clairvoyant` es cota dura y **nunca** criterio de aprobación.
- **`ceilings(safety=...)` se pide con el `safety` con el que se corre.** Con
  `safety=1.0` el techo es teórico: el venue rechaza la orden por
  `INSUFFICIENT_CASH` porque no deja con qué pagar spread y comisión. Y la
  dirección del efecto no está garantizada sobre un camino realizado: exponerse
  menos puede terminar mejor. No escribas la aserción al revés.
- **Criterio único de los techos: equity terminal.** El reward del entorno (suma
  de retornos simples) tiene un óptimo a `sigma²/2` de distancia; queda declarado,
  no disimulado.
- **El óptimo del nivel 2 es numérico y se declara como tal.** Bellman de
  recompensa media sobre `(r_t, posición)`, resuelto en grilla. Se valida contra
  el umbral cerrado en el límite de costo cero y contra el clarividente por
  arriba. Su salida es una banda de no-operar que es cero **si y solo si** los
  costos son cero: esa banda *es* "operar selectivamente".
- **`Ceilings.capture()` devuelve `None` en el nivel 4, no `0.0`.** Sin brecha
  entre el techo y estar invertido la fracción no está definida, y un cero se
  leería como "no capturó nada" en vez de "no había nada que capturar". Mismo
  criterio que `win_rate` sin trades cerrados.
- **La banda de Bartlett (`1.96/sqrt(n)`) no aplica a los retornos de Heston.**
  Supone iid; con varianza condicional agrupada el estimador de la
  autocorrelación tiene más varianza. Con la banda ingenua el control negativo
  falla en lags aislados con series sanas. Se usa el error estándar robusto para
  diferencias de martingala.

## Invariantes del Agente A y del protocolo (Etapa 3)

Ver `docs/adr/0003-agente-a-y-protocolo-de-validacion.md`.

- **`torch` y `stable-baselines3` son el grupo opcional `rl`**, que
  `uv sync --frozen` no instala: la rueda de torch de PyPI arrastra el stack de
  CUDA (>3 GB) y sacaría al job principal de CI de sus 5 minutos. El núcleo de
  `agents/` —protocolo, reporte por semillas, walk-forward— no depende de torch
  y se testea con políticas deterministas inyectadas. El import de SB3 es
  diferido. **Consecuencia declarada: el adaptador de SB3 no está cubierto por
  el CI**; su test lleva el marcador `rl` y se saltea donde el grupo no está.
- **El agente y los baselines arrancan en la misma barra.** `WarmupDelay`
  envuelve una `Strategy` para que no opere antes del calentamiento. Sin eso, las
  26 barras de ventaja son retorno regalado a uno de los dos y la diferencia
  medida deja de ser atribuible a la estrategia. El techo se pide con el mismo
  `first_decision` y el mismo `safety`.
- **Los criterios de aprobación se declaran antes de correr** (`ProtocolThresholds`)
  y viajan serializados con el resultado. Un umbral elegido después de ver los
  números no es un criterio, es una descripción.
- **El protocolo para en el primer fallo** y deja los niveles no corridos como
  `SKIPPED` con su motivo. Un reporte con tres niveles y sin explicación se lee
  como si el protocolo tuviera tres niveles.
- **Hay tres veredictos, no dos.** El nivel 3 es `MEASURED`: adaptarse y
  memorizar son ambos hallazgos válidos, y forzar un PASS/FAIL sería inventar
  una hipótesis después del hecho.
- **El criterio del nivel 2 está sobre la rotación, no sobre el retorno.**
  Rendir menos con costos es automático —se restan del equity aunque el agente
  los ignore—; lo que prueba que la penalización llegó al reward es el cambio de
  comportamiento.
- **La métrica del barrido de SNR es `capture`, no el retorno crudo.** Al bajar
  el SNR el techo baja también, así que sin normalizar la degradación se mide a
  sí misma.
- **`summarize` falla con menos de 10 semillas.** El escape explícito enciende
  `below_minimum_seeds`, que sale en el JSON y en el render. Se puede producir un
  resultado con pocas semillas; lo que no se puede es que parezca uno con muchas.
- **El criterio del nivel 4 es un contraste, no un umbral.** El exceso sobre estar
  siempre invertido se compara contra la dispersión **entre caminos**, no contra un
  número fijo. El umbral anterior (`max_median_excess = 0.05`) no era demasiado laxo ni
  demasiado estricto: estaba mal planteado, porque medía una cantidad contra una escala
  que no le correspondía. `decompose_variance` exige al menos dos caminos y el nivel 4
  exige diez.
- **El `t` del drift sobre la ventana de entrenamiento va al lado del veredicto del
  nivel 4**, no en una nota al pie. Si el drift no es detectable en la muestra —en el
  régimen `medium`, con μ=8% y 30% de volatilidad sobre 4800 barras, la mediana medida
  sobre 10 caminos es 0.84, con rango −0.91 a +2.21— entonces
  "converger a estar invertido" le pide al agente aprender algo que la muestra no
  contiene, y un fallo del nivel no es un fallo del agente. `drift_t_statistic` lo
  calcula y `MultiPathResult.drift_detectable` lo contrasta.
- **Entrenar y juzgar son dos comandos** (`agents.cli arm` / `assemble`). Los
  criterios se aplican siempre sobre resultados guardados: revisar un umbral no
  exige reentrenar, y si alguien lo cambia después de ver los números, se ve en
  el diff. `ArmResult.from_dict` recalcula las distribuciones desde las corridas
  en vez de leerlas.
- **La política recurrente es una opción y se corre donde tiene sentido.** En
  los niveles 0, 1 y 2 el estado es completamente observable (el óptimo depende
  solo de `r_t`, que está en la observación) y la memoria solo agrega parámetros
  que sobreajustar. Donde importa es en el nivel 3.
- **Walk-forward: los tests de ventanas consecutivas no se solapan** con el paso
  por defecto, y `coverage` reporta `overlapping_test_bars` cuando sí. Con tests
  solapados hay más observaciones que información independiente, y hay que
  decirlo antes de calcular cualquier estadístico con ellas.

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
  configs/
tests/
docs/adr/
notebooks/    # solo exploración
```

## Etapas

1. **Simulador + baselines + `eval/`.** Cerrada. `data/`, `sim/` y `eval/`, CI en verde.
   Refactor de arquitectura para ejecución real aplicado sobre esta base: `ExecutionVenue`,
   `Clock`, `client_order_id` y `risk/`. 330 tests.
2. **Entorno Gymnasium** como wrapper delgado, con tests de anti-leakage. Cerrada.
   `envs/`, `features/` y `sim/sizing.py`; `drive()` como costura. 478 tests.
3. **Agente A** (solo precio), un activo, un régimen. ¿Supera buy-and-hold neto de costos?
   Fixtures sintéticos con señal conocida y óptimo calculable (`data/fixtures.py`) y
   PPO con el protocolo de validación de cinco niveles (`agents/`). **Pendiente el paso
   6**: datos reales con walk-forward, que necesita un dataset versionado que todavía
   no existe en el repositorio.
4. **Pipeline de noticias** con validación point-in-time estricta.
5. **Agente B** y comparación controlada contra A.
6. **Barrido** de regímenes y capital. `PortfolioSimulator` multi-activo.
7. **Paper trading** en vivo.

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
