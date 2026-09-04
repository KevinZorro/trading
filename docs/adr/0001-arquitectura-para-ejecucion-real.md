# ADR 0001 — Arquitectura para ejecución real

- **Estado:** aceptado
- **Fecha:** 2026-09-04
- **Contexto de etapa:** Etapa 1 cerrada (simulador, baselines, `eval/`). No hay
  entorno Gymnasium ni agentes todavía.

## Contexto

El sistema debe poder operar con dinero real en el futuro. No se construye
infraestructura de trading en vivo ahora, pero cada decisión que se tome hoy
determina si el mismo código de estrategia y de gestión de riesgo podrá correr
contra un broker sin ramas condicionales por entorno.

El costo de postergar esto no es lineal. Una vez que existan agentes entrenados,
resultados publicados y un pipeline de noticias, cambiar la interfaz de ejecución
significa reentrenar y revalidar todo. Y las tres cosas que más caro salen de
descubrir tarde —duplicación de órdenes por reintento, desincronización de
estado, ausencia de límites de riesgo— no se manifiestan en backtest: se
manifiestan la primera vez que hay una interrupción de red con dinero real.

Este ADR documenta cuatro decisiones tomadas como refactor **sin cambio
funcional** sobre el simulador de la Etapa 1.

## Decisión 1 — Protocolo `ExecutionVenue`

Se extrae `sim.venue.ExecutionVenue` con cinco métodos: `submit_order`,
`cancel_order`, `get_positions`, `get_fills`, `reconcile`. `Simulator` pasa a ser
una implementación; en la Etapa 7 un adaptador de broker será otra. El runner y
la capa de riesgo hablan con el protocolo y no saben cuál tienen enfrente.

Dos consecuencias de forma, que son el punto:

**`submit_order` devuelve `OrderAck`, no `Fill`.** Acusa recepción, no ejecución.
En vivo la orden viaja, el venue la acepta y el fill llega después por otro
canal. Un simulador que devolviera el fill en el mismo llamado enseñaría a
escribir estrategias que no pueden existir en vivo. Esto obliga a partir la
validación en dos tiempos, como un broker real:

| Momento | Qué se valida | Dónde |
|---|---|---|
| **Admisión** (al enviar) | identificador presente, no duplicado, venue activo | `submit_order` |
| **Ejecución** (al llenar) | volumen de la barra, mínimos del instrumento, cash | `_execute` |

La separación no es ceremonial: en el momento del envío, el volumen y el precio
de la barra de ejecución **todavía no existen**. Son de `t+1`. Validarlos al
enviar sería lookahead.

**`reconcile` existe desde el primer día**, aunque en simulación sea trivial. El
runner de la Etapa 7 no debe estrenar el paso de reconciliación el día que haya
dinero real. `run()` lo llama al arrancar y falla si el venue reporta posiciones
abiertas.

### Tensión con la garantía anti-leakage, y cómo se resuelve

El motor documenta que no expone un `step()` público, porque un caller que
pudiera adelantar el cursor leería la barra siguiente antes de decidir. Agregar
cinco métodos públicos toca ese contrato y hubo que verificarlo, no asumirlo.

Ninguno de los cinco recibe ni devuelve una barra, ninguno mueve el cursor `_t`,
y `MarketView` sigue siendo el único canal hacia los datos de mercado. Está
fijado por tres tests nuevos en `tests/sim/test_leakage.py`: que ningún método
público adelanta el cursor, que el venue detenido no ejecuta nada, y que las
salidas del protocolo no contienen barras. El test de superficie pública pasa de
`== {"run"}` a una lista blanca explícita de seis nombres, así que un séptimo
método público sigue rompiendo el test y obliga a justificarlo.

## Decisión 2 — `Clock` inyectable

`sim.clock.Clock` con dos implementaciones: `SimulatedClock`, que el motor avanza
barra por barra, y `SystemClock`, el único punto del repositorio que consulta el
reloj de pared.

En backtest "ahora" es el timestamp de la barra que se procesa, no la hora de la
máquina que corre el experimento. Sin la inyección, un límite de pérdida diaria
preguntaría a `datetime.now()` y el mismo backtest daría resultados distintos
corrido un martes o un domingo, y distintos según el huso horario.

`advance_to` prohíbe retroceder: si el reloj pudiera ir hacia atrás, el contador
del día se reiniciaría a mitad de corrida y el límite diario dejaría de morder.
`reset_to` es la única forma de rebobinar y existe porque una instancia de
`Simulator` se corre más de una vez (un barrido de semillas la reusa).

La regla se hace mecánica: `tests/sim/test_clock.py` recorre `src/` y falla si
algún módulo que no sea `clock.py` contiene `datetime.now`, `time.time()`,
`Timestamp.now` o `date.today`. Incluye un test del propio patrón, para que no
quede en verde por buscar algo que nunca coincide.

## Decisión 3 — `client_order_id` generado por el emisor

En vivo, si la respuesta a un envío se pierde por timeout, el emisor no sabe si
la orden llegó. Reintentar sin un identificador propio duplica la posición. Con
un `client_order_id` estable, el reintento lleva el mismo identificador y el
venue lo reconoce como duplicado en vez de ejecutarlo de nuevo.

`MarketOrder` gana el campo; el runner lo asigna antes de enviar si la estrategia
no trajo el suyo; `submit_order` **rechaza** cualquier orden que llegue sin él, y
mantiene un registro de identificadores vistos: reenviar uno conocido devuelve el
mismo acuse con `is_duplicate=True` y no encola nada.

### La tensión: reproducibilidad contra unicidad entre reinicios

Dos requisitos que ningún generador único satisface:

- **Reproducibilidad total** (principio 7 de CLAUDE.md): dos corridas del mismo
  backtest deben producir exactamente los mismos identificadores, o sus logs no
  son comparables. Esto descarta `uuid4()`.
- **Unicidad entre reinicios**: en vivo, si el proceso se reinicia y el contador
  vuelve a cero, la primera orden después del reinicio reusa un identificador ya
  visto. La protección contra duplicados se convierte en pérdida silenciosa de
  órdenes legítimas — el fallo es peor que el que se quería evitar, porque es
  silencioso.

**Se resuelve con un generador inyectable**, no eligiendo uno de los dos:

| Generador | Ámbito | Garantía |
|---|---|---|
| `SequentialIds` | simulación | `run_id-00000000`, determinista corrida tras corrida |
| `PrefixedSequentialIds` | vivo (Etapa 7) | prefijo del reloj al arrancar, único entre reinicios |

El prefijo se calcula una sola vez en la construcción y no vuelve a consultar el
reloj: dentro de una instancia la secuencia sigue siendo determinista y
auditable. `ASSUMPTION`: dos instancias del runner no arrancan dentro del mismo
nanosegundo; si alguna vez corren varios runners contra la misma cuenta, el
prefijo necesita un identificador de instancia explícito, no la resolución del
reloj.

## Decisión 4 — `RiskLayer` fuera del agente

Un agente de RL ante un estado fuera de distribución puede decidir cualquier
cosa, y no hay entrenamiento que garantice lo contrario. El límite de riesgo es
lo que separa una mala racha de una pérdida total, y no puede depender de que la
política aprendida se comporte.

`risk.RiskLayer` implementa `sim.gate.OrderGate` y se aplica en un único punto:
entre la decisión de la estrategia y el envío al venue. El runner en vivo la
inserta en el mismo lugar. La dependencia va `risk` → `sim` y nunca al revés: el
simulador no sabe que la capa existe, igual que un broker real no lo sabe.

Límites: posición máxima (nocional y como fracción del equity), pérdida máxima
diaria, turnover diario máximo y kill switch.

### 4a. Rechaza, nunca redimensiona

Misma regla que ya cumple `check_tradable`. Una capa que recorta una orden al
límite hace indistinguible *"el agente pidió esto"* de *"el límite lo recortó
hasta acá"*, que es exactamente el borrado de información que el proyecto prohíbe
en los baselines que se autocensuran. `GateDecision` ni siquiera tiene un campo
donde devolver una orden modificada, y hay un test que lo verifica sobre los
campos de la dataclass.

Los vetos van a `SimResult.gate_rejections`, **lista separada** de los rechazos
del venue. *"El venue no pudo llenarla"* y *"nuestra capa de riesgo no la dejó
salir"* son dos diagnósticos distintos; agregarlos en una sola cifra los volvería
indistinguibles, que es el mismo error que agregar el gap a los costos.

### 4b. El kill switch permite cerrar, no abrir

Un kill switch que bloquea todo deja la posición abierta justo cuando algo salió
mal, que suele ser peor que el riesgo que lo disparó. Solo pasan las órdenes que
reducen exposición: con `allow_short=False`, vender teniendo posición.

La reposición es **manual**. Si se repusiera solo, no sería un kill switch sino
una pausa.

### 4c. El ancla del día es el cierre del día anterior

Descubierto al escribir el test de integración: con barras diarias —donde una
barra *es* un día entero— anclar el límite al equity de la propia barra compara
el equity contra sí mismo y el límite no muerde nunca. El ancla del día nuevo es
el equity con que cerró el anterior. Con barras intradiarias las dos definiciones
casi coinciden; con diarias, la diferencia es entre un límite que funciona y uno
decorativo.

También es un límite de **pérdida diaria**, no de drawdown intradiario: el ancla
es la apertura del día, no el máximo alcanzado. Hay un test que fija la elección.

`ASSUMPTION`: el día de riesgo es el día calendario UTC, no la sesión del
mercado. Para una acción que opera de 9:30 a 16:00 en Nueva York coinciden; para
cripto, que opera 24/7, el corte a medianoche UTC es una convención y no un hecho
del mercado. Cuando la Etapa 6 agregue instrumentos con sesiones distintas, esto
debe preguntarle al `Calendar`.

## Consecuencias

Positivas: el código de estrategia y de riesgo escrito contra el simulador corre
contra un broker sin cambios. La reconciliación y la idempotencia existen antes
de que hagan falta. La regla del reloj y la superficie del motor están fijadas
por tests, no por disciplina.

Negativas: la superficie pública de `Simulator` pasó de un método a seis. El
motor tiene estado de instancia (libro, fills, orden encolada) donde antes era
todo local a `run()`, lo que obligó a un `_reset_venue()` explícito — y el test
que corre dos veces el mismo simulador atrapó una regresión real durante el
desarrollo: el reloj no se rebobinaba.

## Qué falta para operar en vivo

Nada de esto está construido. Es el inventario de lo que la Etapa 7 tiene que
resolver, con lo que ya está resuelto marcado.

| Pieza | Estado | Qué falta |
|---|---|---|
| Interfaz de ejecución | ✅ `ExecutionVenue` | — |
| Tiempo inyectable | ✅ `Clock` | — |
| Idempotencia de órdenes | ✅ `client_order_id` | Generador con prefijo de instancia en producción |
| Capa de riesgo | ✅ `RiskLayer` | Límites calibrados con datos reales, no supuestos |
| **Adaptador de broker** | ❌ | Implementar `ExecutionVenue` contra una API real: autenticación, reintentos con backoff, traducción de códigos de error del venue a `RejectReason`, manejo de fills parciales asíncronos y de rechazos que llegan después del acuse |
| **Feed en tiempo real** | ❌ | Reemplazar `BarSeries` por un stream. La `MarketView` debe seguir sin poder ver el futuro, lo que en vivo es automático pero exige decidir qué pasa con barras incompletas y datos que llegan tarde o fuera de orden |
| **Persistencia de estado** | ❌ | El proceso se reinicia: hay que persistir identificadores emitidos, posición conocida y estado de la capa de riesgo (kill switch, ancla del día, turnover). Sin esto, un reinicio repone el kill switch en silencio |
| **Reconciliación al arranque** | ⚠️ parcial | `reconcile()` existe y el runner lo llama, pero en simulación no puede haber discrepancia. Falta el caso real: qué hacer cuando el venue reporta una posición distinta de la persistida. La respuesta por defecto debe ser **detenerse y alertar**, nunca adivinar |
| **Monitoreo y alertas** | ❌ | Latencia de órdenes, tasa de rechazos, desvío entre slippage real y simulado, brecha entre `equity_mark` y `equity_liquidation`, heartbeat del feed. Una alerta que nadie mira no es monitoreo |
| **Paper trading 3–6 meses** | ❌ | Gate obligatorio antes de cualquier capital real. Compara ejecución real contra lo que el simulador predijo para las mismas señales. Si el slippage real difiere materialmente, todos los resultados del estudio se releen con ese factor |

## Alternativas descartadas

**Un `SimulatedVenue` separado del `Simulator`.** Habría dejado la superficie
pública del motor en `{"run"}` y una separación de responsabilidades más nítida
entre "el que conduce el bucle" y "el que ejecuta". Se descartó porque el
simulador ya *es* el venue —tiene el libro, la serie y el modelo de costos— y
partirlo habría creado dos objetos que se pasan el mismo estado, con más
superficie de sincronización y ninguna garantía adicional: los tests muestran que
los métodos del protocolo no comprometen el anti-leakage.

**`uuid4()` para los identificadores.** Resuelve la unicidad y rompe la
reproducibilidad, que es un principio no negociable del proyecto.

**Redimensionar en vez de rechazar en la capa de riesgo.** Deja al agente operar
al límite en vez de no operar, pero borra la distinción entre lo que el agente
pidió y lo que el límite permitió. Es el mismo anti-patrón que el proyecto ya
prohíbe en los baselines.

**Que `sim` importe `risk` directamente.** Funcionaría sin ciclos, pero el venue
pasaría a conocer la política de riesgo. El `Protocol` en `sim.gate` mantiene la
dependencia en un solo sentido, que es lo que permite insertar la misma capa en
el runner en vivo.
