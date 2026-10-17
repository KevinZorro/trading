# CLAUDE.md — `src/sim/`

Se carga al trabajar en `src/sim/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual.

## Invariantes consolidados en la Etapa 1

No los redefinas. Impórtalos o respétalos.

- `allow_short=False`. El flag existe en la firma y lanza `NotImplementedError`.
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
- Corwin-Schultz para estimar spread desde high-low, no el rango crudo. Devuelve spread
  relativo; el costo de cruzar es la mitad.
- `cash_rate` configurable, 0.0 por defecto.

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

## Costura con el entorno y dimensionamiento (Etapa 2)

- **`Simulator.drive()` es la costura de inversión de control.** Generador que cede en
  cada punto de decisión y recibe la orden por `send()`. Se admite como séptimo método
  público porque **`send` fusiona avanzar y decidir en una operación atómica**: para
  llegar a `t+1` hay que entregar la decisión de `t`, así que no se puede adelantar el
  cursor, mirar y volver. `next(gen)` equivale a `send(None)`, o sea "decidí no operar".
- **`run` y `drive` son dos bucles públicos, no dos implementaciones.** Los cuatro pasos
  por barra (`_start`, `_open_bar`, `_dispatch`, `_finish`) son la única copia de la
  contabilidad. Si agregas un tercer bucle, consume esos helpers.
- **Dimensionar es ejecución: `sim.sizing.TargetWeightSizer`, no el env.** El env y
  cualquier estrategia directa usan el mismo sizer. Si el env tuviera el suyo, el
  backtest dejaría de corresponder al simulador validado.
- **El sizer no pre-redondea a cero ni recorta contra el cash.** `round_qty` redondea
  hacia cero: redondear ahí convertiría un delta chico en `qty=0` y la orden
  desaparecería del log. Se manda sin redondear y el venue rechaza con su motivo.
  `deadband=0.0` por defecto; una banda muerta se configura explícita y se serializa.
- **Un delta que es cero numérico no es una orden** ("no quise"); un delta real que se
  redondea a cero sí lo es, y el venue lo rechaza ("no pude"). El ledger acumula
  residuo de punto flotante (~1e-18 BTC tras cerrar una posición), así que "cero" se
  evalúa con tolerancia relativa (`sim.sizing.CERO_NUMERICO_REL = 1e-12` del equity en
  nocional), no con igualdad exacta: la regla literal "diferencia exactamente 0" no
  resolvía nada. Medido en una muestra in-sample posterior a C3: 61 de 70 rechazos por
  redondeo eran residuo (≤ 3.9e-18 BTC) y 9 eran deltas reales bajo el lote (≥ 1.4e-6
  BTC); el umbral cae en el medio de doce órdenes de magnitud.
  **Este cambio altera la observación del agente** (`last_order_rejected` deja de marcar
  "no pude" donde correspondía "no quise") y el conteo de órdenes: **los resultados
  posteriores no son comparables con C3**. El sizer entra al digest del sello del
  walk-forward, así que C3 no se puede reanudar con el sizer nuevo.
- **`safety` define qué significa la acción, no censura órdenes.** Un peso de 1.0 es "lo
  más invertido que se puede estar sin conocer el precio de ejecución" (98% por defecto).
  Estar exactamente all-in exigiría lookahead.
