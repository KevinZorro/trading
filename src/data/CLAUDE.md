# CLAUDE.md — `src/data/`

Se carga al trabajar en `src/data/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual.

## Invariantes de datos consolidados en la Etapa 1

- Fraccionales, tick, lote, mínimos y comisiones son propiedad de `InstrumentSpec` por
  instrumento, no configuración global.
- Precios sin ajustar para ejecución; factores de ajuste en columnas separadas.
  `total_return_index` es point-in-time; `backward_adjusted_close` exige
  `allow_lookahead=True`.
- El generador sintético de Heston produce caminos continuos y por tanto gap cero.
  `overnight_gap_frac` existe para hacer testeable ese efecto.

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
- **La señal va embebida en el precio, nunca en una columna aparte, salvo en el
  nivel 5a** (ADR 0012 §5), que existe justamente para probar que una columna
  exógena le llega a B: ahí `tone_mean` predice `r_{t+1}` con `corr = beta = 0,3` y
  el precio casi no contiene nada (`R² = 0,0024`). El 5b es el 4b más siete columnas
  de ruido. Los niveles 0 a 4 siguen con la regla. Para el Agente A sobre el 5b,
  `Fixture.without_exog()`; el 5a no lo admite porque su techo usa la columna.
  En los demás niveles la observación del entorno es cerrada y una columna
  exógena sería invisible para el agente. El estado predictivo es el último log-retorno, o sea la feature
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
