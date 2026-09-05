# ADR 0002 — Fixtures sintéticos con señal conocida y óptimo calculable

- **Estado:** aceptado
- **Fecha:** 2026-09-05
- **Contexto de etapa:** Etapa 3, primera mitad. Existen `data/`, `sim/`, `eval/`,
  `risk/`, `features/` y `envs/`. Todavía no hay agentes de RL.

## Contexto

La Etapa 3 entrena el primer agente. El problema no es entrenarlo: es saber si lo
que salga significa algo. Un agente que rinde 12% anual sobre datos reales no se
puede evaluar, porque nadie sabe si el máximo alcanzable era 13% o 400%. Y un
agente que rinde mal puede estar mal entrenado, mal conectado al simulador, o
perfectamente sano frente a datos sin señal.

Esas tres causas se confunden entre sí y ninguna se distingue con datos reales.
Hacen falta series donde la respuesta correcta se conozca de antemano.

El `CLAUDE.md` ya anticipa el problema al hablar del generador de Heston: sobre
`generate_gbm_sv` la dirección es impredecible por construcción, así que un
agente que no supera a buy-and-hold **está en lo correcto**. Heston sirve como
control negativo y no sirve para nada más. Para preguntar "¿puede aprender una
señal?" hace falta una serie que tenga una.

## Decisión 1 — La señal va embebida en el precio, no en una columna

`envs.observation.ObservationBuilder` construye un vector cerrado de 19 features
derivadas del precio y de la cuenta. Una columna exógena de señal —lo primero que
uno haría— **sería invisible para el agente**: el entorno no tiene por dónde
pasarla.

Por lo tanto el estado predictivo es el último log-retorno, que el agente ya ve
como la feature `log_return_{lookback-1}`. El proceso es un AR(1) sobre
log-retornos:

```
r_{t+1} = mu + beta * r_t + sigma * eps_{t+1}
```

Consecuencia que hay que decir en voz alta: los niveles 0 y 1 validan **el
pipeline**, no la capacidad de descubrir una señal escondida. La señal está en
una feature cruda de la observación. Que PPO los resuelva no dice nada sobre
datos reales, y no se debe reportar como si lo dijera.

## Decisión 2 — Parametrizar por momentos estacionarios, no por `mu` y `sigma`

`SignalSpec` toma `drift` y `sigma_target` —la media y el desvío marginales— y
deriva los parámetros del proceso:

```
mu(beta)    = drift * (1 - beta)
sigma(beta) = sigma_target * sqrt(1 - beta^2)
```

Dos razones, y ninguna es cosmética:

1. El R² del pronóstico a un paso queda **exactamente** `beta²`. El barrido de
   SNR del nivel 1 es literalmente un barrido de `beta`: no hay que estimar el
   SNR, se declara.
2. La media y la varianza marginales quedan fijas al mover `beta`. Sin esto, el
   barrido de SNR movería también el régimen de volatilidad, y un agente que
   rinde peor con menos SNR podría estar rindiendo peor por operar en otro
   régimen. Las dos explicaciones serían inseparables. Lo mismo vale para el
   cambio de régimen del nivel 3: invierte el signo de la predictibilidad y
   **nada más**.

## Decisión 3 — Gap cero, sin redondeo a tick, instrumento sin mínimos

El camino intra-barra es continuo entre barras, así que `open[t+1] == close[t]`
exactamente. El precio con el que se dimensiona la orden y el de referencia de la
ejecución son el mismo número, mantener la posición durante `t+1` captura
exactamente `r_{t+1}`, y el techo es exacto en vez de depender de la realización
del gap.

Por el mismo motivo el instrumento del fixture es fraccional, con tick de 1e-8 y
sin nocional mínimo. El trabajo de este objeto es ser una regla de medir: un
rechazo por `MIN_NOTIONAL` haría que el agente no alcanzara el techo por un
motivo ajeno a aprender la señal, y el diagnóstico se volvería ambiguo.

**Es una limitación, no una mejora.** Estos fixtures no ejercitan la ruta de gap
del motor ni las restricciones de venue. Lo primero lo cubren los tests de la
Etapa 1; lo segundo es exactamente lo que mide el barrido de capital de la Etapa
6, y ahí se mide contra volúmenes reales.

Sí se conserva el camino intra-barra estocástico (o determinista en el nivel 0):
sin él, `high = max(open, close)` y `CorwinSchultzSpread` devolvería cero, con lo
cual el nivel 2 correría **con el spread apagado en silencio**.

## Decisión 4 — El techo se expone como cuatro curvas, no como un número

`Fixture.ceilings()` devuelve series de equity, no escalares, para que
`eval.metrics` se aplique tal cual: el techo tiene Sharpe, drawdown y CAGR igual
que cualquier corrida. Comparar solo retornos totales contra un techo esconde que
el techo puede alcanzarse por caminos con riesgo muy distinto.

| Curva | Qué es | Cómo se usa |
|---|---|---|
| `informed` | óptimo ex-ante: conoce el proceso, no el ruido | **la barra a superar** |
| `clairvoyant` | óptimo ex-post sobre el camino realizado | cota dura, nunca criterio |
| `always_long` | acción 1.0 en cada barra | referencia inferior |
| `always_flat` | no operar, devengando `cash_rate` | referencia inferior |
| `memorizer` | regla de la primera mitad aplicada a toda la serie (nivel 3) | mide memorización |

`clairvoyant` exige ver el ruido y ningún agente puede alcanzarlo. Se expone para
acotar por arriba y nunca como barra de aprobación; usarlo como criterio sería
declarar fracasado a un agente óptimo.

`Ceilings.capture()` normaliza el resultado entre `always_long` (0) e `informed`
(1). Es la métrica con la que hay que reportar el barrido de SNR: el retorno
crudo no sirve porque al bajar el SNR **el techo baja también**, y una
degradación trivial del techo se leería como degradación del agente.

## Decisión 5 — El techo depende de `safety`, y por eso se pide con el propio

`TargetWeightSizer` usa `safety = 0.98`: una acción de 1.0 invierte el 98% del
equity, porque estar exactamente all-in exigiría conocer el precio de ejecución
antes de enviar la orden.

`ceilings(safety=...)` es por lo tanto un parámetro obligatorio de hecho. El
techo con `safety=1.0` es **teórico y el venue lo rechazaría**: la orden pide
todo el cash y no deja con qué pagar spread y comisión, así que el motor la
rechaza entera por `INSUFFICIENT_CASH` y la curva de equity se queda plana. Hay
un test que lo demuestra corriendo `Simulator`.

El diseño inicial de esta capa afirmaba que el techo con `safety=0.98` es
estrictamente menor que con `safety=1.0`. **Es falso sobre un camino realizado**,
y el test lo detectó: exponerse menos puede terminar mejor cuando los retornos
realizados salen peores que su media condicional. Medido en el nivel 1,
`safety=0.98` termina 1% por encima. Lo que se afirma es que el techo *cambia*, y
la dirección solo se garantiza sin ruido.

## Decisión 6 — Criterio único: equity terminal

Todos los techos maximizan el equity terminal, equivalente a maximizar el
crecimiento logarítmico medio por barra. Es lo que miden el retorno total y el
CAGR, que son las métricas primarias del estudio.

**El reward del entorno tiene un óptimo marginalmente distinto y no se disimula.**
`NetReturnReward` es la suma no descontada de retornos simples; su umbral óptimo
está en `E[R] > rate` y el de aquí en `E[log(1+R)] > log(1+rate)`, o sea a
`sigma²/2` de distancia. Con `sigma = 1.2%` por barra eso son 7e-5 contra un edge
típico de 2e-3: un 3% del edge. Queda declarado en el módulo y en el docstring.

Sin costos el óptimo es miope y cerrado —con `safety=1` y sin cash rate,
`w_t = 1` si y solo si `mu + beta*r_t > 0`—, porque las decisiones se desacoplan.

## Decisión 7 — El óptimo del nivel 2 es numérico, y se declara como tal

Con costos el óptimo deja de ser miope: cambiar de estado cuesta, así que la
decisión depende de si se está dentro o fuera. `solve_optimal_policy` resuelve la
ecuación de Bellman de recompensa media sobre el estado `(r_t, posición)` por
iteración de valor relativa sobre una grilla de 201 puntos.

El planteo es analítico; **la resolución no tiene forma cerrada** y llamarla
"fórmula" sería falso. Se ata a dos cosas verificables: con costo cero converge
al umbral cerrado `r_t < -mu/beta`, y su resultado sobre el camino queda siempre
por debajo del clarividente.

El resultado es una `HysteresisPolicy` con dos umbrales: se entra con un edge más
exigente del que hace falta para quedarse. Esa brecha —`no_trade_band`— **es** lo
que "operar selectivamente" significa, y es cero si y solo si los costos son
cero. Medido en el nivel 2 con 2000 barras y la semilla de los tests: el óptimo con
costos rota un 24% menos (912 cambios de estado contra 1200) y termina en 8.63x
contra 7.45x de la misma regla ignorando los costos.

## Decisión 8 — La etiqueta de "no es dato de investigación" viaja con los datos

`series.source` empieza con `fixture:` y `Fixture.__post_init__` lo exige. Como
`source` termina dentro de `SimResult.config` vía `BarSeries.meta()`, cualquier
resultado corrido sobre un fixture es identificable en el tracking de
experimentos sin depender de que alguien se acuerde de anotarlo.

Un párrafo en un docstring se ignora; una etiqueta que viaja dentro del resultado
serializado, no.

## Lo que estos fixtures no prueban

- **No prueban nada sobre mercados reales.** Un AR(1) en retornos es una
  caricatura y no hay nada aquí que sugiera que los mercados la tengan.
- **No ejercitan gap ni restricciones de venue** (Decisión 3).
- **Los niveles 0 y 1 validan el pipeline, no el descubrimiento de señal**
  (Decisión 1).
- **El nivel 4 hereda una limitación del generador de la Etapa 1**:
  `generate_gbm_sv` usa un solo stream aleatorio, así que pedir 2000 barras no
  produce en las primeras 1000 las mismas barras que pedir 1000. No es una fuga
  —dentro de una serie el camino se construye hacia adelante—, es una limitación
  de reproducibilidad entre longitudes distintas. Queda fijada por test.

## Consecuencias para la Parte B

El protocolo de validación corre los niveles en orden y **para** en el primero
que falle:

1. Si el agente no se acerca a `informed` en el nivel 0, el bug está en el
   pipeline. No se tocan hiperparámetros.
2. En el nivel 1, `capture()` debe degradar suavemente al bajar `|beta|`.
3. En el nivel 2 debe rotar **menos** que en el nivel 1, no solo rendir menos.
   Si opera igual con y sin costos, la penalización no está llegando al reward.
4. En el nivel 3, `memorizer` da la referencia contra la cual se mide si se
   adapta o memoriza. Ambos son resultados válidos si quedan medidos.
5. En el nivel 4 debe converger a `always_long` sin rotar. `capture()` devuelve
   `None` ahí a propósito: sin brecha entre el techo y estar invertido, la
   fracción no está definida, y un cero se leería como "no capturó nada" cuando
   lo cierto es "no había nada que capturar".

Los techos se piden siempre con el `safety` y el `first_decision` (calentamiento)
con los que corre el agente.
