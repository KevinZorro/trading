# ADR 0003 — Agente A: PPO, protocolo de validación y dependencia opcional

- **Estado:** aceptado
- **Fecha:** 2026-09-05
- **Contexto de etapa:** Etapa 3, segunda mitad. Los fixtures de validación
  (ADR 0002) están mergeados. Existe el entorno Gymnasium de la Etapa 2.

## Contexto

Entrenar un agente es fácil; saber si lo que sale significa algo, no. Un agente
que rinde mal puede estar mal entrenado, mal conectado al simulador, o
perfectamente sano frente a datos sin señal. Un agente que rinde bien puede
estar aprendiendo, o puede estar viendo el futuro por un bug en la observación.

La escalera de fixtures existe justamente para separar esas causas. Este ADR
documenta cómo se recorre, y las decisiones de arquitectura que hicieron falta.

## Decisión 1 — `torch` es una dependencia **opcional**

`stable-baselines3` y `sb3-contrib` viven en el grupo `rl` de `pyproject.toml`,
que `uv sync --frozen` **no** instala.

El motivo es concreto: la rueda de `torch` de PyPI para Linux arrastra el stack
completo de CUDA (más de 3 GB instalados). Ponerla como dependencia por defecto
saca al job principal de CI de su presupuesto de 5 minutos, y lo hace en cada
corrida de cada PR, para un job que no entrena nada.

La consecuencia de diseño es que **el núcleo de `agents/` no depende de torch**:
el protocolo de validación, el reporte por semillas, el walk-forward y los
criterios de aprobación se testean con políticas deterministas inyectadas. El
import de SB3 está diferido dentro de `agents.ppo` y falla con un mensaje que
dice cómo arreglarlo.

**Costo asumido y declarado: el adaptador de SB3 no está cubierto por el CI.**
Existe `tests/agents/test_ppo_smoke.py`, marcado `rl` y `slow`, que verifica el
cableado (rango de acciones, determinismo de la evaluación, reinicio del estado
de la LSTM), pero se saltea donde el grupo no está instalado. No se agregó un
job de CI que lo instale porque el índice de ruedas de CPU de PyTorch
(`download.pytorch.org`) no era alcanzable desde el entorno donde se desarrolló
esto, y agregar un job que no se pudo verificar es peor que no agregarlo.

## Decisión 2 — El agente y los baselines arrancan en la misma barra

`agents.policy.WarmupDelay` envuelve una `Strategy` para que no opere antes del
calentamiento de los indicadores.

Sin esto la comparación está sesgada y no de forma menor. El entorno empieza a
decidir en la barra 26 —antes, MACD y Bollinger no están definidos— mientras que
`BuyAndHold` compra en la barra 0. Sobre una serie con drift, esas 26 barras son
retorno regalado al baseline; sobre una que empieza cayendo, al agente. En los
dos casos la diferencia medida deja de ser atribuible a la estrategia.

El techo del fixture se pide con el mismo `first_decision` por la misma razón.

## Decisión 3 — Los criterios se declaran antes de correr y viajan serializados

`ProtocolThresholds` es un dataclass congelado con un número por criterio, cada
uno con su justificación en el docstring, y se guarda dentro del JSON del
resultado. Un umbral elegido después de ver los números no es un criterio, es
una descripción.

Los cinco criterios, tal como quedaron declarados:

| Nivel | Criterio | Por qué ese número |
|---|---|---|
| 0 | mediana de `capture` ≥ 0.80 | Sin ruido y con la señal en una feature cruda, un agente sano lo resuelve casi entero. |
| 1 | ordenado por SNR decreciente, `capture` no sube más de 0.10 al bajar el SNR; el brazo de mayor SNR captura ≥ 0.50 | Con 10 semillas el error estándar de la mediana ronda 0.05: exigir monotonía estricta haría fallar por ruido muestral. |
| 2 | la rotación anualizada mediana cae ≥ 10% respecto del mismo proceso sin costos | Rendir menos con costos es automático. Lo que prueba que la penalización llegó al reward es el cambio de **comportamiento**. |
| 3 | **sin criterio** | Adaptarse y memorizar son dos hallazgos válidos. Poner un umbral sería inventar una hipótesis después del hecho. |
| 4 | tiempo invertido mediano ≥ 0.80 y exceso de crecimiento sobre estar siempre invertido ≤ 0.05 | Sobre Heston el óptimo **es** estar invertido. Ganarle a una serie sin señal es sobreajuste. |

La rotación **no** es criterio en el nivel 4, aunque el enunciado del protocolo
diga "sin rotar". El único baseline con rotación comparable sería estar siempre
invertido, que rebalancea al peso objetivo en cada barra igual que el agente,
mientras que `BuyAndHold` compra una vez y su rotación anualizada es ~0.05: un
criterio contra esa referencia daría ratios de tres cifras y fallaría siempre,
incluso para un agente perfectamente convergido. Lo que sí captura "no rota" es
el tiempo invertido: un agente que se queda dentro el 95% del tiempo no está
entrando y saliendo. Un umbral de rotación llegó a estar declarado en
`ProtocolThresholds` sin aplicarse en el veredicto; se quitó al detectarlo,
antes de correr el nivel 4. Un criterio declarado que no se evalúa es peor que
no tenerlo, porque el reporte afirma haberlo verificado.

El nivel 3 tiene veredicto `MEASURED`, un tercer valor junto a `PASS` y `FAIL`.
Forzarlo a binario habría obligado a inventar una hipótesis.

## Decisión 4 — El protocolo para en el primer fallo, y lo dice

Si el nivel 0 falla, seguir al nivel 1 solo produce números que hay que
descartar después. Peor: invita a tocar hiperparámetros para arreglar un síntoma
cuya causa está en otro lado. El reporte deja los niveles no corridos como
`SKIPPED` **con su motivo**, en vez de omitirlos: un reporte con tres niveles y
sin explicación se lee como si el protocolo tuviera tres niveles.

## Decisión 5 — Entrenar y juzgar son dos comandos

`python -m agents.cli arm` entrena y evalúa **un** brazo y deja su JSON;
`assemble` lee los JSON y aplica los criterios.

Dos razones. La primera es práctica: 90 configuraciones en un solo proceso tarda
horas y no se paraleliza por dentro sin que los procesos de torch se peleen por
los mismos núcleos. La segunda importa más: los criterios se aplican **siempre**
sobre resultados guardados, así que revisar un umbral no exige reentrenar, y si
alguien lo cambia después de ver los números, se ve en el diff.

`ArmResult.from_dict` recalcula las distribuciones desde las corridas en vez de
leerlas, así que un archivo con distribuciones inconsistentes con sus corridas
no puede pasar inadvertido.

## Decisión 6 — Menos de 10 semillas es un error, no una advertencia

`eval.distribution.summarize` **falla** con menos de `MIN_SEEDS`. El escape
(`allow_fewer_seeds=True`) enciende `below_minimum_seeds`, que viaja dentro del
JSON y sale impreso en el render como `[MENOS DE 10 SEMILLAS]`.

Un resultado con pocas semillas se puede producir; lo que no se puede es que
parezca uno con muchas.

Sobre significancia: comparar 10 semillas contra un baseline y quedarse con el
p-value del mejor es el ejemplo de manual de comparaciones múltiples. Por eso
`eval.distribution` implementa el **Deflated Sharpe Ratio** (Bailey y López de
Prado, 2014), que descuenta explícitamente cuántas configuraciones se probaron.

## Decisión 7 — La métrica del barrido de SNR es `capture`, no el retorno

Al bajar el SNR **el techo también baja**. Una caída del retorno crudo es
compatible con un agente que captura exactamente la misma fracción de lo
capturable: sin normalizar, la degradación se mediría a sí misma.

`Ceilings.capture()` normaliza entre estar siempre invertido (0) y el óptimo
informado (1), y devuelve `None` en el nivel 4, donde la fracción no está
definida porque el techo coincide con la referencia.

## Decisión 8 — La política recurrente es una opción, no el default

El estado del mercado es parcialmente observable y una LSTM puede inferir el
régimen de la historia reciente. Eso es exactamente lo que el nivel 3 pone a
prueba.

Pero en los niveles 0, 1 y 2 el estado **sí** es completamente observable: el
óptimo depende solo de `r_t`, que está en la observación. Ahí la memoria no
puede ayudar y solo agrega parámetros que sobreajustar. Por eso el brazo
recurrente se corre donde tiene sentido —el nivel 3— y no como reemplazo del MLP
en todos lados.

## Lo que este ADR no cubre

- **El paso 6 del protocolo (datos reales) no se ejecutó**: no hay ningún
  dataset real versionado en el repositorio, y la suite corre sin red por
  diseño. La maquinaria de walk-forward está construida y testeada; falta la
  decisión de qué fuente, qué símbolo y qué período, que es del autor del
  estudio y tiene consecuencias de point-in-time, licencia y survivorship.
- **La reproducibilidad de torch no es bit a bit entre arquitecturas.** Por eso
  todo se reporta como distribución sobre semillas y la procedencia (versiones,
  commit) se guarda junto a cada resultado.
