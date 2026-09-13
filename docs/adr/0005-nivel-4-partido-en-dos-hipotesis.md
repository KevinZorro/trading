# ADR 0005 — El nivel 4 mezclaba dos hipótesis; se parte en 4a y 4b

- **Estado:** aceptado
- **Fecha:** 2026-09-11
- **Contexto de etapa:** Etapa 3. El criterio del exceso del nivel 4 ya pasa
  (ADR 0004). **El nivel 4b no se ha corrido todavía a la fecha de este ADR.**

## Contexto

El nivel 4 tenía dos criterios y por lo tanto contestaba dos preguntas con un
solo veredicto:

1. ¿El agente **inventa señal** donde no la hay? — el exceso sobre estar
   invertido, contrastado contra la dispersión entre caminos.
2. ¿El agente **converge a estar invertido**? — el tiempo invertido.

El ADR 0004 arregló la primera y midió algo que vuelve insostenible a la
segunda: el `t` del drift sobre la ventana de entrenamiento tiene **mediana
0.84** sobre diez caminos y valor poblacional **0.51**. El drift no está en la
muestra.

Sobre una serie donde el drift no es detectable, **abstenerse no es un error**.
Es la respuesta defendible: no hay nada que lleve al agente a invertirse. Un
tiempo invertido de 0.33 no puede hacer fallar el control negativo, porque el
control negativo pregunta otra cosa.

Y sin embargo la pregunta "¿reconoce drift cuando existe?" es legítima e
importante: un agente que nunca se invierte, ni cuando el drift es obvio, es un
agente roto. Lo que hacía falta no era relajar el criterio sino **separarlo, y
darle un fixture donde sea contestable**.

## Decisión 1 — 4a: control negativo puro, un solo criterio

Fixture: `level_4_control` (Heston `medium`, μ=0.08). Sin cambios.

**Criterio único:** sobre ≥ 10 caminos independientes, el exceso de crecimiento
sobre estar siempre invertido **no** se distingue de cero contrastado con la
dispersión entre caminos, con `t` de dos colas al 95%.

El tiempo invertido **sale del veredicto** y se sigue reportando como evidencia,
junto con el `t` del drift, para que quien lea sepa que abstenerse era
defendible en ese fixture. Un nivel que no mide algo no es un nivel que deba
ocultarlo.

**Esto no relaja 4a.** El criterio que fallaba —el del exceso— sigue idéntico y
sigue pudiendo fallar; hay un test que lo verifica con un exceso consistente
entre caminos. Lo que se quitó fue una segunda hipótesis que nunca perteneció
ahí.

## Decisión 2 — 4b: un fixture nuevo con drift detectable

Fixture: `level_4b_detectable_drift`. **Idéntico al 4a salvo por `mu`.** Que
cambie una sola cosa es el punto: si el agente se comporta distinto entre 4a y
4b, la única explicación disponible es el drift.

### La calibración, con la fórmula correcta

El generador integra `d log S = (μ − v_t/2) dt + √v_t dW`, así que el drift de
los **log-retornos** no es `μ·dt` sino `(μ − v_t/2)·dt`. Usar el aritmético fue
el error que produjo la cifra ≈1.8 que se publicó en el PR #7 y se corrigió en
el ADR 0004.

El estadístico sobre `n` barras de entrenamiento:

```
t = media_por_barra / (desvío_por_barra / √n)
  = [(μ − θ/2)/bpy] / [√(θ/bpy) / √n]
  = (μ − θ/2)·√n / √(θ·bpy)
```

Con `θ = 0.09`, `bpy = 252` y `n = 4800` (el 60% de 8000 barras que es el tramo
de train), el coeficiente es `√4800/√(0.09·252) = 14.55`:

| μ | `t` poblacional |
|---|---|
| 0.08 (nivel 4a) | **0.51** |
| 0.20 | 2.25 |
| 0.30 | 3.71 |
| **0.42 (nivel 4b)** | **5.46** |

Se elige **μ = 0.42**. El `t` muestral se distribuye alrededor del poblacional
con desvío ≈ 1, así que `P(t < 3) ≈ 0.7%`: prácticamente todo camino supera el
umbral. Verificado sobre los diez caminos del estudio: `t` entre **3.82 y 8.21**,
mediana 5.77, los diez por encima de 3.

**No es un drift realista** —42% anual con 30% de volatilidad es un Sharpe
aritmético de ≈1.4 sostenido durante veinte años— y no pretende serlo. El
fixture no imita un mercado: hace contestable una pregunta que el 4a no puede
contestar.

### El criterio de 4b, declarado antes de correrlo

Tres condiciones, en este orden:

1. **Sobre el fixture, no sobre el agente:** el `t` mediano del drift en la
   ventana de entrenamiento debe ser ≥ **3.0**. Si no lo es, el fixture no tiene
   lo que dice tener y el nivel no puede medir lo que pretende. El veredicto lo
   reporta como fallo de calibración, no de aprendizaje.
2. **Converge a estar invertido:** media entre caminos del tiempo invertido
   ≥ **0.80**. Con drift detectable y sin señal direccional, el óptimo es estar
   invertido.
3. **No rota:** media entre caminos de la dispersión de la acción ≤ **0.15**.

Sobre el tercero, que es el que más fácil se elige mal. Se usa la **desviación
estándar de la acción dentro del episodio** y no la rotación anualizada. Una
política fija en 1.0 da dispersión 0; una que alterna entre 0 y 1 la mitad del
tiempo da 0.5; una que oscila entre 0.8 y 1.0 da ≈0.1. El umbral 0.15 admite
oscilación y excluye rotación. La rotación anualizada mezcla el comportamiento
con el crecimiento del equity y con el rebalanceo al peso objetivo, así que
entra al reporte como evidencia pero no como criterio — es el mismo motivo por
el que se quitó `level_4_max_turnover_ratio` en el PR #7.

Las tres se reportan con su varianza de mercado y de entrenamiento separadas,
como exige el principio 5.

### Qué significa cada resultado

- **4b pasa:** el agente reconoce drift cuando es medible. Entonces su
  abstención en 4a es una respuesta correcta a una serie sin drift detectable, y
  no un defecto.
- **4b falla por no converger:** el agente no reconoce un drift que sí está en
  la muestra. Eso es un defecto real del agente o del entrenamiento, y hay que
  entenderlo antes de los datos reales.
- **4b falla por rotar:** el agente encuentra estructura donde el óptimo es
  quedarse quieto. Es el mismo síntoma que el 4a busca, en un fixture donde hay
  drift pero sigue sin haber dirección predecible.

**Se registra antes de correrlo** por el mismo motivo que la predicción del
nivel 3 en el ADR 0004: después de ver el resultado, cualquier umbral se puede
justificar.

## Decisión 3 — **El par es lo que carga la evidencia, no cada nivel por separado**

Un veredicto de 4b aislado **no distingue dos comportamientos muy distintos**:

- el agente **reconoce el drift** y por eso se invierte; o
- el agente **compra por defecto** y se habría invertido igual sin drift.

Con μ=0.42 los dos satisfacen los tres criterios del 4b. El nivel por sí solo no
puede separarlos, y reportarlo como si pudiera sería sobreinterpretar un PASS.

Lo que los separa es la **diferencia contra el 4a**, que es el mismo proceso sin
drift detectable: un agente que compra por defecto también está invertido en 4a y
su diferencia es cero. Hay un test que lo fija: un agente al 95% en ambos
fixtures **pasa** el 4b y **falla** el par.

### El contraste es pareado, y no por elegancia

`generate_gbm_sv` consume los mismos shocks para la misma semilla, así que 4a y
4b con la semilla `s` comparten la realización del ruido **y la del proceso de
varianza**: `mu` solo entra en el término determinista del drift. Verificado a
precisión de máquina: la diferencia entre los log-retornos de los dos fixtures
con la misma semilla es la constante `(μ_b − μ_a)/252 = 1.349e-3`, con desvío
1.4e-15, y la correlación entre los dos caminos es exactamente 1.

La varianza de mercado —que es la grande, σ≈0.49 en el nivel 4a— **se cancela
dentro de cada par**, y lo que queda es el efecto de `mu`. Un contraste no
pareado tendría que atravesar esa dispersión y podría no detectar un efecto
grande solo porque los caminos son ruidosos.

### El criterio del par

La diferencia pareada de tiempo invertido entre 4b y 4a es **positiva y se
distingue de cero** (t de dos colas al 95%, N=10 pares).

El umbral es **cero** y no un número elegido: la hipótesis nula es "el agente se
comporta igual con drift y sin drift", y cualquier umbral positivo la estaría
reemplazando por otra hipótesis.

Un detalle que los tests destaparon y vale documentar: si la diferencia entre
pares fuera **constante** —dispersión nula— el estadístico `t` no está definido,
pero eso **no** vuelve al efecto indistinguible de cero: lo vuelve determinista.
La primera versión del código devolvía "no distinguible" en ese caso, que es la
conclusión opuesta a la que los datos sostienen. Corregido con el mismo criterio
de tolerancia relativa que el resto del proyecto usa para dispersión nula.

## Decisión 4 — El orden entre 4a y 4b

4b se evalúa **después** de 4a y se saltea si 4a falla. Medir si el agente
reconoce drift real solo tiene sentido una vez establecido que no inventa señal
donde no la hay; al revés, un 4b que pasa podría estar pasando por sobreajuste.

Los dos llevan `level = 4` en el reporte y se distinguen por etiqueta: no son
dos niveles, son las dos hipótesis del mismo.

## Resultado (añadido después de correr, con el criterio ya registrado)

Los criterios de arriba se commitearon antes de ejecutar nada: `13b0ae1` para
4b, `2775631` para el par. El log de la corrida registra el primer hash.

| | Veredicto | Números |
|---|---|---|
| **4a** control negativo puro | ✅ PASS | exceso `t = +0.69` contra 2.262 → **no distinguible de cero** |
| **4b** drift detectable | ❌ FAIL | `t` del drift +5.77; tiempo invertido **0.63** (< 0.80); dispersión de la acción **0.383** (> 0.15) |
| **par 4a↔4b** | ⚠️ PARTIAL | diferencia pareada de tiempo invertido **+0.3065**, `t = +14.58` → reconoce; pero 4b en rojo → no explota |

### Lo que dice el par, y que ninguno de los dos niveles dice solo

**El agente sí responde al drift.** Pasa de 0.33 a 0.63 de tiempo invertido entre
los dos fixtures, una diferencia pareada de +0.31 con `t = 14.6`. No compra por
defecto: si lo hiciera, estaría igual de invertido en 4a y la diferencia sería
cero. Esa era la hipótesis que 4b por sí solo no podía descartar, y queda
descartada.

**Y aun así no aprovecha el drift.** 0.63 no es 0.80, y la dispersión de la
acción es 0.383 contra un umbral de 0.15: el agente sigue entrando y saliendo
cuando el óptimo es quedarse quieto. El exceso sobre estar siempre invertido en
4b es **−0.65** (`t = −7.33`): deja mucha plata sobre la mesa.

La lectura conjunta es **reconocimiento parcial**: detecta la dirección con
altísima confianza estadística y no la explota. Ese es un diagnóstico distinto
tanto de "no reconoce nada" como de "reconoce y converge", y solo el par lo
produce.

**4b no se relaja por esto.** Falla, y falla por los dos criterios. Que el par
explique *qué* clase de fallo es no lo convierte en un pase.

### Por qué el par es PARTIAL y no PASS

El veredicto del par estuvo en PASS en la primera versión de este ADR, porque su
criterio registrado —la diferencia pareada— se cumple. Eso era un agregado en
verde sobre un componente en rojo, que es exactamente lo que este protocolo
existe para evitar: quien leyera la línea del par sin leer la de 4b se llevaría
"el nivel 4 pasó".

Corregido: `Verdict.PARTIAL`. El par evalúa ahora las dos cosas por separado y
las reporta juntas.

- `responde` = la diferencia pareada es positiva y se distingue de cero. **Se
  cumple** (+0.3065, `t = +14.58`).
- `explota` = los umbrales absolutos de 4b (tiempo invertido ≥ 0.80 y dispersión
  ≤ 0.15). **No se cumple** (0.63 y 0.383).

`responde and explota` → PASS. `responde` sin `explota` → **PARTIAL**, con las
dos cifras y sus dos umbrales en el hallazgo. Nada de esto cambia el criterio de
4b ni el de 4a: PARTIAL es una etiqueta nueva para un estado que antes se
colapsaba a PASS, no un umbral relajado.

## Diagnóstico: por qué 4b no converge

Tres hipótesis distinguibles con los datos ya producidos. Script en
`scratchpad`, no versionado: es diagnóstico, no infraestructura.

### (b) ¿el reward es casi indiferente entre 0.8 y 1.0? — **NO**

Pesos constantes sobre la ventana de validación de 4b (camino 701):

| peso | retorno total | reward/barra |
|---|---|---|
| 0.60 | +1.382 | +0.06248 |
| 0.63 | +1.473 | +0.06560 |
| 0.80 | +2.023 | +0.08331 |
| 0.90 | +2.373 | +0.09372 |
| 1.00 | +2.738 | +0.10413 |

Monótono y con pendiente estable: pasar de 0.8 a 1.0 vale **+0.0341 de reward
por barra**, con `t = 6.58` sobre las 4799 barras de entrenamiento y `t = 23.27`
sobre los 60 000 timesteps que el agente efectivamente ve. La señal de gradiente
está ahí y es grande. Hipótesis **descartada**.

### (a) ¿la entropía mantiene la política estocástica? — **no por `ent_coef`**

`ent_coef = 0.0`: no hay bonus de entropía, así que la hipótesis tal como estaba
formulada no aplica. Pero el mecanismo que describe sí está presente en otra
forma: el `log_std` **aprendido** de la gaussiana de SB3 arranca en 0.0
(σ = 1.00) y después de 60 000 timesteps está en −0.095, o sea **σ = 0.909**. Se
encogió un 9%. Con σ ≈ 0.9 sobre un espacio de acción de ancho 1, la acción
muestreada durante el entrenamiento es casi independiente de la media, así que la
ventaja que PPO atribuye a la media llega enterrada en ruido y el gradiente de la
media queda mal identificado.

Aclaración que evita una lectura errónea: **la evaluación es determinista** (usa
la media, no muestrea), así que la dispersión medida de 0.383 en 4b **no** es ese
ruido de muestreo. Es la media misma moviéndose con el estado. Y su forma
importa: los percentiles de la acción en validación son
`[0.00, 0.33, 0.90, 1.00, 1.00]`. El agente no está "invertido al 63% siempre";
está **plano en algunas barras y all-in en otras**. Eso es una política sin
converger, no una política convergida a una asignación parcial.

### (c) ¿alcanzan 4800 barras / 60 000 timesteps? — **es la causa operativa**

Mismo fixture, mismo camino, misma semilla, solo más presupuesto:

| timesteps | tiempo invertido | dispersión | σ de la política |
|---|---|---|---|
| 60 000 | 0.673 | 0.392 | 0.909 |
| 240 000 | **0.777** | 0.366 | 0.651 |

Las tres métricas se mueven monótonamente hacia donde el criterio las quiere, y
el tiempo invertido llega a 0.777, al borde del umbral de 0.80. **El presupuesto
de optimización es la causa operativa**, y el `log_std` que se encoge despacio es
el mecanismo por el que se agota.

### Veredicto del diagnóstico

**Es (c), con el mecanismo de (a) en su forma de `log_std` aprendido y no de
`ent_coef`.** Es decir: **un hiperparámetro y un presupuesto, no un hallazgo
sobre la capacidad de aprender.** El drift está en la muestra (`t` = 5.77), el
reward lo premia con `t` = 23 y el agente se mueve en la dirección correcta
cuando se le da cuádruple presupuesto.

Límite honesto de este diagnóstico: (a) y (c) son **una sola corrida** (camino
701, semilla 11). No es una distribución sobre 10 caminos × 10 semillas como los
veredictos. Alcanza para descartar (b) —que es aritmética sobre el fixture, no
una corrida— y para señalar la dirección de (a) y (c); no alcanza para poner un
número sobre cuántos timesteps harían falta.

### Consecuencia para la Etapa 5, que es por lo que hay que saberlo ahora

Una política que a 60 000 timesteps todavía tiene σ = 0.91 y una media que oscila
entre 0 y 1 **no está en su óptimo**. Comparar el Agente A contra el Agente B en
ese estado mide la diferencia entre dos políticas a medio entrenar, y la varianza
de entrenamiento —que en 4a ya medimos en 0.197 contra 0.490 de mercado— se come
cualquier efecto de las noticias antes de que se pueda ver.

Lo que hay que hacer antes de la Etapa 5, y queda anotado como deuda:

1. Fijar el presupuesto de timesteps con una curva de convergencia, no por
   defecto.
2. Revisar `log_std_init` y considerar un `log_std` programado, para que la
   exploración se apague.
3. Verificar que el presupuesto elegido deja el σ de la política estable, y
   reportarlo junto a la comparación A vs B.

**Nada de esto se aplica retroactivamente a 4b.** Su criterio queda como está y
su veredicto sigue siendo FAIL. Cambiar los hiperparámetros y volver a correr
para que pase sería exactamente el ajuste que el protocolo prohíbe. Lo que
cambia, cuando se cambie, es el presupuesto de todo el estudio, y entonces se
re-corre el protocolo completo, no solo el nivel que falló.

### Consecuencia para los datos reales

Un agente que reconoce la dirección pero no se compromete con ella va a quedar
sistemáticamente por debajo de buy-and-hold en un activo con drift fuerte.
BTCUSDT desde 2017 es exactamente eso. No es una predicción registrada —se
deriva de lo medido, no se arriesga antes— pero conviene tenerla escrita antes
de mirar el resultado, para no descubrirla después como si fuera un hallazgo.

## Lo que este ADR no cambia

- El criterio del exceso de 4a, que es el que fallaba y ahora pasa.
- Los niveles 0 a 3.
- La predicción del nivel 3 registrada en el ADR 0004.
