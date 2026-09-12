# ADR 0004 — Varianza de mercado contra varianza de entrenamiento

- **Estado:** aceptado
- **Fecha:** 2026-09-07
- **Contexto de etapa:** Etapa 3. El protocolo de validación corrió entero y se
  detuvo en el nivel 4 (PR #7). Todavía **no se miraron datos reales**.

## Contexto

El protocolo de la Etapa 3 reportó el nivel 4 así: el agente excede a estar
siempre invertido en **+0.164** de crecimiento logarítmico, con las diez
semillas de acuerdo entre sí (p25 +0.140, 9 de 10 positivas), sobre un camino
donde el baseline rindió −0.013. El criterio declaraba significativo cualquier
exceso por encima de 0.05, así que el nivel falló.

Al investigar por qué el agente "ganaba" en una serie sin señal direccional
apareció algo más grande que ese nivel: **el desvío del mismo baseline entre 20
caminos de Heston independientes es 0.781**, con 8 de 20 negativos y un rango de
−0.475 a +1.896. El "hallazgo" de +0.164 era cinco veces más chico que la
dispersión del sorteo.

Diez semillas de acuerdo entre sí no lo revelaron, y no podían: **todas
entrenaron sobre el mismo camino**. Lo que medían era cuánto cambia el resultado
al reentrenar sobre la misma serie, que es una pregunta distinta de cuánto
cambia al haber tocado otro mercado.

## Decisión 1 — Dos varianzas, reportadas por separado

`CLAUDE.md` decía "distribución sobre >= 10 semillas" sin distinguir qué fuente
de variación mide esa distribución. Ahora distingue:

- **Varianza de entrenamiento** (`within_path_std`): cuánto cambia el resultado
  al reentrenar sobre la misma serie. Es lo que miden M semillas sobre un camino.
- **Varianza de mercado** (`between_path_std`): cuánto cambia al muestrear otro
  camino del mismo proceso. Es lo que miden N caminos.

`eval.distribution.decompose_variance` las separa. Cada camino aporta **una**
observación —su mediana sobre las M semillas— porque las semillas de un mismo
camino no son independientes entre sí: comparten la serie. Un contraste contra
cero usa por lo tanto el error estándar entre caminos, con N−1 grados de
libertad, no N×M.

La tabla de valores críticos de `t` está escrita a mano y no aproximada por la
normal: con N=10 la diferencia entre 2.262 y 1.960 decide si un exceso se declara
significativo.

## Decisión 2 — La asimetría con datos reales se declara, no se resuelve

En un fixture sintético los caminos se muestrean: cambiar la semilla del proceso
da otra realización del mismo mercado. **En datos reales no.** La historia es un
solo camino y N=1 por construcción; no hay forma de estimar la varianza de
mercado porque no hay repeticiones del experimento.

Esa asimetría es la razón de fondo por la que el walk-forward fuera de muestra es
la única defensa que queda ahí: con N=1 no se puede *medir* el error de muestreo,
solo acotarlo con más ventanas fuera de muestra —y cada ventana adicional es una
observación más, no una repetición del mercado—. Cualquier reporte sobre datos
reales lo declara como limitación.

Consecuencia práctica que conviene tener escrita: un resultado positivo sobre
datos reales con una sola ventana de test **no es distinguible** de haber tenido
suerte con el período, y ninguna cantidad de semillas de entrenamiento cambia eso.

## Decisión 3 — El criterio del nivel 4 es un contraste, no un umbral

El criterio anterior (`level_4_max_median_excess = 0.05`) comparaba la mediana
del exceso contra un número fijo. **No era demasiado laxo ni demasiado estricto:
estaba mal planteado**, porque medía una cantidad contra una escala que no le
correspondía.

El nuevo criterio pregunta lo único que se puede preguntar sobre una serie sin
señal direccional: *¿el exceso se distingue de cero cuando se lo mide contra el
ruido del sorteo?* Formalmente, `t` de dos colas al 95% sobre las N medianas por
camino, con N ≥ 10.

El umbral de tiempo invertido (≥ 0.80) se conserva sin cambios: ese sí es un
enunciado sobre el comportamiento del agente y no sobre significancia.

## Decisión 4 — El `t` del drift va al lado del veredicto

Sobre Heston `medium` (μ=8% anual, 30% de volatilidad) el `t` del drift medio
sobre las 4800 barras de entrenamiento, **medido sobre los 10 caminos**, tiene
mediana **0.84** y rango −0.91 a +2.21: solo uno de los diez supera 1.96. El
drift no es detectable en la muestra. Pedirle al agente que "converja a estar
invertido" le pide aprender algo que la muestra no contiene.

**Corrección a una cifra publicada.** El PR anterior (y el ADR 0003) daban ≈1.8
para ese estadístico. Era una estimación mía, no una medición, y estaba mal por
dos motivos: usaba el drift **aritmético** `μ·dt` cuando el generador produce
log-retornos con drift `(μ − v/2)·dt`, y estaba redondeada hacia arriba. Con la
fórmula correcta el valor poblacional es ≈0.51, y la mediana muestral sobre 10
caminos da 0.84. La conclusión cualitativa no cambia —el drift sigue sin ser
detectable— pero el número que se publicó era el equivocado y ahora está medido
en vez de estimado.

`drift_t_statistic` lo calcula y `evaluate_level_4` lo pone **en el hallazgo**,
no en una nota al pie. Sin ese número al lado, un fallo del nivel se lee como un
fallo del agente cuando puede ser un fallo del enunciado.

Esto **no** relaja el criterio: si el exceso sobrevive al contraste entre
caminos, el nivel falla igual. Lo que cambia es que el lector sabe si el otro
brazo del criterio —converger a estar invertido— era alcanzable en esa muestra.

## Decisión 5 — El nivel 3 queda registrado como PREDICCIÓN, fechada

**Fecha: 2026-09-07. Ningún dato real ha sido cargado ni mirado en este
repositorio a esta fecha.** El primer dataset real entra en un PR posterior.

Lo medido en el nivel 3, sobre el tramo de validación entero posterior al cambio
de régimen:

| | Retorno | `capture` |
|---|---|---|
| Techo informado (regla nueva) | +829% | 1.000 |
| Estar siempre invertido | −1.3% | 0.000 |
| Memorizador (regla vieja) | −89.5% | −0.998 |
| Agente MLP | −80.4% | −0.721 |
| Agente LSTM | −79.7% | −0.706 |

El agente **memoriza el régimen de entrenamiento**: queda pegado al memorizador
puro y aplica con convicción la regla de reversión sobre datos que ya son de
momentum. La política recurrente no cambia el resultado (−0.706 contra −0.721,
adentro de la dispersión entre semillas): **la LSTM no infiere el régimen**.

### La predicción

Si ese comportamiento es una propiedad del agente y no un artefacto del fixture,
entonces sobre datos reales con walk-forward **debe observarse degradación del
desempeño fuera de muestra en las ventanas que siguen a un cambio de régimen**, y
esa degradación debe ser mayor que la de ventanas comparables sin cambio de
régimen.

Predicciones subsidiarias, todas falsables:

1. **La ventana de test que cubre un giro de mercado pronunciado rinde peor que
   la mediana de las ventanas**, y la brecha no se explica por la dirección del
   mercado en esa ventana —hay que compararla contra los baselines de la
   **misma** ventana, no contra el promedio del período—.
2. **La política recurrente no cerrará esa brecha materialmente.** En el fixture
   no lo hizo con un cambio de régimen limpio, instantáneo y de signo conocido;
   en datos reales, donde los cambios son graduales y ambiguos, hay menos motivo
   para esperar que lo haga.
3. **El agente rotará más y estará invertido más tiempo del que corresponde
   justo después del giro**, que es la firma de estar aplicando la regla anterior.

Qué la refutaría: que el desempeño fuera de muestra no se degrade tras los
cambios de régimen, o que se degrade por igual en ventanas con y sin cambio. En
ese caso la memorización medida en el fixture sería un artefacto de un cambio de
régimen artificialmente abrupto y no una propiedad del agente.

**Por qué se registra antes.** Después de ver los datos reales, cualquier
degradación observada se puede explicar con la historia que uno quiera, y "el
agente memoriza" es una explicación demasiado cómoda para dejarla sin comprometer
de antemano. Esta predicción queda fechada, en el repositorio, y se contrasta
tal como está escrita.

## La predicción QUEDA FIRME — confirmada sobre 10 caminos el 2026-09-12

**A esta fecha sigue sin haberse cargado ni mirado ningún dato real en este
repositorio.** La confirmación es sobre fixtures sintéticos, que es donde
correspondía verificarla.

El nivel 3 se re-corrió con **10 caminos independientes × 10 semillas**, MLP y
LSTM (Tarea E). El resultado:

| Brazo | `capture` medio entre caminos | σ mercado | σ entrenamiento | Memoriza en |
|---|---|---|---|---|
| MLP | **−1.108** | 0.586 | 0.097 | **10/10 caminos** |
| LSTM | **−1.108** | 0.521 | 0.096 | **10/10 caminos** |

Memorizador de referencia: **−1.436**. El umbral de clasificación es la mitad del
camino hacia esa referencia (−0.718), y los veinte brazos-camino quedan del lado
del memorizador.

**La premisa de la predicción no era un artefacto del camino que se había medido
antes.** Se sostiene en los diez, con varianza de mercado 5 a 6 veces la de
entrenamiento —otra confirmación de que el reporte de un solo camino describía la
fuente de variación equivocada—.

### La predicción subsidiaria 2 también queda confirmada, en fixture

"La política recurrente no cerrará esa brecha materialmente." Contraste **pareado**
entre arquitecturas sobre los mismos diez caminos:

```
capture:  LSTM −1.1084  contra  MLP −1.1084
          diferencia pareada −0.0001 (sigma 0.1297, N=10)  t = −0.00  ->  NO distinguible de cero
```

No es "las dos medias se parecen": es cero, con un contraste que cancela la
varianza de mercado dentro de cada par. La memoria no aporta nada sobre este
fixture. Lo único distinguible entre arquitecturas es que la LSTM **rota un poco
más** (dispersión de la acción +0.0176, `t` = 6.84), que va en contra de la
memoria y no a favor.

### Qué sigue en juego

Las predicciones 1 y 3 son sobre **datos reales** y siguen sin contrastar:

1. La ventana de test que cubre un giro pronunciado rinde peor que la mediana de
   las ventanas, comparada contra los baselines de **esa** ventana.
3. El agente estará invertido de más y rotará más justo después del giro.

Se contrastan tal como están escritas, en la Tarea C.

## Lo que queda pendiente

- **El nivel 3 ya está re-reportado con N=10 caminos** (ver arriba). **Los
  niveles 1 y 2 quedan como deuda documentada y no se van a re-correr.** Su
  conclusión es robusta a la ruta: el nivel 1 mide una degradación *monótona* a
  lo largo de un barrido de SNR —0.928 / 0.878 / 0.741 / 0.300 para R² de
  25/9/4/1%—, y un ordenamiento de cuatro puntos con esa separación no se invierte
  por haber tocado otro camino; el nivel 2 mide un cambio de comportamiento del
  19.3% en la rotación entre dos fixtures que solo difieren en los costos, que es
  un contraste dentro del mismo proceso. Re-correrlos costaría del orden de 500
  corridas para confirmar conclusiones que no están en duda, y ese cómputo rinde
  más en la Etapa 6. El nivel 0 es la excepción real: es determinista, su varianza
  de mercado es exactamente cero y N=1 ahí es completo.
- El nivel 4 sí se re-corrió con el criterio nuevo; el resultado está en
  `results/multipath_level_4.json`.
