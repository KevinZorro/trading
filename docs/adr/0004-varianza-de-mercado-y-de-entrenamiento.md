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
sobre las 4800 barras de entrenamiento es **≈1.8**: por debajo de cualquier
umbral de significancia. Pedirle al agente que "converja a estar invertido" le
pide aprender algo que la muestra no contiene.

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

## Lo que queda pendiente

- **Los niveles 1, 2 y 3 siguen reportados con N=1 camino.** El cambio de regla
  los alcanza y sus resultados actuales describen la varianza de entrenamiento,
  no la de mercado. Re-reportarlos con N caminos cuesta del orden de 400 corridas
  para el barrido de SNR solo; queda como trabajo explícito y no como algo
  resuelto. El nivel 0 es la excepción: es determinista, su varianza de mercado
  es exactamente cero y N=1 ahí es completo.
- El nivel 4 sí se re-corrió con el criterio nuevo; el resultado está en
  `results/multipath_level_4.json`.
