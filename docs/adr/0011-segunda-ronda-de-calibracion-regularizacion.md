# ADR 0011 — Segunda ronda de calibración: regularización del Agente A sobre 4b

- **Estado:** borrador. **Kevin elige la escalera (sección 4) antes de que se selle.**
  Se sella al mergearse, igual que el ADR 0010: el script de esta ronda no corre si
  el SHA256 de este archivo no coincide con el sellado.
- **Fecha:** 2026-10-23
- **Contexto de etapa:** prerrequisito de la Etapa 5. Es la segunda ronda que el
  ADR 0010 §4 manda decidir en un ADR nuevo cuando ningún punto de la grilla pasa.
- **A la fecha de este ADR no se corrió ningún peldaño.** Lo medido que lo informa
  es la calibración del ADR 0010 (G1 a G4) y el diagnóstico de #57, ambos abajo.

## Contexto

### Lo que midió el ADR 0010

Los cuatro puntos fallan 4b, y fallan por el mismo criterio (resumen en
`results/calibracion/summary.json`, 5 caminos × 10 semillas por punto):

| Punto | Timesteps | Tiempo invertido | Dispersión de la acción | σ de la política | Rotación anual |
|---|---|---|---|---|---|
| G1 | 120 000 | 0,795 | 0,341 | 0,81 | 69 |
| G2 | 240 000 | 0,823 | 0,329 | 0,63 | 67 |
| G3 | 480 000 | 0,814 | 0,346 | 0,30 | 70 |
| G4 | 960 000 | 0,804 | 0,350 | 0,05 | 72 |

Umbrales: tiempo invertido ≥ 0,80 y dispersión ≤ 0,15. **La σ bajó de 0,81 a 0,05 y
la dispersión no se movió.** La evaluación es determinista (`deterministic=True`),
así que lo que rota es la media de la política, no la exploración. El ADR 0005
atribuía el fallo a la σ sin colapsar; esa hipótesis queda refutada, y con ella
`log_std_init` y más timesteps como palancas.

### Lo que midió el diagnóstico (#57)

Diez políticas a 120 000 timesteps sobre `DIAGNOSIS_PATH_SEEDS` (839 a 863),
disjuntos de los de calibración y del protocolo. Fijando un grupo de columnas a su
media en lazo cerrado:

| Grupo fijado | Dispersión mediana | Lectura |
|---|---|---|
| ninguno | 0,345 | — |
| posición (`position_weight` + `cash_weight`) | cambio medio 0,0125 | no es un bucle de la posición |
| resultado de la orden | cambio medio 0,004 | no influye |
| retornos rezagados (10 columnas) | **0,19** | el motor principal |
| técnicos (8 columnas) | 0,31 | aporta |

Ningún grupo solo baja de 0,15. **La política se ajusta al ruido de las
características de mercado.**

### Por qué pasa, y por qué no lo arregla el presupuesto

En 4b no hay señal direccional: toda columna de mercado es ruido respecto del
retorno siguiente. Y 4b no tiene costos (`ZERO_COSTS`): rotar es gratis en el
reward salvo por el drift que se pierde estando fuera. Ese drift es chico frente
al ruido de una barra: `(μ − θ/2)/252 = 0,00149` contra `√(θ/252) = 0,0189`, una
relación de **0,08 por barra**. El gradiente que empuja a "estar invertido siempre"
existe, pero cada lote de 2 048 pasos también trae correlaciones espurias entre las
18 columnas de mercado y el retorno, y con 10 épocas sobre el mismo lote la red las
aprende. Más pasos no ayudan: cada lote nuevo trae otras correlaciones espurias que
mantienen la media moviéndose. Es sobreajuste al ruido del lote, y lo que lo ataca
es **regularización**: menos entradas que ajustar, menos capacidad, pasos de
optimización más cortos o un objetivo con menos varianza.

## 1. Qué se calibra y qué no

Se calibra **una configuración del Agente A** (`PPOConfig` + `ObservationSpec`) a
**120 000 timesteps fijos**, el presupuesto de G1. No se cruzan palancas con la
grilla de timesteps: G1 a G4 muestran que el presupuesto no mueve la dispersión, y
cruzarlas multiplicaría los intentos.

No cambian:

- **Los criterios de 4b, 4a y el par** (ADR 0005), ni ningún umbral del protocolo.
  En particular, **el umbral de 0,15 no se toca**.
- El fixture 4b, el reward (`NetReturnReward`, `reward_scale` 100) y el entorno.
- Los veredictos históricos: 4b a 60 000 y G1 a G4 siguen en FAIL.
- Los Agentes A congelados de la sombra (ADR 0009). No se reemplazan dentro de la
  ventana de 30 días que cierra hacia el 2026-11-18.

## 2. Herencia: lo que cambie en A, B lo hereda igual

La comparación A contra B de la Etapa 5 solo es atribuible a las noticias si los dos
agentes son idénticos salvo por las columnas de noticias. **La configuración que salga
de esta ronda es la del estudio**: el Agente A, el Agente B, el placebo y cualquier
re-corrida usan el mismo `PPOConfig`, el mismo `ObservationSpec` (incluido el
`lookback`) y el mismo presupuesto. B no se calibra aparte. Si B rinde peor con la red
más chica porque tiene más entradas, eso es parte de lo que se mide, no algo que se
corrija después.

## 3. Las palancas, con su tradeoff

Cada una se evalúa por cuatro cosas: qué ataca, qué evidencia hay de que sirva, qué le
cuesta al estudio sobre datos reales y qué riesgo corre en la confirmación (que
re-corre el protocolo entero: un agente que deja de rotar en 4b puede dejar de ver la
señal en los niveles 1 y 2).

### L1 — `lookback` de 10 a 1

- **Ataca:** el grupo que el diagnóstico señala como motor. Quedan 9 columnas de
  mercado menos que ajustar. Se queda `log_return_0`, que es donde viven las señales
  de los niveles 0 a 2 (`src/agents/CLAUDE.md`: "el óptimo depende solo de `r_t`").
- **Evidencia:** la más directa que hay. Pero el diagnóstico fijó **los diez**
  retornos y la dispersión bajó a 0,19, no a 0,15. **Predicción: L1 sola no pasa.**
- **Costo en datos reales:** A pierde el camino de los últimos diez días como
  entrada. RSI, MACD y Bollinger ya resumen ese camino, y C3 no encontró ventaja con
  los diez retornos. Bajo.
- **Riesgo en la confirmación:** nivel 3 (cambio de régimen). Es `MEASURED`, no
  bloquea, pero su lectura cambia y se reporta.
- **Cómputo:** baja un poco (observación de 24 a 15 columnas).

### L2 — Optimización más corta: `learning_rate` 3e-4 → 1e-4, `n_epochs` 10 → 5

- **Ataca:** el sobreajuste al lote. Diez épocas sobre 2 048 pasos con relación
  señal/ruido de 0,08 es la receta para aprender el ruido del lote. Bajar las épocas
  es la palanca más directa; bajar el lr achica cuánto se mueve la media con cada
  correlación espuria. **`n_epochs` no estaba en la lista original y se agrega aquí
  porque es el mecanismo, no el lr.**
- **`clip_range` se queda en 0,2.** Clip y lr limitan lo mismo (cuánto cambia la
  política por actualización); bajar los dos contaría el mismo efecto dos veces y
  sumaría un eje sin información nueva.
- **Evidencia:** indirecta. Ningún punto del ADR 0010 la probó.
- **Costo en datos reales:** ninguno sobre el problema; es cómo se optimiza.
- **Riesgo:** aprender menos en el mismo presupuesto. A 120 000 el tiempo invertido
  ya está en 0,795, en el borde de 0,80, y con aprendizaje más lento puede caer por
  debajo. Entonces 4b fallaría por "no converge" en vez de "rota", que es un fallo
  distinto y se reporta como tal.

### L3 — Red de 64×64 a 32×32

- **Ataca:** la capacidad de representar una función ruidosa de las entradas.
- **Evidencia:** débil. Con 15 entradas una red de 32×32 sigue sobrando para ajustar
  ruido; la capacidad no es lo que falta para sobreajustar sino lo que sobra.
- **Costo en datos reales:** B tiene 6 columnas más que A (D2 del plan de la
  Etapa 4). Una red más chica puede limitar a B más que a A. Por la sección 2 eso se
  mide, no se corrige, pero es un sesgo en contra de B que hay que declarar.
- **Riesgo:** bajo en los niveles 0 a 2 (el óptimo es una función simple de `r_t`).

### L4 — `gamma` de 0,99 a 0,9

- **Ataca:** la varianza del objetivo. Con 0,99 (y `gae_lambda` 0,95) la ventaja de
  cada paso suma ~17 barras de reward ruidoso; con 0,9, ~7. En 4b sin costos el óptimo
  es miope, así que gamma no cambia el óptimo ahí y solo baja el ruido del gradiente.
  **Tampoco estaba en la lista original.**
- **Costo en datos reales:** cambia el problema, no solo la optimización. Con costos
  la decisión de hoy acopla con las siguientes (la banda de histéresis del nivel 2
  existe por eso). Un horizonte de 10 barras puede volver al agente miope frente a
  los costos.
- **Riesgo en la confirmación:** alto en el nivel 2, que es justamente el que prueba
  que el agente internaliza los costos. Por eso va al final.

### L5 — Penalización del cambio de acción en el reward

- **Ataca:** el síntoma directamente: un término `−c·|a_t − a_{t−1}|`.
- **Por qué recomiendo dejarla fuera.** Es reward shaping no económico. En 4b la
  penalización castiga exactamente la cantidad que el criterio 3 mide: pasar 4b con
  ella sería casi circular, el agente aprendería a no mover la acción porque se le
  cobra, no porque reconozca que el óptimo es quedarse quieto. En datos reales los
  costos ya están en el reward (principio 3); sumar una penalización los cuenta dos
  veces con un precio inventado, y `c` sería un hiperparámetro más que contaría
  como intento. Si se incluye, va como último peldaño y el reporte lo dice al lado
  del veredicto.

### Descartadas sin peldaño

- **Más timesteps y `log_std_init`:** refutadas por G1 a G4 (σ a 0,05, dispersión
  igual).
- **Quitar los técnicos de A:** A está definido como precio, volumen y técnicos
  (`CLAUDE.md`). Sacarlos cambia qué es el Agente A, no cómo se entrena.
- **Suavizar la acción o una banda muerta en el entorno:** esconde la rotación en vez
  de arreglarla, con el mismo problema de circularidad que L5, y toca `envs/`, que es
  un wrapper delgado.
- **Weight decay:** cumple el papel de L3 y no está en `PPOConfig`; agregarlo es
  interfaz nueva para una palanca que se solapa con otra.

## 4. La escalera (Kevin elige)

**Los peldaños son acumulativos y se corren en orden; se para en el primero que pasa.**
Van de lo que menos cambia el problema a lo que más. Así el resultado es la
configuración **menos invasiva** que pasa, y no hay selección entre alternativas: el
orden es la elección, y queda escrita antes de correr.

Acumulativos y no de a una, porque el diagnóstico ya mostró que ninguna causa sola
alcanza. Probar cada palanca aislada mediría efectos marginales que no deciden nada y
multiplicaría los intentos.

**Recomendada — tres peldaños, sin L5:**

| Peldaño | Configuración | Cambia respecto de G1 |
|---|---|---|
| R1 | L1 + L2 | `lookback` 1, `learning_rate` 1e-4, `n_epochs` 5 |
| R2 | R1 + L3 | red 32×32 |
| R3 | R2 + L4 | `gamma` 0,9 |

L1 no va sola porque el diagnóstico ya predice que no alcanza (0,19 con los diez
retornos fijados). Correrla sola cuesta una hora y la respuesta esperada ya se sabe.

**Alternativas:**

- **B — cuatro peldaños, L1 sola primero.** Aísla el efecto del lookback y contrasta
  la predicción del diagnóstico. Cuesta ~1 h más en el peor caso.
- **C — la recomendada más L5 como R4**, con `c` fijado aquí antes de correr. Da una
  salida más si R1 a R3 fallan, a cambio de un 4b que pasaría por construcción.

## 5. Caminos, semillas y regla de decisión

- **Caminos de calibración: `CALIBRATION_PATH_SEEDS` (809, 811, 821, 823, 827) × las
  10 `STUDY_SEEDS`**, 50 entrenamientos por peldaño. Se reusan a propósito: G1 corrió
  sobre los mismos caminos y semillas con el mismo presupuesto, así que cada peldaño
  se compara **pareado** contra G1, con el mismo ruido. Las palancas se eligieron por
  el diagnóstico, que corrió sobre otros caminos (839 a 863); de G1 a G4 solo se usó
  el agregado ("el presupuesto no mueve la dispersión"), no nada por camino.
- **Criterio:** los tres de 4b del ADR 0005, sin cambios, con
  `level_4_min_paths = 5` como override local, igual que en el ADR 0010.
- **Regla:**
  1. Se corre R1. Si pasa, `C*` = R1. Si no, R2, y así.
  2. **Confirmación:** con `C*` a 120 000 timesteps se re-corre el **protocolo
     completo** sobre `PATH_SEEDS` y `STUDY_SEEDS` (niveles 0 a 3, 4a, 4b y el par).
     Más regularización puede apagar la señal en los niveles 1 y 2.
  3. Protocolo en verde → **`C*` y 120 000 timesteps quedan fijos para todo el
     estudio** (sección 2), con el traslado a datos reales del ADR 0010 §7.
  4. 4b falla en la confirmación habiendo pasado en la calibración → se prueba el
     peldaño siguiente **una sola vez**, con la misma confirmación.
  5. Falla otro nivel → no se fija configuración. Hallazgo, ADR nuevo.
  6. Ningún peldaño pasa → no se fija configuración y no se agregan peldaños sobre la
     marcha. La conclusión declarada sería que PPO con esta observación no aprende a
     quedarse quieto sobre 4b, y la Etapa 5 tendría que decidir en un ADR si sigue con
     un Agente A que no pasa 4b (declarándolo en cada reporte) o cambia de algoritmo.

## 6. Qué se reporta

Para cada peldaño corrido, lo mismo que el ADR 0010 §5 (las tres cantidades de 4b
con las dos varianzas separadas, σ de la política, exceso sobre estar invertido,
tiempo de pared) y además:

- **la diferencia pareada contra G1** de la dispersión y del tiempo invertido, por
  camino y semilla, con su `t` entre caminos;
- **el diagnóstico de #57 re-corrido sobre la política de `C*`**, para ver qué grupo
  queda moviendo la acción.

Al repositorio van el resumen y el manifest; los JSON completos, al Release.

## 7. Cómputo

En el PC de Kevin, con la medida de la ronda anterior: ~4 min por entrenamiento a
120 000 timesteps con 3 procesos en paralelo (`docs/calibracion-presupuesto.md`).

| | Entrenamientos | Pared con 3 procesos |
|---|---|---|
| Un peldaño | 50 | ~1 h 10 min |
| Escalera recomendada, peor caso (3 peldaños) | 150 | ~3 h 30 min |
| Alternativa B, peor caso (4 peldaños) | 200 | ~4 h 40 min |
| Confirmación a 120 000 | 480 | ~11 h, más el LSTM, que es más lento |

La confirmación hay que hacerla reanudable antes de correrla: hoy no lo es, y 11 h
seguidas en un PC de escritorio es pedir una interrupción.

## 8. Lo que este ADR no cambia

- El ADR 0010, que sigue sellado y con su resultado (ningún punto pasa).
- El ADR 0007 (pre-registro de la sombra) ni los Agentes A congelados del ADR 0009.
- Ningún criterio ni umbral del protocolo.
