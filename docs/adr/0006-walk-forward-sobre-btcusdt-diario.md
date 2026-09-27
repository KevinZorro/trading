# ADR 0006 — Walk-forward del Agente A sobre BTCUSDT diario (Tarea C)

- **Estado:** propuesto, en revisión. Se sella al aprobarse (ver *Sellado*).
- **Fecha:** 2026-09-27
- **Contexto de etapa:** Etapa 3, Tarea C. Dataset: `datasets/binance/BTCUSDT-1d.csv`,
  SHA256 `0940f089d796da4d5bda34b30463e0c3b19b1e6b8c322f8fcd3517f2aee49cc8`
  (PR #14).
- **A la fecha de este ADR no se entrenó ni se evaluó ningún agente sobre datos
  reales.** Lo único mirado son los precios del mercado, que son historia pública:
  la validación del dataset y la clasificación de régimen de la sección 2, que no
  usa ningún resultado del agente.

Este ADR fija, **antes de correr**, qué se mide, cómo se agrega y qué cuenta como
resultado. La corrida es la Tarea C3 y toca el test una sola vez.

---

## 0. Qué puede concluir esta corrida, y qué no

**La política está subentrenada.** Se usa la misma configuración de PPO que en el
protocolo de fixtures (opción i), con 60 000 timesteps por fold. El diagnóstico
del 4b (ADR 0005) mostró que a ese presupuesto la política no converge: σ = 0.91
y una media que oscila entre 0 y 1. El presupuesto se calibra sobre el fixture
4b antes de la Etapa 5, nunca sobre estos datos.

**Por lo tanto el resultado valida el pipeline, no concluye sobre el mercado.**
"El agente no supera a buy-and-hold" es un resultado válido y esperado. No
autoriza a decir "PPO no funciona en BTC", y un resultado a favor tampoco
autoriza lo contrario. Nada de lo que salga de C3 se usa para elegir
hiperparámetros.

## 1. Geometría

| | Valor |
|---|---|
| Serie | 3.328 barras diarias. `timestamp` = cierre, de 2017-08-18 a 2026-09-27 |
| Ventana | train 800 · validación 200 · test 200 barras, paso 200, rodante (`anchored=False`) |
| Folds | **11**, alineados al inicio de la serie (barra 0) |
| Tests | Barras 1000 a 3199, de 2020-05-14 a 2026-05-22: 2.200 barras contiguas, sin solaparse (`overlapping_test_bars = 0`) |
| Cola sin usar | 128 barras, de 2026-05-23 a 2026-09-27. Se declara y no se usa |
| 2022 | **Cubierto entero** por los tests de los folds 2, 3 y 4. Lo fija un test del runner |

**Decisión D1 (revisar): alineación al inicio.** Alinear las ventanas al final
usaría las 128 barras más recientes y descartaría las primeras 128, que incluyen
la liquidez delgada de agosto y septiembre de 2017. Se descarta esa opción
porque hoy esa elección se tomaría **después** de haber visto que esos meses son
ilíquidos (`VALIDACION.md`), y eso sería excluir datos por lo que se vio en
ellos. Alinear al inicio es la opción que no depende de nada observado.

**Decisión D2 (revisar): el papel de la validación bajo la opción (i).** No se
elige nada sobre la validación, porque no hay hiperparámetros que elegir. Sus
200 barras no entrenan y no seleccionan. Se usan solamente para clasificar el
régimen reciente fuera de entrenamiento (sección 2). El costo es que cada agente
se entrena con datos que terminan 200 días antes de su test. Se declara como
limitación en lugar de reabsorber esas barras al train: la geometría 800/200/200
fue la aprobada, y cambiarla ahora movería la frontera train/test después de
haber visto los precios.

### Por fold

- **Escalador:** se ajusta **solo con la ventana de train** de ese fold
  (`fit_scaler_on_train`), se serializa con el fold y no se reajusta.
- **Calentamiento:** las decisiones del test arrancan en la primera barra del
  test. Los indicadores de esa barra se calculan con barras anteriores, las de
  validación, que son pasado y no fuga. El agente y los tres baselines toman la
  primera decisión en la misma barra (`WarmupDelay`).
- **Inicio:** agente y baselines arrancan cada fold en cash, con el mismo capital.
  El buy-and-hold vuelve a comprar al inicio de cada fold y paga su entrada en
  cada uno, igual que el agente. La simetría importa más que imitar a un
  inversor que compró una sola vez en 2020.
- **Semillas:** las 10 de `STUDY_SEEDS`, las mismas en cada fold. En total, 110
  entrenamientos.

## 2. Criterio principal: Agente A contra buy-and-hold, neto de costos

### Estadístico

Para cada semilla `s`, fold `k` y serie de equity `q ∈ {mark, liquidation}`:

```
E(k,s,q) = log(q_agente[fin_k] / q_agente[inicio_k]) − log(q_BH[fin_k] / q_BH[inicio_k])
```

Es el exceso de crecimiento logarítmico sobre buy-and-hold en el test del fold,
neto de todos los costos, porque ambas series ya los pagaron. Es aditivo entre
folds, y por eso el agregado de una semilla es su suma:

```
X(s,q) = Σ_k E(k,s,q)          (crecimiento de la trayectoria fuera de muestra concatenada)
```

Además, para cada semilla, el **DSR** de la serie diaria de exceso de retorno
simple (agente `mark` − B&H `mark`) sobre las 2.200 barras de test
concatenadas. Usa `n_trials = 10`, las semillas, y como varianza del Sharpe
entre intentos la de esas 10 semillas. La deflación supone que se eligió la
mejor de diez, aunque se reporta la mediana: es conservadora a propósito.

### Veredicto

**SUPERA** si y solo si se cumplen las tres condiciones:

1. `mediana_s X(s, mark) > 0`
2. `mediana_s X(s, liquidation) > 0`
3. `mediana_s DSR(s) ≥ 0.95`

En cualquier otro caso, **NO SUPERA**. Se reportan siempre los tres números, las
dos series, y la distribución completa (mediana, p25, p75, min y max sobre las
semillas), tanto por fold como del agregado. Nunca la mejor semilla.

Los baselines aleatorio y de cruce de medias se reportan con las mismas
métricas. **No entran al veredicto**: el criterio registrado es contra
buy-and-hold.

## 3. Dependencia entre folds

Dos folds consecutivos comparten 600 de sus 800 barras de entrenamiento. Los 11
folds **no son muestras independientes** del desempeño del agente, así que:

- **No se usa ningún test que suponga independencia entre folds.** Nada de `t`,
  p-values, intervalos de confianza ni "k de 11 folds positivos" como prueba
  binomial.
- La agregación es la **trayectoria concatenada**: la suma de `E` sobre folds
  para cada semilla, más la distribución entre semillas. Los tests no se solapan
  y cada barra fuera de muestra cuenta una sola vez.
- El DSR va sobre retornos **diarios**, no sobre folds. Supone retornos diarios
  sin autocorrelación, con corrección por asimetría y curtosis. Es una
  aproximación y se declara.
- Los resultados por fold se muestran para leer **dónde** pasó algo, no como
  observaciones de un contraste.

## 4. La predicción del ADR 0004, operacionalizada

La predicción registrada el 2026-09-07 dice que debe haber degradación fuera de
muestra en las ventanas que **siguen** a un cambio de régimen, y que debe ser
mayor que en ventanas comparables sin cambio, **contra los baselines de la misma
ventana**.

### 4.1 Clasificación de las ventanas (mecánica, sin resultados del agente)

Se definen:

- `R_train`: retorno log total de la ventana de train.
- `R_val` y `t_val = R_val / (σ̂_diaria · √200)`: retorno de los 200 días
  inmediatamente previos al test (la validación, fuera de entrenamiento) y su
  cociente contra su propia volatilidad.

**Decisión D3 (revisar): la regla.** Propusiste comparar el signo de los 200 días
previos contra el signo dominante del train, y así queda. La diferencia es que
agrego un piso de **una desviación estándar** sobre la ventana reciente:

| Clase | Condición |
|---|---|
| **tras cambio** | `signo(R_train) ≠ signo(R_val)` **y** `|t_val| ≥ 1` |
| **sin cambio** | `signo(R_train) = signo(R_val)` **y** `|t_val| ≥ 1` |
| **ambigua** | `|t_val| < 1`. Se excluye de la comparación y se reporta |

Por qué: con el signo solo, retornos indistinguibles de cero deciden la clase.
El fold 5 contaría como "tras cambio" por una validación de −2 %
(`t_val` = −0.04), y el fold 0 como "sin cambio" por una de +2 %. Un cambio de
signo sobre un retorno que es ruido no es una observación de régimen. El piso de
una desviación estándar es el requisito más débil que distingue dirección de
ruido y no está afinado contra nada.

Clasificación resultante, calculada **antes de entrenar ningún agente**:

| Fold | Test | R_train | R_val | t_val | Regla con signo solo | **Regla con piso** |
|---|---|---|---|---|---|---|
| 0 | 2020-05-14..2020-11-29 | +0.71 | +0.02 | +0.02 | sin cambio | ambigua |
| 1 | 2020-11-30..2021-06-17 | −0.27 | +0.70 | +1.82 | tras cambio | **tras cambio** |
| 2 | 2021-06-18..2022-01-03 | +1.00 | +0.77 | +1.16 | sin cambio | **sin cambio** |
| 3 | 2022-01-04..2022-07-22 | +1.99 | +0.21 | +0.41 | sin cambio | ambigua |
| 4 | 2022-07-23..2023-02-07 | +1.70 | −0.71 | −1.36 | tras cambio | **tras cambio** |
| 5 | 2023-02-08..2023-08-26 | +0.97 | −0.02 | −0.04 | tras cambio | ambigua |
| 6 | 2023-08-27..2024-03-13 | +0.25 | +0.14 | +0.40 | sin cambio | ambigua |
| 7 | 2024-03-14..2024-09-29 | −0.39 | +1.01 | +2.96 | tras cambio | **tras cambio** |
| 8 | 2024-09-30..2025-04-17 | +0.41 | −0.08 | −0.21 | tras cambio | ambigua |
| 9 | 2025-04-18..2025-11-03 | +1.05 | +0.24 | +0.64 | sin cambio | ambigua |
| 10 | 2025-11-04..2026-05-22 | +1.31 | +0.27 | +1.10 | sin cambio | **sin cambio** |

Con la regla con piso quedan **3 folds tras cambio (1, 4 y 7), 2 sin cambio
(2 y 10) y 6 ambiguos**. Con el signo solo serían 5 y 6.

Este ADR sella **una sola** de las dos reglas. Correr las dos y quedarse con la
que confirme sería el camino bifurcado que la pre-registración existe para
cerrar.

### 4.2 Métrica de degradación

**Decisión D4 (revisar): no se usa el exceso crudo sobre buy-and-hold.** Ese
exceso depende mecánicamente de la dirección del mercado cuando la exposición
del agente no es 1. Con exposición media `e`, el retorno del agente es
aproximadamente `e · R_mercado`, así que el exceso es aproximadamente
`(e − 1) · R_mercado`. Un agente invertido al 63 %, como en el 4b, "le gana" a
buy-and-hold en toda ventana bajista sin tener ninguna habilidad. La clase "tras
cambio" contiene justamente los giros bajistas, así que la métrica cruda
confundiría la dirección del mercado con la degradación. Es lo que la predicción
misma pidió excluir.

La métrica es el **timing contra un buy-and-hold con la misma exposición**, en la
misma ventana:

```
T(k,s) = log(1 + r_agente_mark) − log(1 + e(k,s) · r_BH_mark)
```

- `e(k,s)` es la fracción media del equity invertida por el agente en el test
  del fold (posición valuada a close sobre `equity_mark`, promediada por barra).
- `cash_rate = 0`, así que la parte no invertida del benchmark rinde cero.

`T` es cero para un agente que mantiene cualquier exposición constante. Es
positivo si se expuso más cuando el mercado subió, y negativo si hizo lo
contrario. Aplicar la regla del régimen anterior después de un giro produce
exactamente eso último. El exceso crudo se reporta al lado, sin decidir.

Por fold se toma `T(k) = mediana_s T(k,s)`.

### 4.3 Qué cuenta como confirmación

```
G = mediana_{k ∈ tras cambio} T(k) − mediana_{k ∈ sin cambio} T(k)
F = mediana_k IQR_s T(k,s)          (piso de ruido de entrenamiento: cuánto se mueve T al reentrenar)
```

| Veredicto | Condición |
|---|---|
| **NO CONTRASTABLE** | Menos de 2 folds en alguna de las dos clases |
| **CONFIRMADA** | `G < −F`: las ventanas tras cambio rinden peor, por más que el ruido de reentrenar |
| **NO CONCLUYENTE** | `−F ≤ G < 0`: van en el sentido predicho, pero no se separan del ruido de entrenamiento |
| **REFUTADA** | `G ≥ 0`: las ventanas tras cambio no rinden peor |

**Límite declarado:** con 3 contra 2 folds, dependientes, esto es un contraste
de potencia mínima. `F` mide solo varianza de entrenamiento: sobre un camino
real la varianza de mercado no se puede medir. Un CONFIRMADA sería consistente
con la predicción y no la probaría. Un REFUTADA sobre 5 folds tampoco la
enterraría. Lo que sí hace esta sección es impedir que después se elija qué
folds "eran" cambios de régimen.

**Subsidiaria 3** ("invertido de más justo después del giro"), en su forma
mecánica: en cada fold tras cambio, la exposición media de las primeras 50
barras del test se aparta de la exposición del resto del test **en la dirección
del régimen de train**. Eso es más invertido si el train era alcista, menos si
era bajista. Se reporta con signo por fold. Se considera sostenida si ocurre en
la mayoría de los folds tras cambio.

**Subsidiaria 2** (la LSTM no cierra la brecha). **Decisión D5 (revisar): C3 corre
solo MLP**, así que la subsidiaria 2 **no se contrasta** sobre datos reales en
esta tarea. Se declara en lugar de omitirla. Agregar el brazo LSTM duplica el
cómputo, de 110 a 220 entrenamientos.

## 5. Ejecuciones outlier

Una ejecución es **outlier** si su barra de decisión está marcada
(`close_time_desvio_ms ≠ 0`) **o** si entre la barra de decisión y la de
ejecución falta al menos una barra del calendario.

- **El resultado principal incluye todos los fills.** Filtrar outliers del
  resultado principal sería elegir qué ejecuciones cuentan.
- Aparte se reportan (a) la lista de fills outlier con su gap y (b) las métricas
  recalculadas sin ellos, como versión secundaria.
- En el diario, la única barra marcada (2018-02-08) cae en el train del fold 0 y
  el calendario no tiene huecos. **No se espera ningún fill outlier en los
  tests.** La regla se implementa igual y el reporte lo dice con un número, no
  por omisión.

## 6. Limitaciones

1. **Sesgo de selección de BTC.** Se eligió BTCUSDT porque sobrevivió y es el
   activo cripto más líquido de hoy. Un estudio que empieza por el ganador
   conocido sobreestima lo que cualquier estrategia larga habría ganado.
   **Recordarlo en la Etapa 6**: el universo del barrido no puede construirse con
   la lista de hoy.
2. **Entre 700 y 900 barras de entrenamiento por fold** (774 efectivas, tras
   descontar el calentamiento). Es poco para PPO y favorece la memorización.
3. **N = 1 camino por construcción.** La historia es una sola. La dispersión
   entre semillas mide varianza de entrenamiento, no de mercado. No hay forma de
   medir el error de muestreo; el walk-forward solo lo acota.
4. **Liquidez temprana de agosto y septiembre de 2017.** Solo afecta al
   **entrenamiento del fold 0**, y a ninguna ventana de test. El salto entre open
   y close anterior llega al 0,81 % en el diario (p99 0,21 %).
5. **Regla de outliers:** barra de decisión marcada **o** al menos una barra
   faltante entre decisión y ejecución (sección 5).
6. **Política subentrenada** (sección 0). Valida el pipeline, no concluye sobre el
   mercado.
7. **Agentes entrenados con datos que terminan 200 días antes de su test**
   (decisión D2).
8. **La predicción 1 tiene potencia mínima**: 3 contra 2 folds, dependientes
   (sección 4.3). La subsidiaria 2 no se contrasta (D5).
9. **DSR sobre retornos diarios**, que supone ausencia de autocorrelación
   (sección 3).
10. **128 barras finales sin usar** (2026-05-23 a 2026-09-27).

## 7. Sellado

1. Este ADR se commitea con las decisiones D1 a D5 resueltas y **se sella**: su
   SHA256 queda registrado en el runner como constante.
2. El runner de C3 **se niega a correr** si el SHA256 del ADR en disco no
   coincide con el registrado. Así, cambiar el criterio después obliga a
   cambiar la constante, y eso se ve en el diff.
3. El test se toca **una vez**. El runner escribe un sello (dataset SHA256 +
   hash de la configuración + hash del ADR) antes de evaluar el primer fold, y
   aborta si el sello ya existe. Una segunda corrida exige borrar el sello a
   mano, y eso queda en la historia.
4. La salida de C3 registra el hash del commit del ADR sellado, como en el 4b.
