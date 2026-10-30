# ADR 0013 — Segunda ronda del Agente B: noticias suavizadas, exploración sobre el test visto y confirmación en un fold nuevo

- **Estado:** propuesto. Pendiente de tres decisiones de Kevin: **R1** (sección 3,
  qué se cambia en B), **R2** (sección 5, si se corre la Fase 1) y **R3** (sección 6,
  cuánto dura el test nuevo). El texto está escrito con las tres opciones
  recomendadas; si Kevin elige otra, el PR cambia antes de mergearse. Se sella al
  mergearse: los runners de esta ronda guardan el SHA256 de este archivo y no corren si
  el del disco es otro, como en los ADR 0006 y 0012.
- **Fecha:** 2026-10-30
- **Contexto de etapa:** Etapa 5, después del veredicto del ADR 0012. **Esta ronda no
  estaba pre-registrada** en el ADR 0012 y se escribe después de ver sus resultados.
  La sección 1 dice qué se vio y cómo se compensa.

Este ADR no cambia el veredicto del ADR 0012 (**NO APORTA, plano**), que queda sellado
como el resultado del estudio pre-registrado. Lo que hace es pre-registrar una pregunta
nueva antes de escribir código o entrenar nada: **¿un Agente B con noticias mejor
construidas aprende a usarlas, y eso aguanta en datos que nadie miró?**

---

## 0. Lo que hay que saber antes de decidir: la potencia

La diferencia que midió N9 entre B y A es demasiado chica para que **cualquier** test
sobre barras diarias con un solo camino la confirme. Con el DSR con `n_trials` = 1 de
N9 (mediana 0,573 para B − A y 0,804 para B − placebo) se despeja el Sharpe diario de
la diferencia, y con él la probabilidad de cruzar 0,95 según el largo del test
(aproximación normal, sin la deflación por intentos, que la empeora):

| Efecto supuesto | Sharpe diario | anualizado | 128 barras | 200 barras | 400 barras | 2 200 barras |
|---|---|---|---|---|---|---|
| B − A de N9 | 0,004 | 0,07 | 6 % | 6 % | 6 % | 7 % |
| B − placebo de N9 | 0,018 | 0,35 | 8 % | 8 % | 10 % | 22 % |
| Efecto con 50 % de potencia a 200 barras | 0,117 | 2,2 | 37 % | 50 % | 75 % | ~100 % |

Con un efecto del tamaño de N9, confirmar B − A con 50 % de probabilidad exigiría unos
175 000 días. **Esta ronda solo puede dar APORTA si las noticias nuevas producen un
efecto de 6 a 30 veces el de N9.** Lo más probable es otro NO APORTA, y está bien: lo
que la ronda sí responde con seguridad es la pregunta mecánica que dejó N8e (¿el agente
llega a usar las noticias?) y, con un fold nuevo, si un efecto grande existe.

Si Kevin prefiere no gastar ~8 h de su PC en una pregunta con esa potencia, la
alternativa honesta es no correr la ronda y dejar el estudio en el ADR 0012.

## 1. Riesgo de sesgo: qué se vio y cómo se compensa

**Qué se vio de N9 y N8e** sobre los tests de los folds 0 a 10 (2020-05-14 a
2026-05-22): el criterio K3 completo, `X` por brazo y por fold, el secundario de
exposición igualada (B +0,22), la σ de la política, y la sensibilidad de la acción por
**grupo** de columnas, por fold y por política (`rediagnosis.json`). No se midió la
sensibilidad **por columna de noticias**: N8e fija las siete juntas.

**Qué no vio nadie:** ningún agente, ninguna feature de noticias y ningún modelo se
evaluó sobre las barras posteriores al 2026-05-22. Las 128 barras hasta el 2026-09-27
están en `datasets/binance/BTCUSDT-1d.csv` y se excluyeron a propósito del test del
ADR 0012 (§1). El precio de BTC en esos meses es público y cualquiera lo conoce; lo
que nadie sabe es cómo le va a un agente con noticias ahí.

**Compensaciones, todas fijas desde este ADR:**

1. **Test nuevo para el veredicto.** El único veredicto APORTA / NO APORTA sale del
   **fold 11** (sección 6), que nadie tocó. Lo que se corra sobre los folds 0 a 10 es
   exploratorio y se rotula así en cada reporte.
2. **Corrección por intentos.** Toda configuración de B evaluada sobre un tramo de
   test suma a `n_trials` de ese tramo (sección 7).
3. **Todo fijado antes.** Las features (sección 3), los criterios de las tres fases y
   las reglas de lectura están en este ADR. **Ningún resultado de las Fases 0 y 1
   cambia las features ni el criterio de la Fase 2.** Si después de la Fase 1 se
   quiere otra variante, es una tercera ronda con su propio ADR y su propio fold nuevo
   (el 12, completo hacia junio de 2027), porque el 11 ya quedó comprometido aquí.
4. **Placebo con políticas guardadas.** Se guardan las políticas de todas las semillas
   de B2 y del placebo, y el re-diagnóstico corre sobre las dos (sección 5.3). N9 solo
   guardó las de B y no permitió comparar contra un agente placebo.
5. **El diseño de las features usa de N9 una sola cosa:** que B no usa las noticias.
   Las razones de cada cambio (sección 3.2) son genéricas, no de un fold ni de una
   columna.

## 2. Lo que no cambia

Todo lo de la sección 1 del ADR 0012: precios, geometría de 11 folds de 1 000 + 200
con paso 200, costos y capital, `STUDY_PPO` y `STUDY_OBSERVATION` (R3 a 120 000
timesteps, red 32×32, `lookback` 1), las 10 `STUDY_SEEDS`, el sizer, la latencia de 60
minutos (K2), la alineación de lotes a barras (§2.3), la deduplicación (§2.4), la
cobertura (§2.5), FinBERT pinneado (§3.1), el placebo a 364 días (§4) y el criterio K3
con sus etiquetas (§6). **B2 no se calibra aparte** (ADR 0011 §2).

## 3. Qué se cambia en B — **R1**

### 3.1 Opciones

| Opción | Qué cambia | A favor | En contra | Costo en el PC de Kevin |
|---|---|---|---|---|
| **F2: noticias suavizadas (recomendada)** | Las 7 columnas de noticias por 7 versiones estandarizadas contra su historia y promediadas en 7 días (3.2). Misma red, mismo presupuesto | Ataca justo lo que encontró N8e con el cambio más chico. Mismo número de columnas que N9, así que el efecto "más columnas" no cambia. A no cambia y se reutiliza de N9: no hay tercer toque del test para A | Si GDELT diario no tiene señal, ningún arreglo de features la crea. Elegir features después de ver N9 es justo el riesgo de la sección 1; lo cubre el fold nuevo | ~8 h |
| LSTM | Política recurrente, mismas 7 columnas | Memoria de varios días sin diseñar features a mano | La evidencia a favor es débil: en la calibración el LSTM le ganó al MLP por 0,045 de `capture` en el nivel 3, donde **los dos memorizan en 10 de 10 caminos** y sin columnas exógenas; no dice nada sobre noticias. Es una arquitectura nueva: hay que pasarle el protocolo completo y la puerta 5a antes, y re-correr A en LSTM, que es un tercer toque del test para A | ~40 h: protocolo (~15 h), puerta 5a (~5 h), tres brazos LSTM (~20 h) |
| Más timesteps (240 000) | El doble de presupuesto | Si B estaba subentrenado para las noticias, es lo directo | El ADR 0010 mostró que más presupuesto sin regularizar hace rotar a 4b; con R3 no se sabe y hay que re-correr el protocolo. Más pasadas por 1 000 barras favorecen memorizar los técnicos, que ya dominan. A también cambia (tercer toque) | ~42 h: protocolo (~22 h), tres brazos (~20 h) |

**Recomendación: F2.** Cuesta cinco veces menos, no exige re-validar el agente y no
vuelve a tocar el test con A. Si F2 tampoco usa las noticias, el LSTM queda como
tercera ronda, ya con la respuesta de la Fase 0 sobre si hay algo que aprender.

### 3.2 Las siete columnas de F2

Las siete de B en N9 eran niveles diarios (`news_share`, `tone_mean`, `finbert_mean`) o
deltas de un día contra pocos días. Dos razones genéricas para cambiarlas, ninguna
sacada de un fold:

- **Los niveles derivan.** La composición de fuentes de GDELT cambia en nueve años; un
  nivel crudo estandarizado con las 1 000 barras del entrenamiento sigue derivando
  dentro de la ventana. Un z contra las 30 barras previas es estacionario por
  construcción.
- **Un día de noticias es ruido.** El valor de un día varía mucho más que el estado
  informativo que se quiere medir. Un promedio de 7 días le da memoria al MLP con
  `lookback` 1 sin cambiar la arquitectura: es la versión barata del LSTM.

| # | Feature | Definición | Máscara | Reemplaza a |
|---|---|---|---|---|
| 1 | `news_share_z30` | igual que en el ADR 0012 | igual | (se queda) |
| 2 | `news_share_z30_m7` | media de `news_share_z30` en las barras válidas de `t−6..t` | menos de 4 válidas | `news_share` |
| 3 | `tone_z30` | `tone_mean` contra media y desvío de las 30 barras previas válidas | menos de 20 válidas o desvío 0 | `tone_mean` |
| 4 | `tone_z30_m7` | media de `tone_z30` en las barras válidas de `t−6..t` | menos de 4 válidas | `tone_delta7` |
| 5 | `finbert_m7` | media de `finbert_mean` en las barras con título de `t−6..t` | menos de 4 con título | `finbert_mean` |
| 6 | `no_title` | igual que en el ADR 0012 | — | (se queda) |
| 7 | `coverage_ok` | igual que en el ADR 0012 | — | (se queda) |

- **Causales:** cada ventana termina en `t` y usa solo barras alineadas con la regla
  del ADR 0012 §2.3. "Válida" es "no enmascarada". Nunca `ffill`.
- **Escalado:** el del ADR 0012 §3 (columnas 1 a 5 con el escalador ajustado solo en
  los valores no enmascarados del entrenamiento del fold; enmascarado entra como 0;
  indicadores sin escalar).
- **Placebo de F2 (`placebo2`):** las mismas siete columnas de la barra `t − 364`.
- `no_title` se mantiene por la decisión de Kevin del 2026-10-23: "sin títulos" no
  debe confundirse con "títulos neutros".

## 4. Fase 0: ¿las features predicen algo? (minutos, en la nube)

Antes de gastar horas de RL, un chequeo lineal: si ni un modelo lineal saca de las
noticias algo que el precio no tenga, la Fase 1 casi no tiene qué aprender.

- **Solo sobre los tests de los folds 0 a 10.** El fold 11 no se toca (el runner lo
  verifica: rechaza cualquier barra posterior al 2026-05-22).
- **Por fold:** regresión ridge con λ = 1, fijo y sin búsqueda, ajustada en las 1 000
  barras del entrenamiento del fold y evaluada en sus 200 de test. Las columnas se
  escalan con las estadísticas del entrenamiento.
- **Columnas base `P`:** retornos y técnicos de `STUDY_OBSERVATION` (los grupos
  `retornos` y `tecnicos` del re-diagnóstico; sin posición ni resultado de órdenes,
  que dependen del agente).
- **Cinco modelos:** `P`, `P + N9`, `P + F2`, `P + placebo N9`, `P + placebo F2`.
- **Dos objetivos:** `y = log(open[t+2] / open[t+1])`, el retorno que gana una decisión
  tomada al cierre de `t` y ejecutada al open de `t+1`, y `|y|`.
- **Métrica:** IC de Spearman entre predicción y objetivo sobre las 2 200 barras de
  test juntas. `ΔIC(X) = IC(P + X) − IC(P)`.

**Regla de parada (solo frena la Fase 1):** la Fase 1 corre si y solo si, para al menos
uno de los dos objetivos, `ΔIC(F2) − ΔIC(placebo F2) ≥ 0,03`. Es laxa a propósito (el
error estándar de un IC sobre 2 200 barras es ~0,021): frena solo cuando no hay ni un
indicio. Los resultados de N9 se reportan al lado, sin que decidan. Se corre una vez.

La Fase 0 no frena la Fase 2: un efecto no lineal podría escaparse a una regresión
lineal, y la Fase 2 es barata.

## 5. Fase 1: exploración sobre el test ya visto — **R2**

### 5.1 Qué se corre

| Brazo | Observación | Entrenamientos |
|---|---|---|
| A | El de N9, **reutilizado** de `records-n9.zip` (Release `n9-resultados`) | 1 de verificación |
| B2 | 15 columnas de A + las 7 de F2 en `t` | 110 (11 folds × 10 semillas) |
| placebo2 | 15 columnas de A + las 7 de F2 en `t − 364` | 110 |

**A se reutiliza.** Su configuración no cambia, y re-correrlo sería un tercer toque del
test para A sin información nueva. Antes de reutilizarlo, el runner reentrena una
unidad (fold 0, semilla 11) y exige que la serie de acciones sea **idéntica bit a bit**
a la de N9. Si no lo es, se re-corre A entero (+110 entrenamientos, ~3 h) y el reporte
lo declara como tercer toque.

Se guardan las políticas de **todas** las semillas de B2 y de placebo2 (220).

### 5.2 Qué se reporta

Las seis cantidades del criterio K3 (ADR 0012 §6) para B2 contra A y contra placebo2,
con DSR con `n_trials` = 20 (sección 7) y el honesto de 2 al lado, más los extras del
ADR 0012 §8. **La Fase 1 no da veredicto APORTA:** cada tabla lleva el rótulo
"exploratorio, test ya visto (ADR 0013 §1)".

### 5.3 Re-diagnóstico, con regla de lectura fijada aquí

El de N8e (cinco grupos, mismas barras de test) sobre las 110 políticas de B2 y las 110
de placebo2. Sobre B2, la mediana del cambio de acción al fijar las noticias:

- **< 0,02:** "B2 no usa las noticias", como en el ADR 0012.
- **≥ 0,05** (el doble del piso de ruido de la puerta 5b, 0,0252): "B2 usa las
  noticias".
- **Entre las dos:** "indicio débil".

Y la comparación que N9 no permitió: si placebo2 es tan sensible a sus columnas como B2
a las suyas (diferencia de medianas < 0,01), la sensibilidad viene de tener columnas,
no de su contenido, y el reporte lo dice así.

### 5.4 Opciones de R2

| Opción | A favor | En contra |
|---|---|---|
| **Fase 1 si la Fase 0 la habilita (recomendada)** | Responde la pregunta mecánica de N8e en 11 regímenes y 110 políticas, y no gasta 7 h si no hay nada que aprender | Si la Fase 0 la frena, la respuesta mecánica queda solo con las 10 políticas del fold 11 |
| Fase 1 siempre | La respuesta mecánica existe pase lo que pase | 7 h que la Fase 0 pudo mostrar inútiles |
| Sin Fase 1, solo Fase 2 | Lo más barato (~1 h); ninguna mirada más al test viejo | La pregunta mecánica queda con un solo régimen, y si la Fase 2 da NO APORTA no se sabrá si B2 usó las noticias en otros mercados |

## 6. Fase 2: confirmación en el fold 11 — **R3**

### 6.1 El fold nuevo

El fold 11 es el siguiente de la misma geometría: test de 200 barras con cierres del
**2026-05-23 al 2026-12-08**, entrenamiento en las 1 000 barras previas (desde
~2023-08). Las primeras 128 ya existen y nadie evaluó nada sobre ellas; las últimas 72
son posteriores a este ADR y no las pudo ver nadie. Los datos completos existen desde
el 2026-12-09.

| Brazo | Entrenamientos |
|---|---|
| A (fold 11, primera vez sobre estos datos) | 10 |
| B2 | 10 |
| placebo2 | 10 |

### 6.2 Orden, para que el test se toque una vez

1. Se entrenan los 30 con datos hasta el 2026-05-22. Se puede hacer apenas esté el
   código, sin esperar diciembre.
2. Un sello (`SEAL.json`) registra el SHA256 de este ADR, de los 30 pesos, de los
   datasets de entrenamiento y el commit, **antes** de evaluar ninguna barra del test.
3. Después del 2026-12-09 se versionan los datos nuevos (6.4) y se evalúa una sola vez.
   El evaluador se niega si algún peso no coincide con el sello.

La Fase 2 corre aunque la Fase 0 o la Fase 1 salgan mal: es el único veredicto de la
ronda, y sus features y criterio están fijos aquí.

### 6.3 Criterio

Las seis condiciones del K3 (ADR 0012 §6) con `X` = el exceso del fold 11 y el DSR
sobre sus 200 barras, `n_trials` = 10 (convención del ADR 0006; el número honesto de
configuraciones evaluadas sobre el fold 11 es 1). **APORTA (CONSISTENTE)** si se
cumplen las seis; si no, **NO APORTA** con una de las tres etiquetas del ADR 0012. Con
la potencia de la sección 0, un NO APORTA aquí **no refuta** un efecto del tamaño de
N9, y el reporte lo dice en el titular.

Secundarios, que no deciden:

- A contra buy-and-hold en el fold 11 con las tres condiciones del ADR 0006 §2. Es el
  primer toque de A sobre datos nuevos.
- Las mismas cantidades solo sobre las 72 barras posteriores a este ADR.
- El re-diagnóstico de 5.3 sobre las 10 políticas de cada brazo.
- **Paridad histórico-vivo:** para los días del 2026-10-20 al 2026-12-08, qué fracción
  de los lotes que bajó el descargador histórico coincide con lo que grabó L1 en la VM.

**Asimetría declarada:** el placebo2 del fold 11 lee noticias del 2025-05-24 al
2025-12-09, que incluyen el apagón de 17 días de 2025: tiene ~19 barras de test con
`coverage_ok` = 0 que B2 no tiene.

### 6.4 Datos nuevos

- **Precios:** nueva versión de `BTCUSDT-1d.csv` hasta el 2026-12-09, con manifest. Las
  filas hasta el 2026-09-27 tienen que ser idénticas byte a byte a las de la versión
  actual; si no, se para y se investiga.
- **Noticias:** el descargador histórico de N2 del 2026-09-28 al 2026-12-09, agregado
  con el mismo código de N3, y los títulos nuevos puntuados con el mismo FinBERT
  pinneado (N2b). Los lotes existentes no se reprocesan. Se usa el histórico y no lo
  grabado por L1 para que el test se arme igual que el entrenamiento; L1 sirve de
  chequeo de paridad (6.3).
- La latencia sigue en 60 minutos; el p99 en vivo se vuelve a medir y se registra en
  el sello de la evaluación.

### 6.5 Opciones de R3

| Opción | A favor | En contra |
|---|---|---|
| **Fold 11 completo, 200 barras (recomendada)** | Misma unidad que los demás folds; 72 barras imposibles de haber visto | Esperar al 2026-12-09 |
| Solo las 128 barras que ya existen | Resultado en noviembre | Menos potencia (sección 0), y ninguna barra es posterior a la decisión |
| 400 barras (folds 11 y 12) | 75 % de potencia contra el efecto de 0,117, en vez de 50 % | Resultado hacia junio de 2027, y la tercera ronda pierde su fold nuevo |

## 7. Corrección por número de intentos

`n_trials` cuenta las configuraciones de B evaluadas sobre cada tramo de test. El
decisivo sigue la convención del ADR 0006 (10 por configuración); el honesto se
reporta al lado y no decide.

| Tramo | Configuraciones de B evaluadas | Honesto | Decisivo |
|---|---|---|---|
| Folds 0 a 10 (2020-05 a 2026-05) | B de N9, B2 | 2 | 20 |
| Fold 11 (2026-05 a 2026-12) | B2 | 1 | 10 |

La Fase 0 es una mirada a los folds 0 a 10, pero no es una configuración de B: se
declara en el reporte y no suma. Para A sobre los folds 0 a 10 el honesto sigue en 2
(C3 y R3), salvo que la verificación de 5.1 obligue a re-correrlo.

## 8. Cómputo

| Paso | Dónde | Entrenamientos | Tiempo |
|---|---|---|---|
| Fase 0 | nube | 0 | minutos |
| Fase 1 (si la habilita la Fase 0) | PC de Kevin, 3 procesos | 221 | ~7 h |
| Fase 2, entrenamiento | PC de Kevin | 30 | ~1 h |
| Fase 2, datos nuevos y evaluación (después del 2026-12-09) | nube | 0 | ~1 h |

La puerta 5a no se re-corre: mostró que el pipeline le hace llegar a B una columna
exógena y que B la aprende, y F2 usa el mismo pipeline, la misma red y la misma
cantidad de columnas. Si R1 elige el LSTM, la puerta y el protocolo se corren antes
(sección 3.1).

Partición propuesta en PRs, cada uno con su hilo y después de mergeado este ADR:

1. **R2a:** las columnas de F2 y su placebo en `features/news.py`, con tests de
   anti-leakage (~200 líneas).
2. **R2b:** la Fase 0 (`study screen`) con la guarda que rechaza barras del fold 11
   (~150).
3. **R2c:** el runner de la ronda: brazos B2 y placebo2, reutilización de A con la
   verificación bit a bit, todas las políticas guardadas, modo fold 11 con sello antes
   de evaluar (~300).
4. **R2d** (después del 2026-12-09): datos nuevos versionados y evaluación de la
   Fase 2 con su reporte.

## 9. Sellado y adendas

Como el ADR 0012 §9: antes del primer sello, este ADR solo cambia por motivos de
ingeniería, con una adenda que diga qué y por qué. **Nunca por un resultado de las
Fases 0 o 1.** Cada fase tiene su `SEAL.json`, escrito antes de su primer
entrenamiento o, en la Fase 2, antes de evaluar.

## 10. Limitaciones

1. **Potencia:** la sección 0. Es la limitación que domina la ronda.
2. **N = 1 camino** en la Fase 2 también. Diez semillas en el fold 11 miden varianza
   de entrenamiento.
3. **Un solo fold de 200 barras** decide, en un solo régimen de mercado.
4. **Diseño posterior a ver N9** (sección 1): lo compensa el fold nuevo, no lo borra.
   Un APORTA en el fold 11 se lee como "una hipótesis sacada de datos viejos aguantó
   en datos nuevos".
5. Las limitaciones 2 a 9 y 11 del ADR 0012 §11 (barras diarias, prensa general,
   hora de detección, sin títulos antes de 2019-09, filas sin deduplicar, apagones,
   red chica, σ ~1,0, costos supuestos) siguen igual.
6. **Placebo2 del fold 11 con el apagón de 2025** (6.3).

## 11. Lo que este ADR no cambia

- El veredicto del ADR 0012 ni ninguno de sus artefactos sellados.
- Los ADR 0006, 0007 (no se edita), 0009, 0010 y 0011.
- La sombra y sus Agentes A congelados: siguen su ventana hasta ~2026-11-18.
- `STUDY_PPO`, `STUDY_OBSERVATION`, los umbrales del protocolo y la regla de N1.
