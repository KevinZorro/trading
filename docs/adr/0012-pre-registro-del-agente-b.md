# ADR 0012 — Pre-registro del Agente B: noticias de GDELT sobre el walk-forward diario

- **Estado:** propuesto. Quedan tres decisiones de Kevin, marcadas **K1**, **K2** y
  **K3** (secciones 7, 2.2 y 6). Mientras estén abiertas, el texto describe la opción
  recomendada. Se sella al mergearse: el runner de la corrida (PR N8) guarda el SHA256
  de este archivo y no corre si el del disco es otro, igual que el del ADR 0006.
- **Fecha:** 2026-10-25
- **Contexto de etapa:** Etapa 5. Es el PR N7 del plan de la Etapa 4
  (`etapa4-noticias/PLAN.md`, en los archivos del proyecto), que todavía lo llama
  "ADR 0011" en un par de lugares: el 0011 lo tomó la regularización del Agente A
  (#60).
- **A la fecha de este ADR no existe ninguna feature de noticias, no se entrenó
  ningún Agente B y nadie cruzó noticias con precios.** Lo mirado de GDELT son
  metadatos de la fuente: cobertura por lote, existencia del título, apagones y
  retraso de publicación de los lotes (secciones 2.2 y 2.5). Ninguno usa retornos.
  Del Agente A sí se miró un resultado sobre datos reales: C3 (60 000 timesteps,
  NO SUPERA, `results/c3/REPORTE.md`). Eso pesa sobre K1.

Este ADR se escribe **antes** de que existan el dataset versionado (N3), FinBERT
(N2b), las features (N4), las columnas exógenas del entorno (N5) y el fixture con canal
exógeno (N6). El plan lo ponía después de N3 y N6. Adelantarlo es más estricto, no
menos: el diseño de B queda fijo antes de que exista código que permita mirar algo.
El costo es que N3 a N6 pueden encontrar un problema de ingeniería que obligue a
cambiar un detalle; para eso está la regla de adendas de la sección 9, que deja fuera
cualquier cambio motivado por un resultado sobre datos reales.

---

## 0. Qué puede concluir esta corrida, y qué no

- **N = 1 camino real.** CONSISTENTE es el veredicto máximo, igual que en el ADR 0006.
  La dispersión entre semillas mide varianza de entrenamiento, no de mercado.
- **Barras diarias.** Se mide si el *estado informativo del día* (atención y tono de
  la prensa) ayuda a decidir la exposición del día siguiente. No se mide operar
  noticias: en cripto eso se descuenta en minutos u horas. Un NO APORTA aquí no
  entierra la pregunta intradía, y el reporte lo dice así.
- **Prensa general, no conversación cripto.** GDELT mide medios generales. Un NO
  APORTA admite la lectura "la fuente era mala".
- **Un activo y un capital** (BTCUSDT, 100 USDT). El barrido es la Etapa 6.
- **Lo más probable es NO APORTA.** Con 1 000 barras por fold y una señal diaria
  débil, es el resultado esperado y es válido.

## 1. Lo que está fijo antes de este ADR y no se toca

| | Valor | Fuente |
|---|---|---|
| Precios | `datasets/binance/BTCUSDT-1d.csv`, SHA256 `0940f089…49cc8` | ADR 0006 |
| Geometría | 11 folds, 1 000 de entrenamiento (train + validación) / 200 de test / paso 200, alineado al inicio, tests 2020-05-14 a 2026-05-22 | ADR 0006 §1 |
| Costos y capital | Corwin-Schultz con tope de 200 bps, comisión 10 bps, raíz cuadrada con `k` = 0,1, 100 USDT; sensibilidad de solo comisión sin reentrenar | ADR 0006 §1.1 |
| Agente | `STUDY_PPO` y `STUDY_OBSERVATION` de `agents.study`: `C*` = R3 a 120 000 timesteps (`lookback` 1, lr 1e-4, 5 épocas, red 32×32, `gamma` 0,9) | ADR 0011, #64 |
| Semillas | Las 10 `STUDY_SEEDS`, las mismas en cada fold y en cada brazo | ADR 0006 |
| Sizer | El de `main`, con la regla del cero numérico (#20) | `sim/sizing.py` |

**B no se calibra aparte** (ADR 0011 §2). Importa `STUDY_PPO` y `STUDY_OBSERVATION`
de `agents.study` y les agrega solo las columnas de la sección 3. Si la red de 32×32
le queda chica a B con 7 entradas más, eso es parte de lo que se mide y se declara
como sesgo en contra de B (sección 11).

La cola de 128 barras sin usar (2026-05-23 a 2026-09-27) y todo lo que grabe L1
desde el 2026-10-20 **no** son parte de este test. Lo grabado en vivo es el período
fuera de muestra genuino de un Agente B en sombra (Etapa 7).

## 2. Datos de noticias

### 2.1 Fuente

GDELT GKG 2.0 en archivos crudos, tema `ECON_BITCOIN`, del 2016-08-01 al 2026-09-28:
2 740 114 artículos en 356 160 lotes de 15 minutos, 7 148 lotes que no existen (2 %).
Lo bajó N2 (#51, #55, #56) al mismo formato que el grabador en vivo (L1), así que
histórico y vivo se leen con el mismo código (`data.news.load_news`).

El dataset que usa la corrida es **el que versione N3**, con su manifest. Su SHA256 no
se puede escribir aquí porque todavía no existe; el sello de N8 lo registra
(sección 9) y el runner se niega a correr si el del disco difiere del manifest de N3.

### 2.2 `pub_time` y latencia — **K2**

`pub_time` = rótulo del lote + latencia. Es la regla de N1 (`NewsDataset.usable`), que
ya está en `main` y tiene sus tests de anti-leakage. GDELT no da la hora de
publicación original sino la de detección, que es posterior: sesgo conservador
(principio 2, `ASSUMPTION` ya declarado en N1).

Lo medido sobre la latencia, sin cruzar con precios:

- **En vivo (L1, 2026-10-20):** GDELT publica casi siempre entre 11 minutos antes y
  unos segundos después del rótulo, pero los lotes de las 20:15 y 21:15 llegaron unos
  **45 minutos tarde**. Con 30 minutos, el backtest habría usado esos dos lotes antes
  de que la versión en vivo pudiera tenerlos.
- **En el histórico (`Last-Modified` de los 349 012 lotes `ok`, medido el
  2026-10-25):** la mediana es 9,2 minutos **antes** del rótulo, el p99 es 1,0 minuto
  después y el p99,9, 5,7 minutos. Solo 67 lotes (0,019 %) tienen `Last-Modified` más
  de 30 minutos después del rótulo y 54 más de 60; 58 de los 67 son de 2016 a 2018,
  y ninguno pasa de 60 desde 2024. `Last-Modified` es una cota: un archivo resubido
  después lo corre, y no mide cuándo apareció el lote en `masterfilelist.txt`.

El histórico dice que 30 minutos sobra casi siempre; lo único que lo contradice es lo
observado en vivo, que es justamente lo que importa para la paridad backtest-vivo.

**Qué afecta en barras diarias.** Con el cierre a las 00:00 UTC, la latencia solo
decide a qué día van los lotes de la última hora o dos: con 30 minutos, el último
lote del día es el de las 23:30; con 60, el de las 23:00. Es el 1 % al 2 % de los
lotes del día, así que la decisión cambia poco la información y mucho la garantía.

**Recomendada: 60 minutos.** Cubre lo observado en vivo con margen, mueve dos lotes
por día al día siguiente, y la regla se mantiene: si L1 mide más antes del sello de
N8, se sube al siguiente múltiplo de 15 minutos **antes** de correr, nunca después.
La alternativa es quedarse en 30 (el valor de N1 y del plan) y declarar los lotes
tardíos como fuga conocida y acotada. Antes del sello, N8 recalcula el p99 del
retraso en vivo con los lotes que tienen `last_404_at` y lo escribe en el sello.

En vivo la latencia es la real: la sombra decide a las 00:00:30 y usa solo lo
recibido hasta entonces. La latencia del histórico existe para que el backtest no use
lo que la versión en vivo no tendría.

### 2.3 De lotes a barras

Un lote pertenece a la barra `t` si y solo si
`close_time[t−1] < rótulo + latencia ≤ close_time[t]`. Es la regla de alineación de
`CLAUDE.md` aplicada a la ventana de la barra: cada lote cae en exactamente una barra,
y en la primera en la que ya estaba disponible. Ningún lote se mira dos veces y
ninguno se adelanta.

### 2.4 Deduplicación

N2 encontró que GDELT **no trae `PAGE_TITLE` antes de ~2019-09-23** y que la
sindicación sin título no se puede deduplicar. Para que las features signifiquen lo
mismo en todo el rango:

- `news_share` y el tono de GDELT se calculan **sobre filas, sin deduplicar**, en el
  numerador y en el denominador (`rows_theme` / `rows_total`). Deduplicar solo desde
  2019-09 cambiaría la escala de la feature a mitad de los entrenamientos de los
  folds 0 a 3.
- FinBERT, que solo existe donde hay título, deduplica por título normalizado
  (minúsculas, espacios colapsados, sin puntuación) dentro de las 24 h previas,
  quedándose con la **primera** aparición. Quedarse con otra sería mirar adelante.

### 2.5 Cobertura

Un lote es **sano** si su estado es `ok` y su `rows_total` es al menos el 25 % de la
mediana de `rows_total` de los lotes `ok` de los 30 días previos (ventana causal, sin
el lote mismo). Así un lote degradado de 6 filas, como los de 2017, no cuenta como
cobertura.

`coverage_ok(t)` = 1 si la barra tiene al menos 87 de sus 96 lotes sanos (90 %).

Apagones de más de 12 h conocidos (N2), con dónde caen:

| Apagón | Barras | Cae en |
|---|---|---|
| 12 tramos en oct-nov 2020 | ~1140 a 1200 | test del fold 0; entrenamiento de los folds 1 a 5 |
| 2021-08-15 | 1458 | test del fold 2 |
| 2022-11-10/11 | 1910 | test del fold 4 |
| 2023-03-22/23 | 2042 | test del fold 5 |
| **2025-06-14 a 2025-07-02 (17 días)** | 2857 a 2875 | **test del fold 9** |

En esos días B ve `coverage_ok` = 0 y las features enmascaradas. **Nunca `ffill`**, y
menos a través de la frontera train/test.

## 3. Features del Agente B

Siete columnas, todas causales, que se suman a las 15 de A. Están fijas aquí; un
cambio después del sello es otra configuración y cuenta en `n_trials`.

| # | Feature | Definición | Máscara |
|---|---|---|---|
| 1 | `news_share` | `log(1 + 10⁴ · Σ rows_theme / Σ rows_total)` sobre los lotes sanos de la barra | sin lotes sanos |
| 2 | `news_share_z30` | `news_share` contra media y desvío de las 30 barras previas válidas | menos de 20 válidas |
| 3 | `tone_mean` | media del tono de GDELT (`V2Tone`, primer campo) de las filas del tema en la barra | sin filas |
| 4 | `tone_delta7` | `tone_mean` menos su media en las 7 barras previas válidas | menos de 4 válidas |
| 5 | `finbert_mean` | media de P(positivo) − P(negativo) de FinBERT sobre los títulos deduplicados de la barra | sin títulos |
| 6 | `no_title` | 1 si la barra no tiene ningún título; 0 si tiene | — (indicador) |
| 7 | `coverage_ok` | sección 2.5 | — (indicador) |

- **Escalado:** las columnas 1 a 5 se estandarizan con el escalador de siempre, ajustado
  **solo con los valores no enmascarados del entrenamiento del fold**. Un valor
  enmascarado entra como 0 después de escalar, es decir, la media del entrenamiento.
  Los indicadores 6 y 7 pasan sin escalar, como las columnas de cuenta.
- **El log en `news_share`** existe porque la fracción tiene cola larga (días de
  noticia grande) y el escalador es lineal. El factor 10⁴ solo evita que el log opere
  cerca de cero; no cambia el orden.
- **Cambios contra el plan.** `tone_delta7` se calcula sobre el tono de GDELT y no
  sobre FinBERT: FinBERT no existe antes de 2019-09, y un delta sobre valores
  enmascarados no está definido en la mayor parte de los primeros folds. Se agrega
  `no_title`, decidido por Kevin el 2026-10-23, para que "sin títulos" no se confunda
  con "títulos neutros". El plan tenía seis columnas; con esa decisión son siete.

### 3.1 FinBERT

Decidido por Kevin (D2 del plan, 2026-10-20, y 2026-10-23):

- **Modelo:** `ProsusAI/finbert`. BERT base preentrenado con texto general anterior a
  2019 y ajustado con noticias financieras de antes de la existencia de los ETF de BTC,
  FTX o los halvings del período de test. No sabe qué le pasó a BTC entre 2020 y 2026,
  así que cumple el principio 1. **Ningún LLM ni modelo preentrenado con texto
  posterior a 2017 puntúa el histórico** (premisa 1 del plan: lookahead paramétrico).
- **Pin:** revisión exacta del repositorio del modelo y SHA256 de cada archivo de
  pesos y del tokenizer, en el manifest que versiona N2b. El sello de N8 los registra.
- **Mismo modelo en backtest y en vivo**, con los mismos pesos y el mismo
  preprocesamiento. Nada de puntuar en vivo con otro modelo "porque es mejor".
- **Solo donde hay título.** Antes de ~2019-09-23 no hay ninguno: `finbert_mean`
  queda enmascarada y `no_title` = 1. Se puntúa el título, nunca el texto del artículo,
  que no existe en el GKG.
- **Inferencia determinista** (modo evaluación, sin dropout), en CPU, por lotes, con
  escritura incremental (append) como exige la persistencia de procesos largos.

**Consecuencia sobre los folds.** Los tests tienen títulos en todas sus barras, pero
los entrenamientos de los primeros folds no:

| Fold | Barras de entrenamiento con título |
|---|---|
| 0 | 234 de 1 000 |
| 1 | 434 |
| 2 | 634 |
| 3 | 834 |
| 4 a 10 | 1 000 |

En los folds 0 a 3, B aprende `finbert_mean` con menos datos de los que la usa en el
test. Y `no_title` vale 1 en la primera parte de esos entrenamientos y 0 en todo el
test: funciona como una variable de época, con la que B podría aprender un
comportamiento "de antes de 2019" que en el test nunca se activa. No es lookahead (es
pasado y está en la observación), pero se declara.

## 4. Los tres brazos

| Brazo | Observación | Para qué |
|---|---|---|
| **A** | Las 15 columnas de `STUDY_OBSERVATION` | La referencia |
| **B** | A + las 7 columnas de la sección 3 en la barra `t` | La hipótesis |
| **Placebo** | A + las mismas 7 columnas de la barra `t − 364` | Separa "más columnas" de "noticias" |

**Placebo con 364 días, no 365.** El plan decía 365. Las noticias tienen estacionalidad
semanal fuerte (los fines de semana hay muchas menos), y 365 días corre el día de la
semana en uno. Con 364 (52 semanas exactas) el placebo conserva el día de la semana, que
es información de calendario y no de noticias, y entonces B contra placebo mide el
contenido de las noticias más allá del calendario. Con 365, un B que solo aprendiera el
patrón semanal le ganaría al placebo sin que las noticias aporten nada.

El placebo desplaza el bloque entero de siete columnas, indicadores incluidos, con el
escalador ajustado sobre sus propios valores de entrenamiento. Usa solo pasado, así que
no tiene lookahead. La descarga arranca en 2016-08-01 para que la barra 0 tenga placebo.

**Asimetría declarada:** el placebo no tiene títulos hasta 2020-09-21 (364 días después
de 2019-09-23). En el test del fold 0 y en los entrenamientos de los folds 0 a 4 tiene
más `finbert_mean` enmascarada que B. En esos folds la forma del placebo no es idéntica
a la de B.

## 5. Puerta sobre fixtures antes de datos reales (N6)

Antes de tocar datos reales, B tiene que demostrar que el pipeline le hace llegar una
señal exógena. El fixture de N6 agrega un nivel al protocolo con un canal exógeno de
óptimo conocido, con **N = 10 caminos × las 10 semillas** (principio 5). Cambia el
invariante de `data/CLAUDE.md` ("la señal va embebida en el precio") solo para este
nivel, y N6 lo documenta ahí.

**5a. Control positivo.** Una de las siete columnas (`tone_mean`) predice `r_{t+1}`
con la misma relación señal/ruido que el nivel 1 con `β` = 0,3, donde A, con la señal
en el precio, capturó 0,908. Las otras seis son ruido con la autocorrelación de su
contraparte real. Sin costos, como el nivel 1.

- **Criterio:** mediana entre caminos del `capture` de B ≥ 0,5 (el umbral del nivel 1
  en su SNR más alto), y diferencia pareada B − A de `capture` con `t` entre caminos
  ≥ 2,262 (N = 10).
- **Si falla:** no se corre sobre datos reales. Un NO APORTA sobre BTC no se podría
  leer: no se sabría si las noticias no informan o si B no puede usar una columna
  exógena. Se escribe un ADR nuevo.

**5b. Control negativo.** El fixture 4b (Heston con drift detectable, sin costos) más
siete columnas de ruido con la autocorrelación de sus contrapartes reales. No predicen
nada.

- **Se mide y no bloquea:** los tres criterios de 4b para B, y la diferencia pareada
  B − A del exceso sobre estar siempre invertido, con su `t` entre caminos.
- **Si B rota con el ruido** (dispersión de la acción > 0,15), es un hallazgo: siete
  entradas de ruido alcanzan para que la configuración del estudio vuelva a rotar. No
  se recalibra B (ADR 0011 §2). Se corre igual sobre datos reales, donde el placebo es
  justamente el brazo que mide ese efecto.

**Re-diagnóstico con las noticias como grupo propio.** Con R3, fijar los técnicos
lleva la dispersión de 0,109 a 0,022, y los demás grupos casi no la mueven
(`results/confirmacion/summary.json`). Con B hay un quinto grupo, `noticias` (las siete
columnas), y el diagnóstico de #57 se corre con los cinco grupos:

- **Sobre 5a y 5b**, con las políticas de B de la puerta. En 5a fijar las noticias
  tiene que hundir el `capture`: confirma que el diagnóstico detecta el uso de una
  columna. En 5b dice si B usa el ruido exógeno para mover la acción.
- **Sobre datos reales**, después del veredicto, en las barras de test de cada fold.
  Es descriptivo y no decide. Regla de lectura fijada aquí: si fijar las noticias
  cambia la acción en menos de 0,02 de media absoluta, el reporte dice "B no usa las
  noticias", y un NO APORTA se lee como "no las usó", no como "las usó y no sirvieron".

## 6. Criterio principal — **K3**

Por semilla `s`, con `X(s,q)` definido como en el ADR 0006 §2 (suma sobre folds del
exceso log contra buy-and-hold, neto de costos) y `q ∈ {mark, liquidation}`:

```
D_A(s,q) = X_B(s,q) − X_A(s,q)
D_P(s,q) = X_B(s,q) − X_placebo(s,q)
```

Buy-and-hold se cancela: `D` es la diferencia de crecimiento log entre los dos agentes
sobre la trayectoria fuera de muestra concatenada. El pareo por semilla es una
convención (la misma semilla no da el mismo ruido con observaciones de distinta
dimensión); se reporta también la distribución sin parear.

Para cada semilla, el **DSR** de la serie diaria `B mark − A mark` y de la serie
`B mark − placebo mark` sobre las 2 200 barras de test, con `n_trials` = 10 y la
varianza del Sharpe entre semillas, como en el ADR 0006 §2. El número honesto de
configuraciones de B evaluadas sobre estos datos es 1; el reporte muestra también el
DSR con `n_trials` = 1, sin que decida.

**APORTA (CONSISTENTE)** si y solo si se cumplen las seis:

1. `mediana_s D_A(s, mark) > 0`
2. `mediana_s D_A(s, liquidation) > 0`
3. `mediana_s D_P(s, mark) > 0`
4. `mediana_s D_P(s, liquidation) > 0`
5. `mediana_s DSR(B − A) ≥ 0,95`
6. `mediana_s DSR(B − placebo) ≥ 0,95`

En cualquier otro caso, **NO APORTA**. Para leer un NO APORTA, el reporte le pone una
de tres etiquetas, que no cambian el veredicto:

- **efecto de forma:** se cumplen 1, 2 y 5 pero falla alguna de 3, 4 o 6. B le gana a
  A pero no al placebo: lo que ayudó fue tener más columnas, no las noticias.
- **B peor:** `D_A` con mediana negativa en las dos series.
- **plano:** cualquier otro caso.

**K3 es la condición 6.** El borrador del plan pedía el DSR solo contra A. Lo recomendado
es pedirlo también contra el placebo: la pregunta del estudio es si el *contenido* de
las noticias aporta, y el contraste que lo aísla es contra el placebo. La alternativa
(solo las condiciones 1 a 5) es más fácil de pasar, y un APORTA podría venir de una
diferencia contra el placebo que no se separa del ruido.

**No se usa ningún test que suponga independencia entre folds** (ADR 0006 §3). Los
resultados por fold muestran dónde pasó algo, no son observaciones de un contraste.

## 7. El Agente A sobre datos reales — **K1**

La comparación necesita a A entrenado con la configuración del estudio. C3 corrió con
60 000 timesteps y el sizer anterior a #20 (premisa 3 del plan), así que sus registros
no sirven como el A de la comparación. Pero volver a correr A sobre los mismos 2 200
días de test es un **segundo toque del test para el Agente A**: ya se sabe que A con
60 000 timesteps no supera a buy-and-hold ahí. R3 se eligió solo sobre el fixture 4b,
sin mirar datos reales, así que no es una configuración elegida contra el test; pero
sí es la segunda que se evalúa sobre él.

**Recomendada: A, B y placebo en la misma corrida sellada**, mismo commit, mismo sello,
y el veredicto de A contra buy-and-hold (las tres condiciones del ADR 0006 §2) se
reporta como **secundario**, rotulado "segundo toque del test para el Agente A". Su DSR
decide con `n_trials` = 10, como en C3, y se muestra con `n_trials` = 2 (C3 y R3), que
es el número honesto. C3 queda sellado como registro y su `WalkForwardConfig` no se
toca (`agents/CLAUDE.md`): N8 arma su configuración desde `agents.study`.

Alternativas consideradas:

- **Misma corrida, pero A con un veredicto propio y `n_trials` = 1**, como si R3 fuera
  una primera configuración. Da un titular más limpio sobre A, pero ignora que esos
  días ya se miraron con A, y es la lectura optimista.
- **No volver a correr A** y comparar B contra los registros de C3. Ahorra 110
  entrenamientos (~3 h), pero la diferencia mediría R3 contra 60 000 timesteps y el
  sizer nuevo contra el viejo, no las noticias. Invalida la comparación.

## 8. Qué se reporta

Para los tres brazos, con la distribución completa sobre las 10 semillas (mediana, p25,
p75, min, max), por fold y agregado, y nunca la mejor semilla:

- `X` en `mark` y `liquidation` contra buy-and-hold, las seis cantidades del criterio
  y las tres etiquetas.
- El secundario de exposición igualada del ADR 0006 §2, la sensibilidad de solo
  comisión (ADR 0006 §1.1) y los rechazos por `MIN_NOTIONAL`.
- La σ de la política, al lado de la comparación (~1,0 con R3; `agents/CLAUDE.md`).
- El re-diagnóstico con cinco grupos (sección 5) y su lectura.
- Para B y el placebo: fracción de barras de test con `coverage_ok` = 0 y con
  `no_title` = 1, y el resultado del fold 9 con y sin los 17 días del apagón de 2025,
  este último como secundario que no decide.
- Las limitaciones de la sección 11, en el reporte y no en una nota al pie.

Al repositorio van `REPORTE.md`, el resumen y el manifest con el SHA256 de cada
artefacto completo; `records.jsonl` va al Release, como en C3.

## 9. Sellado y adendas

1. Se sella al mergearse: N8 registra el SHA256 de este archivo como constante y no
   corre si el del disco no coincide.
2. **Antes del sello de N8** este ADR solo se cambia con una adenda en un PR que diga
   qué cambia y por qué, y el motivo tiene que venir de ingeniería o de los fixtures
   (N3 a N6). Nunca de un resultado de B o del placebo sobre datos reales, que hasta
   ese momento no existen. Cada adenda resella.
3. El sello de N8 (`SEAL.json`) se escribe **antes** del primer entrenamiento e
   incluye: SHA256 de este ADR, de `BTCUSDT-1d.csv`, del manifest del dataset de
   noticias de N3 y de los pesos de FinBERT de N2b; el hash de la configuración; el
   commit; la latencia usada y el p99 medido en vivo a esa fecha. El runner aborta si
   el sello ya existe.
4. **El test se toca una vez.** Después del sello no hay cambios: una feature más, otra
   latencia u otro modelo de sentimiento son otra configuración, que cuenta en
   `n_trials` y se reporta junto a esta, nunca en su lugar.

## 10. Cómputo

Estimaciones con lo medido en el PC de Kevin (~2,7 ms por paso, 3 procesos):

| Paso | Entrenamientos | Pared aproximada |
|---|---|---|
| Puntuar títulos con FinBERT (N2b), ~2 M antes de deduplicar | — | horas en CPU; N2b lo mide antes de comprometerse |
| Puerta 5a + 5b (N6), A y B, 10 × 10 por nivel | 400 | ~9 h |
| Corrida N8: 3 brazos × 11 folds × 10 semillas | 330 | ~10 h |

La corrida de N8 tiene que ser reanudable por unidad (brazo, fold, semilla), como la
confirmación de #62, sin que reanudar sea un segundo toque: el sello se escribe una vez
y las unidades terminadas no se reentrenan.

## 11. Limitaciones

1. **N = 1 camino.** CONSISTENTE es el máximo; no se puede estimar la varianza de
   mercado entre ventanas.
2. **Barras diarias**: no mide la reacción intradía a noticias.
3. **GDELT es prensa general**, con deriva de composición en nueve años (fuentes que
   entran y salen). `news_share` la corrige en parte.
4. **`pub_time` es la hora de detección de GDELT**, no la de publicación original.
5. **Sin títulos antes de 2019-09-23**: FinBERT falta en el 23 % al 77 % de los
   entrenamientos de los folds 0 a 3, y `no_title` actúa como variable de época
   (sección 3.1).
6. **Filas sin deduplicar** en `news_share` y en el tono de GDELT: un artículo
   sindicado en diez dominios cuenta diez veces en la atención y pesa diez veces en
   el tono medio. Se elige así para que la feature signifique lo mismo antes y
   después de 2019-09 (sección 2.4).
7. **Apagones**: 17 días sin noticias dentro del test del fold 9.
8. **Red de 32×32 para 22 entradas**: puede limitar a B más que a A. Se mide, no se
   corrige (ADR 0011 §2).
9. **σ de la política ~1,0**: la evaluación es determinista y el par 4a-4b descarta
   "compra por defecto", pero la exploración no se apagó.
10. **Segundo toque del test para A** (sección 7).
11. **Costos supuestos**, no medidos; la sombra de la Etapa 3.5 los calibra después.
12. **El placebo no es idéntico en forma a B** en los primeros folds (sección 4).

## 12. Lo que este ADR no cambia

- Los ADR 0006 (C3 queda sellado), 0007 (pre-registro de la sombra), 0009 (Agentes A
  congelados de la sombra, 60 000 timesteps, sin tocar dentro de su ventana), 0010 y
  0011.
- `STUDY_PPO`, `STUDY_OBSERVATION` ni ningún criterio o umbral del protocolo.
- La regla de N1: la latencia se suma al rótulo y nunca se resta al cierre.
