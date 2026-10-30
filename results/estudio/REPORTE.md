# N9: A vs B vs placebo sobre BTCUSDT diario (ADR 0012)

Corrida sellada en el PC de Kevin con el commit `d4da04c` (`SEAL.json`), ADR 0012
(SHA256 `5bd0b407…`), latencia de noticias 60 min, p99 en vivo medido 48,4 min (menor
que 60: el sello arrancó). Son 330 entrenamientos (3 brazos × 11 folds × 10 semillas,
R3 a 120 000 timesteps). **El test se tocó una sola
vez.** Los registros completos (`records*.jsonl`) se quedan fuera del repo; el
`MANIFEST.json` lleva el SHA256 de cada uno y de `summary.json`.

**Limitación declarada: N = 1 camino real.** Diez semillas miden varianza de
entrenamiento, no de mercado. El máximo que este diseño permite es CONSISTENTE, y no
se alcanzó.

## Veredicto del criterio K3: **NO APORTA (plano)**

| Condición (mediana sobre 10 semillas) | Valor | Cumple |
|---|---|---|
| `D_A` mark = X(B) − X(A) > 0 | +0,185 (p25 −0,135, p75 +0,228; 7 de 10 semillas > 0) | sí |
| `D_A` liquidación > 0 | +0,197 | sí |
| `D_P` mark = X(B) − X(placebo) > 0 | +0,326 (p25 +0,058, p75 +0,513; 7 de 10 > 0) | sí |
| `D_P` liquidación > 0 | +0,329 | sí |
| DSR(B − A), `n_trials` = 10, ≥ 0,95 | **0,146** (máx 0,585) | **no** |
| DSR(B − placebo), `n_trials` = 10, ≥ 0,95 | **0,200** (máx 0,904) | **no** |

Informativo con `n_trials` = 1: DSR(B − A) mediana 0,573 (máx 0,932) y DSR(B − placebo)
mediana 0,804 (máx 0,999). Ni siquiera sin la corrección por pruebas múltiples cruza
0,95 en la mediana.

**Lectura honesta.** B rinde en promedio algo mejor que A y que el placebo, pero la
diferencia es chica frente a su dispersión entre semillas: `D_A` va de −0,55 a +0,67 y
tres de diez semillas dan negativo. Con DSR por debajo de 0,95 no se puede separar de
ruido. La etiqueta "plano" significa que no hay efecto distinguible de cero; no dice
que las noticias no sirvan. "Efecto de forma" (B gana a A pero no al placebo) no aplica:
B gana más contra el placebo que contra A, porque el placebo (noticias de 364 días
antes) empeora al agente (X mark mediana −1,76 contra −1,53 de A).

## Los tres brazos contra buy-and-hold

Los tres pierden por mucho. En el escenario principal, mediana sobre semillas:

| Brazo | `X` mark | `X` liq | `X` exposición igualada | dispersión de acción |
|---|---|---|---|---|
| A (solo precio) | −1,53 (−1,75 a −1,20) | −1,49 | −0,26 | 0,34 |
| B (con noticias) | −1,37 (−1,74 a −1,08) | −1,32 | +0,22 | 0,26 |
| placebo | −1,76 (−2,37 a −1,12) | −1,71 | −0,05 | 0,28 |

La diferencia de B con A casi no está en el nivel de exposición (todos pierden contra un
mercado que subió mucho) sino en el timing: con exposición igualada, B es el único brazo
con mediana positiva (+0,22; 8 de 10 semillas > 0). Ese es el indicio más favorable a
B y es secundario; no cambia el veredicto.

## Veredicto secundario del Agente A (K1)

**NO SUPERA a buy-and-hold** (DSR mediana 0,017 con `n_trials` = 10; 0,029 con
`n_trials` = 1). Es el segundo toque del test para A; el número honesto de configuraciones
evaluadas es 2 (C3 y R3). De 6 263 órdenes enviadas, 2 670 (42,6 %) se rechazaron por `MIN_NOTIONAL` y 2 190
por redondeo a cero; el rechazo queda en el log.

## Por fold (mediana de `E` mark sobre semillas)

| Fold | Test | A | B | placebo |
|---|---|---|---|---|
| 0 | 2020-05..2020-11 | −0,12 | −0,18 | −0,16 |
| 1 | 2020-11..2021-06 | −0,56 | −0,24 | −0,35 |
| 2 | 2021-06..2022-01 | +0,02 | +0,09 | −0,07 |
| 3 | 2022-01..2022-07 | +0,46 | +0,48 | +0,39 |
| 4 | 2022-07..2023-02 | +0,08 | +0,09 | +0,08 |
| 5 | 2023-02..2023-08 | −0,11 | −0,12 | −0,11 |
| 6 | 2023-08..2024-03 | −0,88 | −1,00 | −0,91 |
| 7 | 2024-03..2024-09 | −0,07 | −0,00 | −0,00 |
| 8 | 2024-09..2025-04 | −0,24 | −0,24 | −0,24 |
| 9 | 2025-04..2025-11 | −0,25 | −0,25 | −0,25 |
| 10 | 2025-11..2026-05 | +0,10 | +0,12 | +0,07 |

Las diferencias entre brazos se concentran en el fold 1 (B −0,24 contra A −0,56) y en
los folds con mucho movimiento; en la mitad de los folds los tres brazos son
indistinguibles. Eso es lo esperable de políticas casi idénticas con una señal débil.

## Extras de la sección 8

- **Cobertura de noticias en el test.** Barras con `coverage_ok` = 0: 4,4 % en B y 3,5 %
  en el placebo; sin título (`no_title`): 0,8 % en B y 5,9 % en el placebo. El fold 0 tiene
  27 % de barras sin cobertura en B (2020), el 2 un 10,5 %.
- **Fold 9 (apagón de GDELT de 2025).** 19 de 199 barras sin cobertura. La suma del
  exceso diario B − A es +0,026 de mediana con esas barras y con ellas excluidas
  (IQR −0,044 a +0,047 con ellas): el apagón no explica nada. ASSUMPTION heredado: suma
  del exceso diario, no X en log.
- **σ de la política.** A 0,93, B 0,92 (mediana): la σ es alta en los tres brazos, igual que en la
  confirmación de R3 (σ ~ 1,0, ya reportado).
  Por eso "casi sin efecto" no se debe leer como "no usa las noticias": no se midió
  todavía qué mueve la acción (N8e).
- **Políticas de B guardadas** para el re-diagnóstico: 2 semillas × 11 folds.

## Qué sigue

1. **N8e**: re-diagnóstico con cinco grupos (retornos, técnicos, posición, banderas,
   noticias) sobre las políticas guardadas de B, para ver si las noticias mueven la
   acción y cuánto, contra el piso de ruido de 0,025 (ver puerta).
2. Subir los registros completos como Release (`n9-resultados`) con el SHA256 del
   manifest. Lo hace Kevin con `gh release create`.
3. Etapa 6 (barrido de capital) solo tiene sentido si algún brazo supera a
   buy-and-hold; hoy ninguno lo hace.

## Reproducir

`SEAL.json` + el commit sellado + `python -m agents.cli study assemble --out results/estudio`
sobre los `records*.jsonl` (verificables contra `MANIFEST.json`).
