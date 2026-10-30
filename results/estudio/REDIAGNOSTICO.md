# Re-diagnóstico de B sobre datos reales (N8e, ADR 0012 §5)

Descriptivo: no cambia el veredicto K3 de N9 (**NO APORTA, plano**). Corre sobre las
22 políticas de B guardadas por la corrida sellada (2 semillas × 11 folds,
`policies-n9.zip` del Release `n9-resultados`, SHA256 `4e7f3787…2a78a`), en las barras
de test de cada fold, sin reentrenar. Reproducir:

```
PYTHONPATH=src uv run --group rl python -m agents.cli study diagnose --policies <carpeta que contiene policies/> --out results/estudio
```

Resultado completo en `rediagnosis.json` (por grupo, por fold y por política).

## Resultado

**Fijar las noticias en su media cambia la acción 0,0106 de media absoluta**
(mediana entre 22 políticas; p25 0,001, p75 0,017, máximo 0,049). Está bajo el umbral de
0,02 que fijó el ADR, así que la regla de lectura da: **"B no usa las noticias"**, y el
NO APORTA se lee como "no las usó", no como "las usó y no sirvieron".

| Grupo fijado | Cambio de la acción (mediana) | p25 | p75 | máx |
|---|---|---|---|---|
| técnicos | 0,105 | 0,011 | 0,119 | 0,206 |
| posición | 0,031 | 0,005 | 0,051 | 0,085 |
| **noticias** | **0,011** | 0,001 | 0,017 | 0,049 |
| retornos | 0,004 | 0,001 | 0,012 | 0,060 |
| resultado de órdenes | 0,003 | 0,001 | 0,006 | 0,017 |

La acción de B depende sobre todo de los técnicos, igual que en el re-diagnóstico de R3
(ADR 0012 §5). Las noticias pesan unas diez veces menos.

## Matices

- **5 de las 22 políticas son constantes** (dispersión de la acción 0: fold 8 y fold 9
  con las dos semillas, y fold 6 semilla 23). Ignoran todo por construcción. Sin ellas
  la mediana es 0,0125: sigue bajo 0,02.
- **Folds 0 a 3 (los más antiguos) reaccionan más:** mediana por fold 0,034 / 0,048 /
  0,014 / 0,022; en los folds 4 a 10 va de 0 a 0,012. Los folds 0, 1 y 3 pasan 0,02 y
  caen en "indicio débil, no concluyente" frente al piso de ruido de la puerta 5b
  (mediana 0,0252, máximo por política 0,100). Los folds 0 y 1 tienen las medianas más
  altas. No se probó por qué; una hipótesis sin verificar es que
  la cobertura de noticias es peor en esos años.
- **Reemplazar las noticias por las del placebo** (t − 364 d) en las mismas barras
  cambia la acción 0,017 de mediana, pero con cola larga: p75 0,090, máximo 0,181. Unas
  pocas políticas sí reaccionan a que las noticias sean otras, aunque fijarlas en su
  media casi no las mueva. Eso es sensibilidad a la entrada, no utilidad: no se
  puede decir que ese uso aporte retorno.
- **Contra el placebo como agente no se puede comparar:** la corrida solo guardó las
  políticas de B, no las del placebo. La comparación es el reemplazo de entradas de
  arriba.

## Limitaciones

- El cambio de acción mide sensibilidad de la política, no si la señal sirve. Un
  `<0,02` dice "no las usa"; un valor alto no diría "las usa bien".
- El piso de ruido (0,0252) viene de fixtures sintéticos (puerta 5b), con otra
  distribución de entradas que la real. Es la referencia disponible, no una equivalencia
  (ASSUMPTION declarado en `study_diagnosis.py`).
- La σ de las políticas es ~0,93 (N9): la política estocástica es casi aleatoria y la
  evaluación determinista es la que se diagnostica.
- N = 1 camino real: 22 políticas son 2 semillas × 11 ventanas, no 22 muestras
  independientes del mercado.

## Qué cambia en la conclusión

El veredicto no cambia, pero su lectura sí: con este agente (R3, MLP 32×32, ventana de
retornos 1) y estas siete columnas, **B no aprendió a usar las noticias**, así que el
resultado no permite concluir que las noticias de GDELT carezcan de alfa; permite
concluir que esta configuración no lo extrajo. Distinguir ambas cosas exigiría otro
estudio (otra arquitectura o features), fuera del alcance pre-registrado.
