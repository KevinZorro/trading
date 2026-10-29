# Puerta del ADR 0012 §5 (N6c)

Corrida de Kevin el 2026-10-28/29: 400 entrenamientos (R3, 120 000 timesteps),
10 caminos x 10 semillas por brazo. Los JSON por unidad y las políticas guardadas
no se versionan; `MANIFEST.json` lleva el SHA256 de cada unidad.

## Veredicto: la puerta pasa, `blocks_real_data = false`

- **5a (control positivo) PASS.** Capture de B, mediana entre caminos 0.925
  (umbral 0.50). A ciego: -0.067. Diferencia pareada +0.9835, t = +14.84 contra
  2.262 (N = 10 caminos).
- **5b (control negativo) MEASURED.** B pasa los criterios del 4b con 7 columnas
  de ruido: invertido 0.94, dispersión de la acción 0.119 (< 0.15), rotación
  10.5/año. Exceso pareado B - A: -0.0268, t = -0.77, no distinguible de cero.

## Re-diagnóstico con el grupo `noticias` (`rediagnosis.json`, 20 políticas por nivel)

- **5a:** fijar las noticias cambia la acción 0.340 (mediana); las 20 políticas
  pierden `capture` (mediana 0.916 -> 0.006).
- **5b:** fijar las noticias cambia la acción 0.0252 (mediana), apenas sobre el
  umbral de 0.02, con columnas de puro ruido. Por política va de 0 a 0.100:
  unas ignoran las columnas y otras reaccionan al ruido.

## Limitaciones y lectura

- ASSUMPTION: la regla "< 0.02 => B no usa las noticias" sirve para afirmar que
  no las usa, no para afirmar que las usa: el piso de ruido medido es 0.025
  (hasta 0.100 en algunas políticas). Sobre datos reales, un cambio cercano a ese
  piso no es evidencia de uso. Hay que leerlo contra este piso y contra el placebo
  a 364 días del ADR 0012. No cambia el veredicto de la puerta.
- En 5a el precio solo casi no lleva señal (R² 0.0024), así que A ciego tiene poco
  que capturar; el t pareado mide sobre todo que B lee `tone_mean`.
- Fixtures sintéticos: N = 10 caminos independientes, con la varianza de
  entrenamiento (10 semillas) separada de la de mercado.
