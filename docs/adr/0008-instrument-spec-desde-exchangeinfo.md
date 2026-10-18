# ADR 0008 — `InstrumentSpec` desde `exchangeInfo`, no desde los defaults

Estado: aceptado · Fecha: 2026-10-17 · Etapa 3.5

## Contexto

`binance_spot_spec` asume `min_notional = 10` USDT. El `exchangeInfo` real de BTCUSDT
grabado en el PR 2 (`tests/live/respuestas/exchange_info.json`, 2026-10-17, SHA256
`3523e03a…ebbb` en su `manifest.json`) dice **5** USDT. Con 100 USDT de capital el
umbral de rechazo de un cambio de exposición pasa de 10 % a 5 %. No hay historia del
filtro: es un solo snapshot, y este ADR no afirma nada sobre el pasado.

C3 corrió con 10 y se conserva así (adenda en `results/c3/REPORTE.md`).

## Decisión

1. **Default histórico intocable.** `binance_spot_spec().min_notional` sigue en 10.0:
   es el valor con que corrió C3 y cambiarlo rompe su reproducibilidad. Un test lo
   fija.
2. **Desde la Etapa 3.5**, todo lo que corra contra el venue o se entrene para operar
   en él toma su `InstrumentSpec` de `MarketData.instrument()` o de un `exchangeInfo`
   grabado con su SHA256, **nunca** de los defaults de `binance_spot_spec`. Alcanza:
   - el Agente A congelado del ADR 0007 §5.4,
   - `ShadowVenue` y el ledger sombra de 100 USDT (PR 5).
3. **El snapshot se serializa junto a la corrida**: su contenido o su SHA256 más la
   fecha de recepción, en el mismo registro que la config y los pesos.

## Motivo

Entrenar con 10 y operar con 5 hace que `last_order_rejected` vea en sombra un venue
distinto del de entrenamiento: órdenes que el agente aprendió a esperar rechazadas
pasan, y la observación queda fuera de distribución justo donde el principio del
proyecto dice que un agente RL puede decidir cualquier cosa.

## Consecuencias

- Los entrenamientos y corridas sobre datos reales posteriores no son comparables con
  C3 también por este factor, además del sizer (ADR 0006 / REPORTE).
- Los fixtures sintéticos siguen con `min_notional = 0` (ADR 0002): no son venue.
- ADR 0006 y 0007 no se editan: el 0006 está sellado por SHA256 y el 0007 es
  pre-registro. Este ADR es el puntero.
