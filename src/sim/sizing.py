"""Conversion de peso objetivo a orden. Vive aqui, no en el entorno.

Dimensionar una posicion es logica de ejecucion. Si el entorno Gymnasium la
implementara por su cuenta, habria dos formas de convertir una intencion en una
orden -la del entorno y la de las estrategias- y divergirian. El wrapper de la
Etapa 2 tiene que llamar a esto, igual que lo llamaria una estrategia directa.

Dos reglas que el proyecto ya fija en otros lados y que aca se respetan:

1. **No se pre-redondea a cero.** ``round_qty`` redondea hacia cero, asi que un
   delta chico se convertiria en ``qty=0`` y la orden desapareceria en silencio.
   Eso hace indistinguible "no quise operar" de "no pude", que es el
   anti-patron que el proyecto prohibe en los baselines. El delta se envia sin
   redondear y ``check_tradable`` lo rechaza con su motivo, que queda en el log.

2. **No se recorta contra el cash.** Si el delta no entra, el venue lo rechaza
   con ``INSUFFICIENT_CASH`` y el rechazo queda registrado. Recortarlo aca
   seria redimensionar en silencio.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim.engine import LONG_ONLY_ACTION_RANGE
from sim.orders import MarketOrder
from sim.view import AccountSnapshot, MarketView


@dataclass(frozen=True)
class TargetWeightSizer:
    """Traduce un peso objetivo en ``[0, 1]`` a la orden que lo alcanza.

    ``safety`` existe por la misma razon que en ``BuyAndHold``: la orden se
    dimensiona con ``close[t]`` pero se ejecuta al ``open[t+1]``, que puede ser
    mas alto, y la comision se cobra encima. Un peso de 1.0 sin margen se
    rechazaria por cash insuficiente casi siempre. Estar exactamente all-in
    exigiria conocer el precio de ejecucion antes de enviar la orden, es decir,
    lookahead.

    ASSUMPTION: un peso de 1.0 significa "lo mas invertido que se puede estar
    sin conocer el precio de ejecucion", que con el valor por defecto es el 98%
    del equity. La constante es configuracion y se serializa con la corrida, no
    un piso escondido en el codigo.

    ``deadband`` es 0.0 por defecto: cualquier delta distinto de cero se envia.
    Con capital bajo eso produce un rechazo por barra, que **es el efecto que el
    barrido de capital de la Etapa 6 quiere medir**, no ruido. Quien quiera una
    banda muerta la pone explicita y queda serializada con el resultado.
    """

    safety: float = 0.98
    deadband: float = 0.0
    tag: str = "target_weight"

    def __post_init__(self) -> None:
        if not 0.0 < self.safety <= 1.0:
            raise ValueError("safety debe estar en (0, 1]")
        if self.deadband < 0.0:
            raise ValueError("deadband no puede ser negativo")

    def order_for(
        self,
        target_weight: float,
        view: MarketView,
        account: AccountSnapshot,
    ) -> MarketOrder | None:
        """Orden que lleva la posicion al peso objetivo, o ``None`` si ya esta.

        Devuelve ``None`` **solo** cuando no hay nada que hacer: el delta cae
        dentro de la banda muerta. Nunca por creer que la orden se va a
        rechazar; eso lo decide el venue y queda en el log.
        """
        bajo, alto = LONG_ONLY_ACTION_RANGE
        if not bajo <= target_weight <= alto:
            raise ValueError(
                f"target_weight={target_weight!r} fuera de "
                f"{LONG_ONLY_ACTION_RANGE}. Recortar es responsabilidad de quien "
                "llama, y el recorte tiene que quedar registrado: un agente "
                "entrenado sobre un rango que alguien recorta en silencio "
                "aprende sobre un mundo que no existe."
            )
        price = view.close()
        if price <= 0:
            raise ValueError(f"precio no positivo en el sizing: {price!r}")

        objetivo_qty = target_weight * account.equity * self.safety / price
        delta = objetivo_qty - account.position

        if abs(delta) * price <= self.deadband * account.equity:
            return None
        return MarketOrder(qty=delta, tag=self.tag)

    def describe(self) -> dict[str, object]:
        return {"safety": self.safety, "deadband": self.deadband, "tag": self.tag}
