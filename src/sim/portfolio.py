"""Contabilidad: un libro de cash y un diccionario de posiciones.

La forma es de portafolio desde el primer dia aunque la Etapa 1 opere un solo
simbolo. Convertir ``self.position: float`` en multi-activo es una reescritura;
convertir ``self.positions: dict[str, float]`` es agregar un bucle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tolerancia de la verificacion contable. Por debajo de esto la diferencia es
# error de redondeo en coma flotante; por encima, es un bug de contabilidad.
ACCOUNTING_TOL = 1e-6


class AccountingError(AssertionError):
    """La identidad ``equity == cash + sum(posiciones a precio de marca)`` fallo."""


@dataclass
class Ledger:
    """Libro mutable de la simulacion. Solo el motor lo toca."""

    cash: float
    positions: dict[str, float] = field(default_factory=dict)

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def apply_fill(
        self, symbol: str, qty_filled: float, fill_price: float, commission: float
    ) -> None:
        """Asienta una ejecucion.

        El spread y el slippage ya estan dentro de ``fill_price``; la comision se
        cobra aparte. El cash se mueve exactamente por el efectivo intercambiado.
        """
        if commission < 0:
            raise ValueError("la comision no puede ser negativa")
        self.cash -= qty_filled * fill_price + commission
        new_position = self.position(symbol) + qty_filled
        if new_position == 0.0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_position

    def accrue_interest(self, rate_per_bar: float) -> float:
        """Devenga interes sobre el cash ocioso y devuelve lo devengado.

        ASSUMPTION: solo se remunera el cash positivo y no se modela costo de
        financiamiento, coherente con ``allow_short=False`` y sin apalancamiento.
        Con capital bajo es irrelevante, pero en el barrido de capital y con
        tasas al 5% cambia de forma material la comparacion contra
        buy-and-hold: una estrategia que esta fuera del mercado la mitad del
        tiempo si cobraria ese interes en la realidad.
        """
        if rate_per_bar == 0.0 or self.cash <= 0.0:
            return 0.0
        interest = self.cash * rate_per_bar
        self.cash += interest
        return interest

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Equity marcado a precio de cierre."""
        equity = self.cash
        for symbol, qty in self.positions.items():
            equity += qty * prices[symbol]
        return equity

    def check_identity(self, equity: float, prices: dict[str, float]) -> None:
        """Verifica la identidad contable. Se llama en cada barra, no solo en tests."""
        expected = self.mark_to_market(prices)
        if abs(expected - equity) > ACCOUNTING_TOL * max(1.0, abs(expected)):
            raise AccountingError(
                f"equity={equity!r} != cash + posiciones={expected!r} "
                f"(cash={self.cash!r}, posiciones={self.positions!r})"
            )
