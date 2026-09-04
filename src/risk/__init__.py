"""Gestion de riesgo, independiente del agente.

La capa vive fuera de la estrategia y se aplica en el mismo punto en simulacion
y en vivo: entre la decision y el envio al venue. Rechaza, nunca redimensiona.
"""

from risk.layer import RiskEvent, RiskLayer, RiskLimits, RiskRejectReason

__all__ = [
    "RiskEvent",
    "RiskLayer",
    "RiskLimits",
    "RiskRejectReason",
]
