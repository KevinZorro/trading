# Imagen de la sombra de costos y de su watchdog (ADR 0007, PR 6).
# Mismo codigo para los dos servicios; cambia solo el comando (ver docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# El proceso no arranca sin leer el ADR 0007 (fija su SHA256 en la cabecera de los
# datos): docs/ tiene que estar en la imagen.
COPY src ./src
COPY docs ./docs

ARG GIT_SHA=unknown
ENV SHADOW_GIT_SHA=$GIT_SHA

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin shadow
USER 10001

CMD ["python", "-m", "live.shadow"]
