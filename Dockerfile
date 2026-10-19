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

# Grupo `rl` (stable-baselines3 + torch) para cargar el Agente A congelado (ADR 0009).
# uv.lock resuelve torch con las ruedas CUDA (varios GB de nvidia-*), que en un VPS sin
# GPU no sirven. Se instala todo lo bloqueado salvo torch y sus dependencias CUDA, y
# torch con la MISMA version del lock desde el indice solo CPU de PyTorch. Asi uv.lock
# y pyproject.toml no cambian (CI y el entorno de desarrollo siguen igual).
# ASSUMPTION: la rueda +cpu es numericamente equivalente a la CUDA en CPU; los pesos
# se entrenan en CPU (scripts/congelar_agente_a.py).
RUN uv export --frozen --no-dev --group rl --no-hashes --no-emit-project \
        | grep -viE '^(torch|triton|nvidia-|cuda-)' > /tmp/rl.txt \
    && TORCH=$(uv export --frozen --no-dev --group rl --no-hashes --no-emit-project \
        | grep -iE '^torch==' | cut -d' ' -f1) \
    && uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/rl.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps \
        --index-url https://download.pytorch.org/whl/cpu "${TORCH}+cpu" \
    && rm /tmp/rl.txt \
    && python -c "import torch, stable_baselines3; assert not torch.version.cuda, torch.version.cuda"

# El proceso no arranca sin leer el ADR 0007 (fija su SHA256 en la cabecera de los
# datos): docs/ tiene que estar en la imagen.
COPY src ./src
COPY docs ./docs

ARG GIT_SHA=unknown
ENV SHADOW_GIT_SHA=$GIT_SHA

# --no-user-group: en Debian el grupo `shadow` ya existe (gid 42) y useradd falla.
RUN useradd --uid 10001 --no-user-group --no-create-home --shell /usr/sbin/nologin shadow
USER 10001

CMD ["python", "-m", "live.shadow"]
