# Stage 1: build — install Python and Node dependencies
FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

RUN pip install uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python aiohttp

# Stage 2: runtime — minimal image with venv + static assets
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/node_modules ./node_modules/

COPY server.py robot_ipython.py RemoteAPI.zip threejs_scene.html viewer.css *.glb ./
COPY js/ ./js/
COPY *.json ./

RUN echo "=== Files in /app ===" && ls -lh /app && \
    echo "=== Verifying Python ===" && \
    /app/.venv/bin/python --version && \
    echo "=== Verifying aiohttp import ===" && \
    /app/.venv/bin/python -c "import aiohttp; print('aiohttp', aiohttp.__version__)" && \
    echo "=== Verifying server.py syntax ===" && \
    /app/.venv/bin/python -m py_compile server.py && echo "server.py OK" && \
    echo "=== Verifying config files ===" && \
    for f in *.json; do \
        /app/.venv/bin/python -c "import json,sys; json.load(open('$f')); print('$f OK')" || \
        { echo "INVALID JSON: $f"; exit 1; }; \
    done && \
    echo "=== Runtime stage verification complete ==="

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd -u 1000 -M -s /sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
