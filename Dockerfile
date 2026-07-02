# Stage 1: build — install Python and Node dependencies
FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

# aiohttp serves the app; numpy/plyfile power the editing endpoints.
# The heavy generation deps (open3d, kiss-icp, torch/gsplat for trained
# splats) are intentionally left out of the image — generation runs on a
# workstation with the scan data.
RUN pip install uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python aiohttp numpy plyfile

# Stage 2: runtime — minimal image with venv + static assets
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/node_modules ./node_modules/

COPY server.py lidar_jobs.py edit_ops.py cloud_ops.py splat_io.py \
     process_pointcloud.py process_splat.py \
     threejs_scene.html viewer.css package.json ./
COPY js/ ./js/

RUN echo "=== Files in /app ===" && ls -lh /app && \
    echo "=== Verifying Python ===" && \
    /app/.venv/bin/python --version && \
    echo "=== Verifying aiohttp import ===" && \
    /app/.venv/bin/python -c "import aiohttp; print('aiohttp', aiohttp.__version__)" && \
    echo "=== Verifying server syntax ===" && \
    /app/.venv/bin/python -m py_compile server.py lidar_jobs.py && echo "server OK" && \
    echo "=== Runtime stage verification complete ==="

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd -u 1000 -M -s /sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
