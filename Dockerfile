# Stage 1: build — install Node dependencies and the Python package
FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

# Install the lidarstudio package (aiohttp/numpy/plyfile) into a venv.
# No .git in the build context, so give setuptools_scm a version.
# The heavy generation extras (open3d, kiss-icp, torch/gsplat for trained
# splats) are intentionally left out of the image — generation runs on a
# workstation with the scan data.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LIDARSTUDIO=0.0.0
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python .

# Stage 2: runtime — minimal image with venv + static assets
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/node_modules ./node_modules/

# Front-end assets, served from the working directory
COPY threejs_scene.html viewer.css ./
COPY js/ ./js/

RUN echo "=== Verifying install ===" && \
    /app/.venv/bin/lidarstudio --version && \
    /app/.venv/bin/python -c "import lidarstudio.lidar_jobs; print('lidar_jobs OK')"

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd -u 1000 -M -s /sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

EXPOSE 8080

ENTRYPOINT ["lidarstudio"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
