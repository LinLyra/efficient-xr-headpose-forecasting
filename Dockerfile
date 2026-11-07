# Interactive training/development Pod. The exact Runpod base keeps CUDA,
# PyTorch, SSH and web-terminal behavior in one tested image family.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEADPOSE_DATA_ROOT=/workspace/headpose-data \
    HEADPOSE_ARTIFACT_ROOT=/workspace/headpose-artifacts

# Keep code and Python packages outside /workspace because a network volume
# replaces that mount at runtime.
WORKDIR /opt/headpose
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps . pytest==8.3.4

COPY config ./config
COPY scripts ./scripts
COPY tests ./tests
COPY cloud ./cloud

# Do not override CMD: inherit /start.sh and its SSH/web-terminal services.
