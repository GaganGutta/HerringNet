# Slim image for the hosted services: the folder watcher, the upload page,
# and the Datasette data viewer. It intentionally does NOT install the heavy
# ML dependencies (torch/ultralytics/opencv), only what these services need.

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install the light service dependencies first (better layer caching).
COPY deploy/requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt

# Copy the project and install it WITHOUT its dependencies (we already have
# the light subset above; this just makes the herringnet/cli packages importable).
COPY pyproject.toml README.md /app/
COPY herringnet /app/herringnet
COPY cli /app/cli
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1

# Default command is overridden per-service in docker-compose.yml.
CMD ["python", "-m", "cli.main", "db", "stats"]
