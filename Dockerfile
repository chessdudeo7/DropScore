# DropScore web service.
#
#   docker build -t dropscore .
#   docker run -p 8000:8000 dropscore
#
# Behind a reverse proxy, set FORWARDED_ALLOW_IPS to the proxy's address (or
# "*" if only the proxy can reach the container). The per-visitor job limit
# is keyed on the client address, and without it every visitor shares the
# proxy's.
FROM python:3.11-slim

# OpenCV's headless wheel brings its own shared libraries, and nothing it
# links is missing from the slim base. The libgl1 and libglib2.0-0 that an
# OpenCV image usually installs are for the GUI build; here they cost 216MB.

RUN useradd --create-home dropscore
WORKDIR /app

COPY pyproject.toml README.md ./
COPY dropscore ./dropscore
COPY web ./web
COPY docs ./docs
# Editable, so the service finds web/ and docs/ beside the package as it does
# in a checkout.
RUN pip install --no-cache-dir -e ".[service,pdf]"     && mkdir -p /home/dropscore/jobs     && chown dropscore /home/dropscore/jobs

USER dropscore
ENV PYTHONUNBUFFERED=1 \
    FORWARDED_ALLOW_IPS=127.0.0.1
EXPOSE 8000
VOLUME ["/home/dropscore/jobs"]

CMD ["dropscore", "serve", "--host", "0.0.0.0", "--port", "8000", "--workdir", "/home/dropscore/jobs"]
