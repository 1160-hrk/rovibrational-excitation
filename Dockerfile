FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      git \
      curl \
      vim \
      sudo \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -G sudo devuser \
 && echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -e ".[dev,io,plot]" jupyter ipykernel

RUN mkdir -p /workspace/notebooks \
 && chown -R devuser:devuser /workspace

USER devuser

EXPOSE 8888

ENTRYPOINT ["tail", "-f", "/dev/null"]
