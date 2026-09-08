FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Install into an isolated prefix
RUN python3 -m pip install --break-system-packages --prefix=/opt/brass .


FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/brass /opt/brass
ENV PATH="/opt/brass/local/bin:${PATH}"
ENV PYTHONPATH="/opt/brass/local/lib/python3.12/dist-packages"
WORKDIR /work

CMD ["/bin/bash"]
