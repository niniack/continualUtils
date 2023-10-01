# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install required tools and libraries, and clean up to reduce layer size
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y libstdc++6 \
    && wget "https://github.com/aristocratos/btop/releases/latest/download/btop-x86_64-linux-musl.tbz" \
    && tar xvjf btop-x86_64-linux-musl.tbz -C /usr/local/bin \
    && rm btop-x86_64-linux-musl.tbz \
    && apt-get remove --purge -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Poetry variables
# The system site packages are important because we are using docker to give 
# us torch!
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=1

WORKDIR /continualUtils

# Install poetry with pinned version
RUN pip install git+https://github.com/python-poetry/poetry.git@master
ENV PATH="$PATH:$POETRY_HOME/bin"

# Run the Python script to update pyproject.toml
RUN python update_pyproject.py