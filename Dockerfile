# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

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

# Install poetry with pinned version
RUN pip install git+https://github.com/python-poetry/poetry.git@master
ENV PATH="$PATH:$POETRY_HOME/bin"

USER $USERNAME