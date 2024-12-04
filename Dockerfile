FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get upgrade -y

# Install apt packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y unzip psmisc wget bc jq htop curl git git-lfs nano ssh gcc tmux -y --no-install-recommends && rm -rf /var/cache/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Visual Studio Code (for interactive tunnelling)
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz

# copy above docker folder to /plancraft and set working directory
COPY .. /plancraft
WORKDIR /plancraft

CMD [ "sleep", "infinity" ]