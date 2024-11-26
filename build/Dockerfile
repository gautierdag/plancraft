FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get upgrade -y

# Install apt packages
COPY apt.txt apt.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN xargs -a apt.txt apt-get install -y --no-install-recommends && rm -rf /var/cache/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh

# Install Visual Studio Code (for interactive tunnelling)
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz

# copy above docker folder to /plancraft and set working directory
COPY ../.. /plancraft
WORKDIR /plancraft

CMD [ "sleep", "infinity" ]