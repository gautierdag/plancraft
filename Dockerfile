FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

RUN apt-get -y install git-lfs unzip psmisc wget git python3 python-is-python3 pip bc htop nano curl
RUN git lfs install 
RUN pip install -U pip

# Install Visual Studio Code (for interactive tunnelling)
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz

RUN rm -rf /app/
ADD . /app/
WORKDIR /app/

RUN pip install -r requirements.docker.txt

ENV CURL_CA_BUNDLE=""

# Set bash as the entrypoint to allow running arbitrary commands
ENTRYPOINT ["/bin/bash", "-c"]