FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

# Install apt packages
COPY apt.txt apt.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN xargs -a apt.txt apt-get install -y --no-install-recommends && rm -rf /var/cache/*

# Install VirtualGL
ARG VIRTUALGL_VERSION=3.0.2
RUN curl -fsSL -O https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb &&\
    apt-get update && apt-get install -y --no-install-recommends ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    chmod u+s /usr/lib/libvglfaker.so && \
    chmod u+s /usr/lib/libdlfaker.so

# java jdk 1.8 needed for minecraft
RUN apt update -y && apt install -y software-properties-common && \
    add-apt-repository ppa:openjdk-r/ppa && apt update -y && \
    apt install -y openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

ENV DISPLAY :0
ENV VGL_REFRESHRATE 60
ENV VGL_ISACTIVE 1
ENV VGL_DISPLAY egl
ENV VGL_WM 1

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# CLONE MINERL
ARG CACHEBUST=1
RUN git clone https://github.com/gautierdag/minerl.git /minerl
RUN pip3 install -e /minerl --no-dependencies
# For some reason needs to be installed separately
RUN pip3 install flash-attn

COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh

# Install Visual Studio Code (for interactive tunnelling)
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz

# copy above docker folder to /plancraft and set working directory
COPY .. /plancraft
WORKDIR /plancraft

ENTRYPOINT ["/etc/entrypoint.sh"]
CMD [ "sleep", "infinity" ]