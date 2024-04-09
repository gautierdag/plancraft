FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

# Install apt packages
COPY apt.txt apt.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN xargs -a apt.txt apt-get install -y --no-install-recommends && rm -rf /var/cache/*

# # Install VirtualGL
ARG VIRTUALGL_VERSION=3.0.2
RUN curl -fsSL -O https://github.com/VirtualGL/virtualgl/releases/download/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb &&\
    apt-get update && apt-get install -y --no-install-recommends ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    chmod u+s /usr/lib/libvglfaker.so && \
    chmod u+s /usr/lib/libdlfaker.so


# Customized for your own application. MineRL (https://github.com/minerllabs/minerl) is installed for testing purposes.
# java jdk 1.8
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
# MineRL
RUN pip3 install git+https://github.com/minerllabs/minerl --no-dependencies

COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh

ENTRYPOINT ["/etc/entrypoint.sh"]
CMD [ "sleep", "infinity" ]