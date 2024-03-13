FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt upgrade -y

RUN apt-get -y install git-lfs unzip psmisc wget git python3 python-is-python3 pip bc htop nano 
RUN git lfs install 
RUN pip install -U pip


RUN rm -rf /app/
ADD . /app/
WORKDIR /app/

RUN pip install -r requirements.docker.txt

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# ENV HF_TOKEN=$HF_TOKEN
# ENV H4_TOKEN=$HF_TOKEN 
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV CURL_CA_BUNDLE=""


# 
RUN 

# Set bash as the entrypoint to allow running arbitrary commands
ENTRYPOINT ["/bin/bash", "-c"]
