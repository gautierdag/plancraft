FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0

RUN conda init bash
RUN conda create -n main python=3.12 -y
RUN echo "conda activate main" >> ~/.bashrc
ENV PATH /opt/conda/envs/main/bin:$PATH

SHELL ["conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN apt update
RUN apt upgrade -y

RUN echo y | pip install wandb openai tinydb tqdm ollama --upgrade
RUN conda install mamba -y 
RUN mamba install -c conda-forge starship jupyterlab black -y

RUN mamba install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN mamba install -c conda-forge timm accelerate datasets transformers -y

RUN rm -rf /app/
ADD . /app/

ENTRYPOINT ["/bin/bash"]