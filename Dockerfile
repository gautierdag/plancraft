FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

SHELL ["/bin/bash", "-c"]

# RUN apt-get update  \
#     && apt-get install -y --no-install-recommends \
#     libgl1 \
#     libglib2.0-0

# RUN apt update
# RUN apt upgrade -y

RUN conda init bash
RUN conda create -n main python=3.12 -y
RUN echo "conda activate main" >> ~/.bashrc

# ENV PATH /opt/conda/envs/main/bin:$PATH

SHELL ["conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
RUN pip install wandb openai tinydb tqdm ollama --upgrade
# RUN pip install -c conda-forge timm accelerate datasets transformers --upgrade

# RUN rm -rf /app/
# ADD . /app/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "main", "python", "test_kube.py"]