FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN printf 'channels:\n  - conda-forge\nchannel_priority: flexible\n' > /opt/conda/.condarc
RUN conda create -y -n bomberman python=3.11
ENV PATH /opt/conda/envs/bomberman/bin:$PATH
RUN conda install -y -n bomberman scipy numpy matplotlib numba
RUN conda install -y -n bomberman pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install -y -n bomberman pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD /bin/bash
