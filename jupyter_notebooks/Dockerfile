FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN conda install -c conda-forge jupyterlab ipywidgets matplotlib 
RUN pip install fastai pytorch-pretrained-bert transformers statsmodels nltk seaborn tqdm contractions fastparquet scikit-learn

RUN apt-get update && apt-get install -y \
    vim \
    rename \
&& rm -rf /var/lib/apt/list/*

RUN echo 'alias jtail="tail -100 ~/jupiter.log"' >> ~/.bashrc
RUN echo 'alias jnote="nohup jupyter notebook --ip=0.0.0.0 --allow-root > ~/jupiter.log 2>&1 & sleep .5; jtail"' >> ~/.bashrc
RUN echo 'alias jtoken="grep \?token= ~/jupiter.log | tail -1"' >> ~/.bashrc

WORKDIR /workspace

