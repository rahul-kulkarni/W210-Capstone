FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN conda install -c conda-forge jupyterlab ipywidgets matplotlib 
RUN pip install fastai pytorch-pretrained-bert transformers statsmodels

RUN apt-get update && apt-get install -y \
    vim \
    rename \
&& rm -rf /var/lib/apt/list/*

RUN echo 'alias jupnote="nohup jupyter notebook --ip=0.0.0.0 --allow-root &"' >> ~/.bashrc

WORKDIR /workspace
