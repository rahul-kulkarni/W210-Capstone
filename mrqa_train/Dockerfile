FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y \
	vim \
	rename \
	git \
	build-essential \
	libssl-dev \
	libffi-dev \
	python3.7 \
	python3.7-dev \
	python3-pip \
&& rm -rf /var/lib/apt/list/*


#RUN python3.7 -m pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.7 -m pip install torch torchvision
RUN python3.7 -m pip install allennlp==0.9.0 flask==1.0.2


WORKDIR /workspace
RUN git clone https://github.com/mrqa/MRQA-Shared-Task-2019.git

RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
