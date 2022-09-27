# FROM ubuntu:focal
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV TZ=America/Guayaquil
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get upgrade -y

ADD ./requirements.txt /

WORKDIR /

# RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN apt install -y git

RUN apt install python3-pip -y

RUN pip install -r requirements.txt

EXPOSE 8888
EXPOSE 5000

WORKDIR /workspace

## this is just for development ##

ARG USER_ID=1000
ARG GROUP_ID=1001

RUN groupadd --system --gid ${GROUP_ID} marcelo && \
    useradd --system --uid ${USER_ID} --gid marcelo -m --shell /bin/bash marcelo

# CMD ["jupyter","notebook"]
