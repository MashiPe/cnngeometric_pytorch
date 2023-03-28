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

RUN apt install -y git git-lfs

RUN apt install -y apt-utils unzip \
    tar \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    ocl-icd-opencl-dev \
    ;

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

ENV NVIDIA_VISIBLE_DEVICE all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

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

COPY  . /app

WORKDIR /app

CMD [ "python3","./api/main.py" ]