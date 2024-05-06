FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y python3 python-is-python3 python3-pip python3-opencv git colmap build-essential cmake wget nvidia-cuda-toolkit && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /

RUN pip install --no-cache-dir -r requirements.txt

#install pytorch3d
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

WORKDIR /OnePose_Plus_Plus
