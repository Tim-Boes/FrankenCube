# DOCKERFILE FOR BIRKA

FROM ubuntu:22.04

# SPECIFY YOUR WORKING DIR

WORKDIR /home/

# COPY ALL THE DATA NECCESSARY FROM YOUR MACHINE

COPY ./ .

# RUN ALL THE COMMANDS NEEDED AT THE START

RUN apt update && \
    apt install -y git-all && \
    git clone https://github.com/Tim-Boes/FrankenCube.git
