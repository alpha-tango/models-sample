FROM ubuntu:latest

# baseline setup

RUN apt-get update

RUN apt-get install -y \
  build-essential \
  curl \
  python3 \
  python3-dev \
  python3-pip

RUN pip3 install --upgrade pip

# custom start

ENV HOME /alpha-iota/models-prod
WORKDIR $HOME

# Python modules

ADD requirements.txt $HOME
RUN pip3 install -r $HOME/requirements.txt

# scripts

ADD scripts/*.py scripts/*.sh scripts/

# compile all python files as sanity check
RUN python3 -m compileall */*.py
