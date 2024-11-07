FROM python:3.12.7

WORKDIR /app

COPY . /app

RUN pip install /app/install_files/*

ENV POPPLER_PATH /usr/local/bin
