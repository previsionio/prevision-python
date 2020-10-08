FROM python:3.6.8-slim-stretch as tests

ENV PATH /usr/src/app/bin:$PATH
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONIOENCODING UTF-8

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update; \
    apt install -y \
    make; \
    rm -rf /var/lib/apt/lists/*

COPY .requirements/requirements.txt requirements.txt
COPY .requirements/requirements-utests.txt requirements-utests.txt

RUN cat /usr/src/app/requirements.txt | xargs -n 1 pip install --no-cache-dir && \
    rm -f /usr/src/app/requirements.txt

RUN cat /usr/src/app/requirements-utests.txt | xargs -n 1 pip install --no-cache-dir && \
    rm -f /usr/src/app/requirements-utests.txt

ENTRYPOINT /bin/bash