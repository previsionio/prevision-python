#!/bin/bash

TAG=${TAG:-dev}
PREVISION_REGISTRY_ADDR=eu.gcr.io/prevision-enterprise

REQUIREMENTS_PATH='.requirements'
mkdir -p $REQUIREMENTS_PATH

cp -a ../requirements.txt $REQUIREMENTS_PATH/
cp -a ../utests/requirements.txt $REQUIREMENTS_PATH/requirements-utests.txt

DOCKER_BUILDKIT=1 docker build -t ${PREVISION_REGISTRY_ADDR}/prevision/prevision-python-tests:${TAG} --secret id=python,src=$HOME/.config/pip/pip.conf .
