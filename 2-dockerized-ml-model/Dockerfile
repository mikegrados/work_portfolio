ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

RUN mkdir /src
WORKDIR /src
ENV BASE_DIR=/src/logreg

COPY . /src

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

RUN mkdir /src/logreg/datasets
RUN mkdir /src/logreg/trained_models

RUN python3 /src/logreg/train_pipeline.py