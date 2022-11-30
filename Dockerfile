FROM ubuntu:20.04 as base

RUN apt-get update -y
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

COPY requirements.txt /usr/src/
RUN pip install -r /usr/src/requirements.txt

COPY . /usr/src/
WORKDIR /usr/src/

ENTRYPOINT ["python3", "/usr/src/launcher_main.py"]