FROM ubuntu:18.04
FROM python:3.8.1
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

RUN chmod 777 startup.sh

CMD ./startup.sh
