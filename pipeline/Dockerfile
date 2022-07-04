FROM python:3.8.10


WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn run:app --host 0.0.0.0 --port 3000

EXPOSE 3000

ARG VERSION
ENV VERSION=$VERSION