
FROM python:3.7

RUN pip install gevent uwsgi flask

COPY app.py /app.py

EXPOSE 3000

ENTRYPOINT ["uwsgi", "--http", ":3000", "--master", "--module", "app:app"]