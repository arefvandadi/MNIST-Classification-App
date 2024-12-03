FROM python:3.12

WORKDIR /app

COPY ./templates ./templates
COPY ./app.py .
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py", "--server.port=8080", "--server.enableCORS=false"]