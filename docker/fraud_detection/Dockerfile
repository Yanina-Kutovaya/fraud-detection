FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN useradd --user-group --shell /bin/false fraud_detection  
USER fraud_detection

EXPOSE 8000

ENV MODEL "Spark_GBTClassifier_v1"

CMD ["bash", "start.sh"]

