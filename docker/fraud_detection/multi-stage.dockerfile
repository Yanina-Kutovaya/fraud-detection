# First stage - train model
FROM python:3.10.8-slim-buster AS trainer

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN ./scripts/train_save_model.py \
 -m make_baseline_model \
 -o Spark_GBTClassifier_v1 \
 -d 'https://storage.yandexcloud.net/airflow-cc-input/train.parquet' \
 -v

# Second stage - create service and get model from previous stage
FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

COPY --from=trainer /usr/src/app/models/Spark_GBTClassifier_v1 ./models/

RUN useradd --user-group --shell /bin/fraud_detection  
USER fraud_detection

EXPOSE 8000

ENV MODEL "Spark_GBTClassifier_v1"

CMD ["bash", "start.sh"]