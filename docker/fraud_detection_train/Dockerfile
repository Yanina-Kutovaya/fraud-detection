FROM python:3.10.8-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

RUN useradd --user-group --shell /bin/false fraud_detection
RUN chown -R fraud_detection models  
USER fraud_detection

CMD ["./scripts/train_save_model.py", "-m", "make_baseline_model", "-o", "Spark_GBTClassifier_v1", "-v"]

