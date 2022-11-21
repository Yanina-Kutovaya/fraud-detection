import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

import json
from fastapi.testclient import TestClient

from src.fraud_detection.data.make_dataset import load_data
from src.fraud_detection.models.train import GBTClassifier_pipeline_v1

from main import app, Model 

import findspark
findspark.init() 

client = TestClient(app)


def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Green"


def test_predict():
    dataset = load_data()
    pipeline = GBTClassifier_pipeline_v1()    
    Model.pipeline = pipeline.fit(dataset)

    transaction = {
        'TransactionID': 123, 
        'TransactionDT': 568, 
        'TransactionAmt': 25.68, 
        'ProductCD': 'M', 
        'card1': 886,      
    }
    response = client.post('/predict?transaction_id=123', json=transaction)
    assert response.status_code == 200
    
    assert response.json()['TransactionID'] == 123
    assert response.json()['cards1'] == 886
    print(f'probability = {response.json()["probability"]}')


def test_predict_batch():
    dataset = load_data()
    pipeline = GBTClassifier_pipeline_v1()    
    Model.pipeline = pipeline.fit(dataset)

    df = load_data(file_name='test.parquet')    
    transactions = df.toJSON().map(lambda j: json.loads(j)).collect()

    response = client.post('/predict_batch?batch_id=1', json=transactions)
    assert response.status_code == 200    
    assert response.json()['batch_id'] == 1
    assert len(response.json()['suspicious_cards']) == 50
    print(f'suspicious_cards = {response.json()["suspicious_cards"]}')