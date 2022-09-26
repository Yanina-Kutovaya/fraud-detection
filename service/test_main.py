import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

from fastapi.testclient import TestClient

from src.fraud_detection.data.make_dataset import load_data
from src.fraud_detection.features import custom_transformers
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
    pipeline.fit(dataset)
    Model.pipeline = pipeline

    transaction = {
        'TransactionID': 123, 
        'TransactionDT': 568, 
        'TransactionAmt': 25.68, 
        'ProductCD': 'M', 
        'card1': 886,      
    }
    response = client.post('/predict?transaction_id=123', json=transaction)
    assert response.status_code == 200
    assert response.json() == {'transaction_id': 123, 'cards1': 886, 'probability': 0.0529}
