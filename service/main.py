"""FastAPI Credit Cards Fraud Detection model inference"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'..'))

import pandas as pd
import numpy as np
from typing import Optional, List

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from fastapi import FastAPI, HTTPException
from starlette_exporter import PrometheusMiddleware, handle_metrics
from pydantic import BaseModel
from pyspark.ml.pipeline import PipelineModel

from src.fraud_detection.features import custom_transformers
from src.fraud_detection.models.serialize import load
from src.fraud_detection.models.inference_tools import (
    get_params, get_k_suspicious_cards
)


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

MODEL = os.getenv("MODEL", default='Spark_GBTClassifier_v1')
spark = SparkSession.builder.getOrCreate()

class Model:
    pipeline: Optional[PipelineModel] = None


class Transaction(BaseModel):       
    TransactionID: int    
    TransactionDT: int
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None 
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D6: Optional[float] = None
    D7: Optional[float] = None
    D8: Optional[float] = None
    D9: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D12: Optional[float] = None
    D13: Optional[float] = None
    D14: Optional[float] = None
    D15: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    V4: Optional[float] = None
    V5: Optional[float] = None
    V6: Optional[float] = None
    V7: Optional[float] = None
    V8: Optional[float] = None
    V9: Optional[float] = None
    V10: Optional[float] = None
    V11: Optional[float] = None
    V12: Optional[float] = None
    V13: Optional[float] = None
    V14: Optional[float] = None
    V15: Optional[float] = None
    V16: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V19: Optional[float] = None
    V20: Optional[float] = None
    V21: Optional[float] = None
    V22: Optional[float] = None
    V23: Optional[float] = None
    V24: Optional[float] = None
    V25: Optional[float] = None
    V26: Optional[float] = None
    V27: Optional[float] = None
    V28: Optional[float] = None
    V29: Optional[float] = None
    V30: Optional[float] = None
    V31: Optional[float] = None
    V32: Optional[float] = None
    V33: Optional[float] = None
    V34: Optional[float] = None
    V35: Optional[float] = None
    V36: Optional[float] = None
    V37: Optional[float] = None
    V38: Optional[float] = None
    V39: Optional[float] = None
    V40: Optional[float] = None
    V41: Optional[float] = None
    V42: Optional[float] = None
    V43: Optional[float] = None
    V44: Optional[float] = None
    V45: Optional[float] = None
    V46: Optional[float] = None
    V47: Optional[float] = None
    V48: Optional[float] = None
    V49: Optional[float] = None
    V50: Optional[float] = None
    V51: Optional[float] = None
    V52: Optional[float] = None
    V53: Optional[float] = None
    V54: Optional[float] = None
    V55: Optional[float] = None
    V56: Optional[float] = None
    V57: Optional[float] = None
    V58: Optional[float] = None
    V59: Optional[float] = None
    V60: Optional[float] = None
    V61: Optional[float] = None
    V62: Optional[float] = None
    V63: Optional[float] = None
    V64: Optional[float] = None
    V65: Optional[float] = None
    V66: Optional[float] = None
    V67: Optional[float] = None
    V68: Optional[float] = None
    V69: Optional[float] = None
    V70: Optional[float] = None
    V71: Optional[float] = None
    V72: Optional[float] = None
    V73: Optional[float] = None
    V74: Optional[float] = None
    V75: Optional[float] = None
    V76: Optional[float] = None
    V77: Optional[float] = None
    V78: Optional[float] = None
    V79: Optional[float] = None
    V80: Optional[float] = None
    V81: Optional[float] = None
    V82: Optional[float] = None
    V83: Optional[float] = None
    V84: Optional[float] = None
    V85: Optional[float] = None
    V86: Optional[float] = None
    V87: Optional[float] = None
    V88: Optional[float] = None
    V89: Optional[float] = None
    V90: Optional[float] = None
    V91: Optional[float] = None
    V92: Optional[float] = None
    V93: Optional[float] = None
    V94: Optional[float] = None
    V95: Optional[float] = None
    V96: Optional[float] = None
    V97: Optional[float] = None
    V98: Optional[float] = None
    V99: Optional[float] = None
    V100: Optional[float] = None
    V101: Optional[float] = None
    V102: Optional[float] = None
    V103: Optional[float] = None
    V104: Optional[float] = None
    V105: Optional[float] = None
    V106: Optional[float] = None
    V107: Optional[float] = None
    V108: Optional[float] = None
    V109: Optional[float] = None
    V110: Optional[float] = None
    V111: Optional[float] = None
    V112: Optional[float] = None
    V113: Optional[float] = None
    V114: Optional[float] = None
    V115: Optional[float] = None
    V116: Optional[float] = None
    V117: Optional[float] = None
    V118: Optional[float] = None
    V119: Optional[float] = None
    V120: Optional[float] = None
    V121: Optional[float] = None
    V122: Optional[float] = None
    V123: Optional[float] = None
    V124: Optional[float] = None
    V125: Optional[float] = None
    V126: Optional[float] = None
    V127: Optional[float] = None
    V128: Optional[float] = None
    V129: Optional[float] = None
    V130: Optional[float] = None
    V131: Optional[float] = None
    V132: Optional[float] = None
    V133: Optional[float] = None
    V134: Optional[float] = None
    V135: Optional[float] = None
    V136: Optional[float] = None
    V137: Optional[float] = None
    V138: Optional[float] = None
    V139: Optional[float] = None
    V140: Optional[float] = None
    V141: Optional[float] = None
    V142: Optional[float] = None
    V143: Optional[float] = None
    V144: Optional[float] = None
    V145: Optional[float] = None
    V146: Optional[float] = None
    V147: Optional[float] = None
    V148: Optional[float] = None
    V149: Optional[float] = None
    V150: Optional[float] = None
    V151: Optional[float] = None
    V152: Optional[float] = None
    V153: Optional[float] = None
    V154: Optional[float] = None
    V155: Optional[float] = None
    V156: Optional[float] = None
    V157: Optional[float] = None
    V158: Optional[float] = None
    V159: Optional[float] = None
    V160: Optional[float] = None
    V161: Optional[float] = None
    V162: Optional[float] = None
    V163: Optional[float] = None
    V164: Optional[float] = None
    V165: Optional[float] = None
    V166: Optional[float] = None
    V167: Optional[float] = None
    V168: Optional[float] = None
    V169: Optional[float] = None
    V170: Optional[float] = None
    V171: Optional[float] = None
    V172: Optional[float] = None
    V173: Optional[float] = None
    V174: Optional[float] = None
    V175: Optional[float] = None
    V176: Optional[float] = None
    V177: Optional[float] = None
    V178: Optional[float] = None
    V179: Optional[float] = None
    V180: Optional[float] = None
    V181: Optional[float] = None
    V182: Optional[float] = None
    V183: Optional[float] = None
    V184: Optional[float] = None
    V185: Optional[float] = None
    V186: Optional[float] = None
    V187: Optional[float] = None
    V188: Optional[float] = None
    V189: Optional[float] = None
    V190: Optional[float] = None
    V191: Optional[float] = None
    V192: Optional[float] = None
    V193: Optional[float] = None
    V194: Optional[float] = None
    V195: Optional[float] = None
    V196: Optional[float] = None
    V197: Optional[float] = None
    V198: Optional[float] = None
    V199: Optional[float] = None
    V200: Optional[float] = None
    V201: Optional[float] = None
    V202: Optional[float] = None
    V203: Optional[float] = None
    V204: Optional[float] = None
    V205: Optional[float] = None
    V206: Optional[float] = None
    V207: Optional[float] = None
    V208: Optional[float] = None
    V209: Optional[float] = None
    V210: Optional[float] = None
    V211: Optional[float] = None
    V212: Optional[float] = None
    V213: Optional[float] = None
    V214: Optional[float] = None
    V215: Optional[float] = None
    V216: Optional[float] = None
    V217: Optional[float] = None
    V218: Optional[float] = None
    V219: Optional[float] = None
    V220: Optional[float] = None
    V221: Optional[float] = None
    V222: Optional[float] = None
    V223: Optional[float] = None
    V224: Optional[float] = None
    V225: Optional[float] = None
    V226: Optional[float] = None
    V227: Optional[float] = None
    V228: Optional[float] = None
    V229: Optional[float] = None
    V230: Optional[float] = None
    V231: Optional[float] = None
    V232: Optional[float] = None
    V233: Optional[float] = None
    V234: Optional[float] = None
    V235: Optional[float] = None
    V236: Optional[float] = None
    V237: Optional[float] = None
    V238: Optional[float] = None
    V239: Optional[float] = None
    V240: Optional[float] = None
    V241: Optional[float] = None
    V242: Optional[float] = None
    V243: Optional[float] = None
    V244: Optional[float] = None
    V245: Optional[float] = None
    V246: Optional[float] = None
    V247: Optional[float] = None
    V248: Optional[float] = None
    V249: Optional[float] = None
    V250: Optional[float] = None
    V251: Optional[float] = None
    V252: Optional[float] = None
    V253: Optional[float] = None
    V254: Optional[float] = None
    V255: Optional[float] = None
    V256: Optional[float] = None
    V257: Optional[float] = None
    V258: Optional[float] = None
    V259: Optional[float] = None
    V260: Optional[float] = None
    V261: Optional[float] = None
    V262: Optional[float] = None
    V263: Optional[float] = None
    V264: Optional[float] = None
    V265: Optional[float] = None
    V266: Optional[float] = None
    V267: Optional[float] = None
    V268: Optional[float] = None
    V269: Optional[float] = None
    V270: Optional[float] = None
    V271: Optional[float] = None
    V272: Optional[float] = None
    V273: Optional[float] = None
    V274: Optional[float] = None
    V275: Optional[float] = None
    V276: Optional[float] = None
    V277: Optional[float] = None
    V278: Optional[float] = None
    V279: Optional[float] = None
    V280: Optional[float] = None
    V281: Optional[float] = None
    V282: Optional[float] = None
    V283: Optional[float] = None
    V284: Optional[float] = None
    V285: Optional[float] = None
    V286: Optional[float] = None
    V287: Optional[float] = None
    V288: Optional[float] = None
    V289: Optional[float] = None
    V290: Optional[float] = None
    V291: Optional[float] = None
    V292: Optional[float] = None
    V293: Optional[float] = None
    V294: Optional[float] = None
    V295: Optional[float] = None
    V296: Optional[float] = None
    V297: Optional[float] = None
    V298: Optional[float] = None
    V299: Optional[float] = None
    V300: Optional[float] = None
    V301: Optional[float] = None
    V302: Optional[float] = None
    V303: Optional[float] = None
    V304: Optional[float] = None
    V305: Optional[float] = None
    V306: Optional[float] = None
    V307: Optional[float] = None
    V308: Optional[float] = None
    V309: Optional[float] = None
    V310: Optional[float] = None
    V311: Optional[float] = None
    V312: Optional[float] = None
    V313: Optional[float] = None
    V314: Optional[float] = None
    V315: Optional[float] = None
    V316: Optional[float] = None
    V317: Optional[float] = None
    V318: Optional[float] = None
    V319: Optional[float] = None
    V320: Optional[float] = None
    V321: Optional[float] = None
    V322: Optional[float] = None
    V323: Optional[float] = None
    V324: Optional[float] = None
    V325: Optional[float] = None
    V326: Optional[float] = None
    V327: Optional[float] = None
    V328: Optional[float] = None
    V329: Optional[float] = None
    V330: Optional[float] = None
    V331: Optional[float] = None
    V332: Optional[float] = None
    V333: Optional[float] = None
    V334: Optional[float] = None
    V335: Optional[float] = None
    V336: Optional[float] = None
    V337: Optional[float] = None
    V338: Optional[float] = None
    V339: Optional[float] = None


class TransactionList(BaseModel):
    transactions: List[Transaction]


@app.on_event('startup')
def load_model():
    Model.pipeline = load(MODEL)


@app.get('/')
def read_healthcheck():
    return {'status': 'Green', 'version': '0.1.0'}


@app.post('/predict')
def predict(transaction_id: int, transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    data.fillna(value=np.nan, inplace=True, downcast=False)   

    if Model.pipeline is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:        
        df = spark.createDataFrame(data)
        predictions = Model.pipeline.transform(df)        
        TransactionID, card1, probability = get_params(predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'TransactionID': TransactionID, 'cards1': card1, 'probability': probability}


@app.post('/predict_batch')
def predict_batch(batch_id: int, transactions: TransactionList):
    data = pd.DataFrame([t.dict() for t in transactions.transactions])
    data.fillna(value=np.nan, inplace=True, downcast=False)   

    if Model.pipeline is None:
        raise HTTPException(status_code=503, detail='No model loaded')
    try:        
        df = spark.createDataFrame(data)
        predictions = Model.pipeline.transform(df)
        suspicious_cards = get_k_suspicious_cards(predictions, k=50)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'batch_id': batch_id, 'suspicious_cards': suspicious_cards}
