import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType


def adjust_predictions(predictions):
    def extract_prob(v):
        try:
            return float(v[1])  
        except ValueError:
            return None
    extract_prob_udf = F.udf(extract_prob, DoubleType())
    predictions = predictions.withColumn(
        'prob', extract_prob_udf(F.col('probability'))
    )
    return predictions 


def get_fraud_probability(predictions):
    predictions = adjust_predictions(predictions)
    TransactionID = predictions.select('TransactionID').collect()[0][0]
    card1 = predictions.select('card1').collect()[0][0]
    probability = predictions.select('prob').collect()[0][0]

    return probability


def get_k_suspicious_cards(predictions, k=50):
    predictions = adjust_predictions(predictions)   
    suspicious_cards = (
        predictions.select('card1', 'prob')
        .groupBy('card1').max('prob')
        .orderBy(F.col('max(prob)').desc())    
        .limit(k)        
        .toPandas()['card1']
        .tolist()    
    )
    return suspicious_cards