import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, DoubleType


def get_k_suspicious_cards(predictions, k=50):

    def extract_prob(v):
        try:
            return float(v[1])  
        except ValueError:
            return None

    extract_prob_udf = F.udf(extract_prob, DoubleType())
    predictions = predictions.withColumn(
          'prob_flag', extract_prob_udf(F.col('probability'))
    )    
    suspicious_cards = (
        predictions.select('card1', 'prob_flag')
        .groupBy('card1').max('prob_flag')
        .orderBy(F.col('max(prob_flag)').desc())    
        .limit(k)        
        .toPandas()['card1']
        .tolist()    
    )
    return suspicious_cards