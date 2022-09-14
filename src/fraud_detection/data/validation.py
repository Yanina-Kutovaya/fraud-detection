import logging
import pyspark
import pyspark.sql.functions as F
from typing import Tuple

logger = logging.getLogger(__name__)

__all__ = ['train_validation_split']

def train_validation_split(data: pyspark.sql.DataFrame
    )  -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    logging.info(f'Train validation split')
    t = 3600 * 24 * 30
    train = data.filter(F.col('TransactionDT') < t)
    compromised_cards = (
        train['card1', 'isFraud'].filter(F.col('isFraud') == 1)
        .groupBy('card1').count().toPandas()['card1'].tolist()
    )           
    valid = (
        data.filter(F.col('TransactionDT') >= t)
        .where(~F.col('card1').isin(compromised_cards))
    )
    return train, valid
