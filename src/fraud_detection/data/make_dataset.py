import logging
import pyspark
from pyspark import SparkFiles
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['load_train_dataset']


APP_NAME = 'data_ingestion'
TRAIN_DATA_URL = 'https://storage.yandexcloud.net/airflow-cc-input/train.parquet'
FILE_NAME = 'train.parquet'


def load_data(datapath: Optional[str] = None) -> pyspark.sql.DataFrame:    
    if datapath is None:
        datapath = TRAIN_DATA_URL

    logging.info(f'Reading dataset from {datapath}')
    spark = (
        pyspark.sql.SparkSession.builder
            .appName(APP_NAME)       
            .getOrCreate()
    )
    spark.sparkContext.addFile(datapath)
    df_train = spark.read.parquet(
        SparkFiles.get(FILE_NAME), header=True, inferSchema=True
    ).repartition(4)

    return df_train   
