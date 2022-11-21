import logging
import pyspark
from pyspark import SparkFiles
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['load_train_dataset']


APP_NAME = 'data_ingestion'
BACKET_URL = 'https://storage.yandexcloud.net/airflow-cc-input/'
FILE_NAME = 'train.parquet'

def load_data(
    file_name: Optional[str] = None,
    datapath: Optional[str] = None    
    ) -> pyspark.sql.DataFrame:  
    if file_name is None:
        file_name = FILE_NAME  
    if datapath is None:
        datapath = BACKET_URL + file_name    

    logging.info(f'Reading dataset from {datapath}')
    spark = (
        pyspark.sql.SparkSession.builder
            .appName(APP_NAME)       
            .getOrCreate()
    )
    spark.sparkContext.addFile(datapath)
    df = spark.read.parquet(
        SparkFiles.get(file_name), header=True, inferSchema=True
    ).repartition(4)

    return df