#!/usr/bin/env python
"""Train and save model for Credit Cards Fraud Detection"""

import logging
import argparse
import sys
from typing import NoReturn

import pyspark
from pyspark.ml import Pipeline

from fraud_detection.data.make_dataset import load_data
from fraud_detection.models import train
from fraud_detection.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-m",
        "--model",
        required=False,
        default='GBTClassifier_pipeline_v1',
        help="model name (must be importable from titanic.models.train module)",
    )
    argparser.add_argument(
        "-d",
        "--datapath",
        required=False,
        default=None,
        help="dataset store path",
    )
    argparser.add_argument(
        "-o", 
        "--output_artifact", 
        required=False,
        default='Spark_GBTClassifier_v1', 
        help="filename to store model"
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    model_train_func = getattr(train, args.model)
    model = model_train_func()
    logging.info(f'Model {args.model} created')
    dataset = load_data(args.datapath)
    train_store(dataset, model, args.output)


def train_store(dataset: pyspark.sql.DataFrame, model: Pipeline, filename: str) -> NoReturn:
    logger.info(f'Training model on {dataset.count()} items')
    model.fit(dataset)
    store(model, filename)    


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
