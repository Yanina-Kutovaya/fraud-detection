import os
import logging
from pyspark.ml.pipeline import PipelineModel

__all__ = ['store', 'load']

logger = logging.getLogger()


def store(model: PipelineModel, filename: str, path: str = 'default'):
    if path == 'default':
        path = models_path()
    filepath = os.path.join(path, filename)

    logger.info(f'Dumpung model into {filepath}')    
    model.save(filepath)


def load(filename: str, path: str = 'default') -> PipelineModel:
    if path == 'default':
        path = models_path()
    filepath = os.path.join(path, filename)

    logger.info(f'Loading model from {filepath}')
    return PipelineModel.load(filepath)  


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, '..', '..', '..', 'models')
    return models_folder
