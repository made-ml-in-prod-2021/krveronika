import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from ml_project.enities import SplittingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def read_data(
    path: str, 
    **kwargs
) -> pd.DataFrame:
    """
    Read pandas DataFrame
    """
    data = pd.DataFrame()
    logger.info("Start reading Dataframe...")
    try:
        data = pd.read_csv(path, **kwargs)
        logger.info(f"Dataframe have succesfully read, shape: {data.shape}")
    except NameError:
        logger.exception(f"File {path} not found")
    except Exception as e:
        logger.exception(f"Error in reading {path}")
        logger.exception(f"Error: {e}")
    return data


def split_train_test_data(
    data: pd.DataFrame, 
    params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting data into train and test
    """
    train_data, test_data = train_test_split(
        data, test_size = params.validation_size, 
        random_state = params.random_state
    )
    logger.info(f"train_df.shape is {train_data.shape}")
    logger.info(f"test_df.shape is {test_data.shape}")
    return train_data, test_data

