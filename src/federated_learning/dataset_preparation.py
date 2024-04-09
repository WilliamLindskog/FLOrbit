"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

from typing import Union
from omegaconf import DictConfig

from flwr_datasets import FederatedDataset

def preprocess_data(data: Union[FederatedDataset,], cfg: DictConfig) -> Union[FederatedDataset,]:
    """Preprocess the dataset.

    Parameters
    ----------
    data : Union[FederatedDataset,]
        The dataset to preprocess.
    cfg : DictConfig
        The configuration file.

    Returns
    -------
    Union[FederatedDataset,]
        The preprocessed dataset.
    """
    # Preprocess data here

    return data