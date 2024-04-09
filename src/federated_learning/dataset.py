"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import Union, Tuple

from flwr_datasets import FederatedDataset

from torch.utils.data import DataLoader

from .utils import data_needs_downloading, download_dataset

def load_dataset(name: str) -> Union[FederatedDataset, Tuple[DataLoader, DataLoader]]:
    """Load the dataset.

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    Returns
    -------
    Union[FederatedDataset, Tuple[DataLoader, DataLoader]]
        The dataset or the dataloaders for the clients and the server.
    """

    # Check if data must be downloaded
    if data_needs_downloading(name):
        download_dataset(name)