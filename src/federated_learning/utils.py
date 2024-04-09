from flwr_datasets import FederatedDataset, partitioner

from .constants import DATASETS

from typing import Tuple, Union
from omegaconf import DictConfig
import requests
# logging 
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
    
def download_dataset(
        cfg: DictConfig
    ) -> Union[FederatedDataset, Tuple[DataLoader, DataLoader]]:
    """Download the dataset.

    Parameters
    ----------
    cfg : DictConfig
        The configuration file.
    """
    # Check if the dataset is available
    if not _dataset_available(cfg.name):
        _logging_messages("error", f"Dataset {cfg.name} is not available.")
    
    # Get the dataset information
    dataset = DATASETS[cfg.name]
    
    # Download the dataset
    download_link = dataset["link"]
    is_flwr_dataset = dataset["flwr_dataset"]
    # Download the dataset using the download link
    # This is a placeholder for the actual download code
    _logging_messages("info", f"Downloading dataset {cfg.name} from {download_link}...")
    if is_flwr_dataset:
        data = _flwr_dataset(cfg)
    else:
        data = _non_flwr_dataset(download_link, cfg.name)

    return data

def _flwr_dataset(cfg: DictConfig) -> FederatedDataset:
    """Load the federated dataset.

    Parameters
    ----------
    cfg : DictConfig
        The configuration file.

    Returns
    -------
    FederatedDataset
        The federated dataset.
    """
    # Get the partitioner
    partition = _get_partitioner(cfg.partition, cfg.num_clients)

    fds = FederatedDataset(
        dataset = DATASETS[cfg.name]["hf_name"],
        partitioners = {"train": partition},
    ) 
    _logging_messages("info", f"Federated dataset {cfg.name} loaded successfully.")

    return fds

def _get_partitioner(partitioner_type: str, num_clients: int) -> FederatedDataset:
    """Get the partitioner for the dataset.

    Parameters
    ----------
    partitioner_type : str
        The type of partitioner to use.
    num_clients : int  
        The number of clients to partition the dataset.

    Returns
    -------
    FederatedDataset
        The partitioner for the dataset.
    """
    if partitioner_type == "iid":
        return partitioner.IidPartitioner(num_partitions=num_clients)
        

def _non_flwr_dataset(url: str, name: str) -> Tuple[DataLoader, DataLoader]:
    """Load the non-federated dataset.

    Parameters
    ----------
    link : str
        The link to the dataset.
    name : str
        The name of the dataset to load.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The dataloaders for the clients and the server.
    """

    # Send a HTTP request to the URL of the file, stream=True to prevent loading the content at once
    response = requests.get(url, stream=True)

    # Check if the request is successful
    if response.status_code == 200:
        # Open the file in write-binary mode
        with open(f"./data/{name}.csv", "wb") as file:
            # Iterate over the response content
            for chunk in response.iter_content(chunk_size=128):
                # Write the chunk to the file
                file.write(chunk)
    else:
        _logging_messages("error", f"Failed to download the dataset from {url}.")

def _dataset_available(name: str) -> bool:
    """Check if the dataset exists.

    Parameters
    ----------
    name : str
        The name of the dataset to check.

    Returns
    -------
    bool
        Whether the dataset exists.
    """
    return name in DATASETS

def _logging_messages(type_log: str, message: str) -> None:
    """Log messages based on the type of log.

    Parameters
    ----------
    type_log : str
        The type of log to use.
    message : str
        The message to log.
    """
    if type_log == "info":
        logger.info(message)
    elif type_log == "error":
        logger.error(message)
        return 
    elif type_log == "warning":
        logger.warning(message)
    else:
        raise ValueError(f"Invalid log type: {type_log}")
    
def is_net_model(model_name: str) -> bool:
    """Check if the model is a neural network model.

    Parameters
    ----------
    model_name : str
        The name of the model to check.

    Returns
    -------
    bool
        Whether the model is a neural network model.
    """
    if model_name in ["resnet", "lenet", "vgg"]:
        return True
    else: 
        raise ValueError(f"Model {model_name} is not a neural network model.")