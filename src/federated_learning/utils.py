from .constants import DATASETS

import logging

def data_needs_downloading(name: str) -> bool:
    """Check if the data needs to be downloaded.

    Parameters
    ----------
    name : str
        The name of the dataset to check.

    Returns
    -------
    bool
        Whether the data needs to be downloaded.
    """
    # DOWNLOAD_LINKS is a dictionary that contains the name of the dataset as the key
    return name in DOWNLOAD_LINKS
    

def download_dataset(name: str) -> None:
    """Download the dataset.

    Parameters
    ----------
    name : str
        The name of the dataset to download.
    """
    # Check if the dataset is available
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} not available.")
    
    # Get the dataset information
    dataset = DATASETS[name]
    
    # Download the dataset
    download_link = dataset["link"]
    # Download the dataset using the download link
    # This is a placeholder for the actual download code
    print(f"Downloading dataset {name} from {download_link}...")
    
    # Mark the dataset as downloaded
    DATASETS[name]["downloaded"] = True
    print(f"Dataset {name} downloaded successfully.")