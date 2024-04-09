"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import torch
import flwr as fl

from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Callable, Union, OrderedDict, Dict

from flwr.common import Scalar
from flwr_datasets import FederatedDataset

from torch.cuda import is_available

from .utils import is_net_model, train, test

class NetClient(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        data: FederatedDataset,
        cid: str,
        device: torch.device,
        cfg: DictConfig,
    ) -> None:
        self.net = net
        self.device = device
        self.cid = int(cid)
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.task = cfg.task

        # Get dataloaders 
        # self.trainloader, self.testloader = _load_data(self.df, cfg, self.batch_size,)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        self.set_parameters(parameters)
        # Get size of parameters in bytes 
        total_bytes = sum([p.numel() * p.element_size() for p in self.net.parameters()])
        train(self.net, self.trainloader, self.cfg)
        final_p_np = self.get_parameters({})

        return final_p_np, len(self.trainloader), {'total_bytes': total_bytes}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.testloader, device=self.device, task=self.task, evaluate=True)
        return float(loss), len(self.testloader), metrics
    

def get_client_fn(
    data: FederatedDataset,
    cfg: DictConfig,
) -> Callable[[str], Union[NetClient,]]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    fds : FederatedDataset
        The federated dataset object that contains the data, can be partitioned. 
    cfg : DictConfig
        An omegaconf object that stores the hydra config for the model.
    Returns
    -------
    Callable[[str], FlowerClient]
        The client function that creates the flower clients
    """

    def client_fn(cid: str) -> Union[NetClient,]:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if is_available() else "cpu")
        if is_net_model(cfg.model.name): 
            model = instantiate(cfg.model).to(device)
        else:
            raise ValueError(f"Model {cfg.model.name} not supported")

        client = instantiate(cfg.client)
        return client(model, df, device, cid, cfg.client)

    return client_fn