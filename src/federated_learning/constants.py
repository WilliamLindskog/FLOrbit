DATASETS = {
    "remote_sensing" : {
        "flwr_dataset" : True,
        "link" : "https://huggingface.co/datasets/blanchon/UC_Merced",
        "hf_name" : "blanchon/UC_Merced",
        "task" : "classification",
    },
}

TARGETS = {
    "mlp" : {
        "path" : "src.federated_learning.models.Net",
        "client_type" : "NetClient",
    },
    "resnet" : {
        "path" : "src.federated_learning.models.ResNet",
        "client_type" : "NetClient",
    },
    "lenet" : {
        "path" : "src.federated_learning.models.LeNet",
        "client_type" : "NetClient",
    },
}