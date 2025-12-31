import torch

def get_beans(dataset_path = "dataset_1.pt"):
    X, y = torch.load(dataset_path)
    return X,y

