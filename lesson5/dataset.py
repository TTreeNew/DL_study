import torch

def get_beans(dataset_path = "dataset_3.pt"):
    X, y = torch.load(dataset_path)
    X = X.numpy()
    y = y.numpy()
    return X,y

