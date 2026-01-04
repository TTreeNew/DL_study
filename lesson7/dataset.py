import torch

def get_beans(dataset_path = "dataset_4.pt"):
    X, y, z = torch.load(dataset_path)
    X = X.numpy()
    y = y.numpy()
    z = z.numpy()
    return X,y,z

