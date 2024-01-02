import torch
import torch.utils.data as data
import torchvision
import os.path as osp

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(0.5, 1)
])


def create_dataloader(batch_size, train=True, shuffle=True):
    mnist = torchvision.datasets.MNIST(root=osp.sep.join(('.', 'datasets')), train=train, download=True, transform=None)
    train_data = data.DataLoader(mnist, batch_size=batch_size, shuffle=shuffle)
    return train_data
