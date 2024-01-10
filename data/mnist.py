import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 1)
])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def create_dataloader(batch_size, shuffle=True):
    mnist = torchvision.datasets.MNIST(root=osp.sep.join(('.', 'datasets')), download=True, transform=transform)
    train_data = data.DataLoader(mnist, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return train_data

if __name__ == "__main__":
    dataloader = create_dataloader(32)
    batch = next(iter(dataloader))
    print(batch[0].size())
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()