import torch.nn as nn


class VanillaDiscriminator(nn.Module):
    def __init__(self, input_size, batch_size):
        super(VanillaDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, data):
        return self.model(data)


if __name__ == "__main__":
    a = VanillaDiscriminator(28 * 28)
    print(a)
