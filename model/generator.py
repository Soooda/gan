import torch
import torch.nn as nn


class VanillaGenerator(nn.Module):
    def __init__(self, output_size, input_size=128):
        super(VanillaGenerator, self).__init__()
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.model(data)
    
    def generate_noise(self):
        noise = self.randn(self.output_size).to(self.get_device())
        return noise
    

if __name__ == "__main__":
    a = VanillaGenerator(28 * 28)
    print(a)
