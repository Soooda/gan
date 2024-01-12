import torch
import torch.nn as nn

# Custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        
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

class VanillaGenerator(nn.Module):
    def __init__(self, output_size, input_size=128):
        super(VanillaGenerator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, data):
        return self.model(data)
    
    def generate_noise(self, n):
        noise = torch.randn(n, self.input_size)
        return noise
    

if __name__ == "__main__":
    a = VanillaGenerator(28 * 28)
    b = VanillaDiscriminator(28 * 28)
    print(a)
    print(b)