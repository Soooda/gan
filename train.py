import torch

from data.mnist import create_dataloader
from model.discriminator import VanillaDiscriminator
from model.generator import VanillaGenerator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

num_epochs = 100
batch_size = 128
input_size = 28 * 28
generator_learning_rate = 1e-6
discriminator_learning_rate = 1e-6
k = 1

mnist = create_dataloader(batch_size=batch_size)

gen = VanillaGenerator(input_size).to(device)
dis = VanillaDiscriminator(input_size).to(device)

gen_optimizer = torch.optim.SGD(gen.parameters(), momentum=0.9, lr=generator_learning_rate)
dis_optimizer = torch.optim.SGD(dis.parameters(), momentum=0.9, lr=discriminator_learning_rate)

for epoch in range(1, num_epochs + 1):
    for i in range(k):