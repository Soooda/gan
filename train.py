import torch
import torch.nn as nn
import os

from data.mnist import create_dataloader
from model.discriminator import VanillaDiscriminator
from model.generator import VanillaGenerator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

num_epochs = 50
batch_size = 16
input_size = 28 * 28
generator_learning_rate = 1e-4
discriminator_learning_rate = 1e-4
k = 1

mnist = create_dataloader(batch_size=batch_size)

gen = VanillaGenerator(input_size).to(device)
dis = VanillaDiscriminator(input_size, batch_size).to(device)
criterion = nn.BCELoss()
gen_optimizer = torch.optim.SGD(gen.parameters(), momentum=0.9, lr=generator_learning_rate)
dis_optimizer = torch.optim.SGD(dis.parameters(), momentum=0.9, lr=discriminator_learning_rate)

for epoch in range(1, num_epochs + 1):
    for n, data in enumerate(mnist):
        imgs, _ = data
        dis_loss = 0.0
        gen_loss = 0.0
        for i in range(k):
            # Train Discriminator
            fake = gen.generate_noise(batch_size).to(device)
            fake = gen(fake)
            real = imgs.view(-1, input_size).to(device)

            dis_optimizer.zero_grad()
            pred_real = dis(real)
            error_real = criterion(pred_real, torch.ones(pred_real.size()).to(device))
            error_real.backward()

            pred_fake = dis(fake)
            error_fake = criterion(pred_fake, torch.zeros(pred_fake.size()).to(device))
            error_fake.backward()
            dis_loss += error_real + error_fake
            dis_optimizer.step()

        # Train generator
        fake = gen(gen.generate_noise(batch_size).to(device))
        gen_optimizer.zero_grad()
        pred = dis(fake)
        gen_loss += criterion(pred, torch.ones(pred_real.size()).to(device))
        gen_loss.backward()
        gen_optimizer.step()

    print("Epoch {:<4} Dis Loss: {:<8.4f} Gen Loss: {:<8.4f}".format(epoch, dis_loss / (i + 1), gen_loss / (i + 1)))
    checkpoints = {
        "gen_state_dict": gen.state_dict(),
        "dis_state_dict": dis.state_dict(),
        "gen_optimizer": gen_optimizer.state_dict(),
        "dis_optimizer": dis_optimizer.state_dict(),
    }

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(checkpoints, os.sep.join(("checkpoints", str(epoch) + ".pth")))

