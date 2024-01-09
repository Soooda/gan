import torch
import matplotlib.pyplot as plt
import os.path as osp

from model.vanillagan import VanillaGenerator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint = osp.sep.join(("checkpoints", "vanilla.pth"))
input_size = 28 * 28
model = VanillaGenerator(input_size).to(device)

with torch.no_grad():
    model.eval()
    temp = torch.load(checkpoint, map_location=device)
    ret = model.load_state_dict(temp['gen_state_dict'])
    print(ret)

    for i in range(10):
        noise = model.generate_noise(1).to(device)
        generated_data = model(noise).view(28, 28).cpu()

        plt.figure(figsize=(4, 4))
        plt.axis('Off')
        plt.imshow(generated_data, cmap="gray")
        plt.show()