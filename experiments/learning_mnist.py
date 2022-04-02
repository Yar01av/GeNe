from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from gene.optimisers.division import ParallelDivisionOptimiser, DivisionOptimiser
import torch
import torch.nn.functional as F
from torch import nn


DEVICE = ["cpu", "cuda"][1]
N_EPOCHS = 3


def get_accuracy(test_loader, model):
    preds = []
    trues = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        y_pred = model(images)

        preds.extend(y_pred.to("cpu").detach().numpy())
        trues.extend(labels.to("cpu").detach().numpy())

    total_matches = np.sum(np.argmax(preds, axis=-1) == trues)

    return total_matches/len(preds)


# Define the model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28*28, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
)
models = [model.to(DEVICE)]

# Define the optimiser
optimiser = DivisionOptimiser(target_func=nn.CrossEntropyLoss(reduction="mean"),
                              random_function=lambda shape: torch.normal(0, 0.1, shape), selection_limit=10,
                              device=DEVICE)

# Define the data
train_data = datasets.MNIST(
        root="./cache",
        download=True,
        train=True,
        transform=transforms.ToTensor()
)
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
test_data = datasets.MNIST(
        root="./cache",
        download=True,
        train=False,
        transform=transforms.ToTensor()
    )
test_loader = DataLoader(test_data, batch_size=1024)

latest_scores = []

start = datetime.now()

for e in tqdm(range(N_EPOCHS)):
    for images, labels in tqdm(train_loader, leave=False):
        models = optimiser.step(models, images.to(DEVICE), labels.to(DEVICE))

        latest_scores.append(np.mean([get_accuracy(test_loader, m)
                                      for m in models]))

end = datetime.now()
print(end-start)

plt.plot(latest_scores)
plt.show()
