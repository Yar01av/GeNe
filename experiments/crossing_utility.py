from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from gene.optimisers.crossing import CrossingOptimiser
from gene.optimisers.division import ParallelDivisionOptimiser, DivisionOptimiser
import torch
import torch.nn.functional as F
from torch import nn

from gene.targets import get_negative_accuracy_target
from gene.util import get_accuracy

DEVICE = ["cpu", "cuda"][1]
N_EPOCHS = 30


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
optimiser = CrossingOptimiser(target_func=get_negative_accuracy_target,
                              random_function=lambda shape: torch.normal(0, 15.0, shape),
                              selection_limit=10,
                              max_couples=10,
                              n_children_per_couple=2,
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

for e in tqdm(range(N_EPOCHS)):
    for images, labels in tqdm(train_loader, leave=False):
        models = optimiser.step(models, images.to(DEVICE), labels.to(DEVICE))

    latest_score = np.mean([get_accuracy(test_loader, m, DEVICE) for m in models])
    latest_scores.append(latest_score)
    print(latest_score)


plt.plot(latest_scores)
plt.show()
