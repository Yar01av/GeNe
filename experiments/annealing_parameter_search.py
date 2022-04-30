import random

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from gene.optimisers.annealing import AnnealingOptimiser
import torch

from gene.targets import get_negative_accuracy_target
from gene.util import get_accuracy, grid_search


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

DEVICE = ["cpu", "cuda"][1]
N_EPOCHS = 5


def train(decay=0.999, init_std=5):
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
    optimiser = AnnealingOptimiser(target_func=get_negative_accuracy_target,
                                   init_std=init_std,
                                   std_updater=lambda std: std*decay if std >= 0.0001 else std,
                                   selection_limit=10,
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

    for e in tqdm(range(N_EPOCHS)):
        for images, labels in train_loader:
            models = optimiser.step(models, images.to(DEVICE), labels.to(DEVICE))

    return np.mean([get_accuracy(test_loader, m, DEVICE) for m in models])


parameters = {"decay": np.logspace(1, np.log10(0.9), 10),
              "init_std": np.logspace(-2, 1.3, 10   )}
n_param_0, n_param_1 = len(list(parameters.values())[0]), len(list(parameters.values())[1])
results = grid_search(trainer=train, parameters=parameters)
print(results)
scores = [result.score for result in results]
scores = np.reshape(scores, (n_param_0, n_param_1))

fig, ax = plt.subplots()
ax.imshow(scores)

for i in range(n_param_0):
    for j in range(n_param_1):
        text = ax.text(j, i, str(scores[i, j])[:4],
                       ha="center", va="center", color="w")
ax.set_xticks(range(n_param_1))
ax.set_xticklabels([str(val)[:6] for val in list(parameters.values())[1]])
ax.set_yticks(range(n_param_0))
ax.set_yticklabels([str(val)[:6] for val in list(parameters.values())[0]])
ax.set_xlabel(list(parameters.keys())[1])
ax.set_ylabel(list(parameters.keys())[0])
ax.xticks(rotation=45)

plt.show()
