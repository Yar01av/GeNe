from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gene.optimisers.division import ParallelDivisionOptimiser, DivisionOptimiser
import torch
import torch.nn.functional as F
from torch import nn


DEVICE = "cuda"


class TestNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self._model = nn.Sequential(nn.Linear(input_shape, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, n_actions))

    def forward(self, x):
        return self._model(x)


optimiser = ParallelDivisionOptimiser(loss=lambda x, y: nn.KLDivLoss(reduction="batchmean")(F.log_softmax(x, dim=1), y),
                                      random_function=lambda shape: torch.normal(0, 0.1, shape),
                                      selection_limit=100,
                              device=DEVICE)
models = [TestNetwork(4, 2).cuda()]

latest_scores = []

start = datetime.now()

for _ in tqdm(range(1000)):
    models = optimiser.step(models, torch.zeros(2, 4).to(DEVICE), F.softmax(torch.zeros(2, 2), dim=1).to(DEVICE))
    latest_scores.append(np.mean(optimiser.get_losses()))

    if len(latest_scores) > 100:
        # print(np.mean(latest_scores[0:-100]))
        pass

end = datetime.now()
print(end-start)

plt.plot(latest_scores)
plt.show()
