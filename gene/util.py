from collections import namedtuple
import itertools

import numpy as np
from tqdm import tqdm


def split_into_batchs(items, batch_size):
    if len(items) <= batch_size:
        return [items]
    else:
        return [items[:batch_size]] + split_into_batchs(items[batch_size:], batch_size)


def flatten_2d_list(nested_list):
    return [item for items in nested_list for item in items]


def get_accuracy(test_loader, model, device):
    preds = []
    trues = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)

        preds.extend(y_pred.to("cpu").detach().numpy())
        trues.extend(labels.to("cpu").detach().numpy())

    total_matches = np.sum(np.argmax(preds, axis=-1) == trues)

    return total_matches/len(preds)


def grid_search(trainer, parameters: dict):
    Result = namedtuple("Result", [*parameters.keys(), "score"])
    output = []

    combinations = itertools.product(*list(parameters.values()))
    combinations = [{param_name: param_value for param_name, param_value in zip(parameters.keys(), combination)}
                    for combination in combinations]

    for combination in tqdm(combinations):
        print(combination)
        score = trainer(**combination)
        result = Result(**combination, score=score)
        output.append(result)

    return output
