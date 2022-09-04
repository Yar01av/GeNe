import torch


def get_negative_accuracy_target(model_out, labels):
    preds = torch.argmax(model_out, dim=-1)
    total_correct = torch.eq(preds, labels)

    return -torch.sum(total_correct)/len(preds)


def make_supervised_loss(loss, X, y):
    return lambda model: loss(model(X), y)
