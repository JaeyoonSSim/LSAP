import torch
import numpy as np

def compute_accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)