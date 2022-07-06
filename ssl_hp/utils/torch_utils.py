import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy


def sharpening(label, T):
    label = label.pow(1 / T)
    return label / label.sum(-1, keepdim=True)


def customized_weight_decay(model, weight_decay, ignore_key=["bn", "bias"]):
    for name, p in model.named_parameters():
        if not any(key in name for key in ignore_key):
            p.data.mul_(1 - weight_decay)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]



def my_interleave(inputs, batch_size):
    """
    * This function will override inputs
    * Make the data interleave.
    * Swap the data of the first batch (inputs[0]) to other batches.
    * change_indices would be a increasing function.
    * len(change_indices) should be the same as len(inputs[0]), and the elements
      denote which row should the data in first row change with.
    """

    def swap(A, B):
        """
        swap for tensors
        """
        return B.clone(), A.clone()

    ret = inputs
    inputs_size = len(inputs)

    repeat = batch_size // inputs_size
    residual = batch_size % inputs_size

    # equally switch the first row to other rows, so we compute how many repeat for range(inputs_size),
    # which store the rows to change.
    # some of the element cannot evenly spread two rows, so we preferentially use the rows which are farer to 0th row.

    change_indices = list(range(inputs_size)) * repeat + list(
        range(inputs_size - residual, inputs_size)
    )
    change_indices = sorted(change_indices)
    # print(change_indices)

    # the change_indices is monotone increasing function, so we can group the same elements and swap together
    # e.g. change_indices = [0, 1, 1, 2, 2, 2]
    #      => two_dimension_change_indices = [[0], [1, 1], [2, 2, 2]]
    two_dimension_change_indices = []

    change_indices.insert(0, -1)
    change_indices.append(change_indices[-1] + 1)
    start = 0
    for i in range(1, len(change_indices)):
        if change_indices[i] != change_indices[i - 1]:
            two_dimension_change_indices.append(change_indices[start:i])
            start = i

    two_dimension_change_indices.pop(0)

    i = 0
    for switch_rows in two_dimension_change_indices:
        switch_row = switch_rows[0]
        num = len(switch_rows)
        ret[0][i : i + num], ret[switch_row % inputs_size][i : i + num] = swap(
            ret[0][i : i + num], ret[switch_row % inputs_size][i : i + num]
        )
        i += num

    return ret


def split_weight_decay_weights(model, weight_decay, ignore_key=["bn", "bias"]):
    weight_decay_weights = []
    no_weight_decay_weights = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(key in name for key in ignore_key):
            no_weight_decay_weights.append(p)
        else:
            # print(name)
            weight_decay_weights.append(p)

    return [
        {"params": no_weight_decay_weights, "weight_decay": 0.0},
        {"params": weight_decay_weights, "weight_decay": weight_decay},
    ]


class WeightDecayModule:
    def __init__(self, model, weight_decay, ignore_key=["bn", "bias"]):
        self.weight_decay = weight_decay
        self.available_parameters = []
        for name, p in model.named_parameters():
            if not any(key in name for key in ignore_key):
                # print(name)
                self.available_parameters.append(p)

    def decay(self):
        for p in self.available_parameters:
            p.data.mul_(1 - self.weight_decay)
