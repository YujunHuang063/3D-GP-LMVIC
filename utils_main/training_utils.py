import pdb
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
    ) if aux_parameters else None

    return optimizer, aux_optimizer

def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    if not aux_list:
        aux_loss_sum = torch.tensor(0.0)

    return aux_loss_sum