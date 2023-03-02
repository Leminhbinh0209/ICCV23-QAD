import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20
from  helper.utils import prob_loss

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AWP(object):
    """
    REF: https://github.com/csdongxian/AWP/tree/main/trades_AWP
    """
    def __init__(self, model, proxy, proxy_optim, gamma):

        super(AWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_compressed, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict() \
            if not isinstance(self.model, nn.DataParallel)\
             else self.model.module.state_dict())
        self.proxy.train()
        logits_compressed = self.proxy(inputs_compressed)
        loss = - F.cross_entropy(logits_compressed, targets)
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




