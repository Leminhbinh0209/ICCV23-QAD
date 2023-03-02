import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import itertools


class CCALoss(nn.Module):
    def __init__(self, n_modality, weights, device, **kwargs):
        super(CCALoss, self).__init__()
        self.n_modality = n_modality
        self.weights = weights
        self.device = device
        self.all_components = list(
            itertools.combinations(range(n_modality), 2))
        self.reg_param = 1e-1

    def corr_loss(self, intermediate_output, channel_view=True):
        b, c, h, w = intermediate_output.size()

        n_base = b//self.n_modality
        x_input = torch.reshape(
            intermediate_output, (self.n_modality, n_base, c, h, w))
        x_input = torch.reshape(
            x_input, (self.n_modality, n_base, c, h * w))
        if channel_view:
            n_propection = c
            x_input = x_input.mean(dim=3)  # N x B x npropection
        else:
            n_propection = h * w
            x_input = x_input.mean(dim=2)
        corr_loss = 0.0
        id_matrix = torch.eye(n_propection).to(self.device)

        for (i, j) in self.all_components:
            view_one, view_two = x_input[i].transpose(
                1, 0), x_input[j].transpose(1, 0)  # n_propection x B
            view_one_mean, view_two_mean = view_one.mean(
                dim=1, keepdim=True), view_two.mean(dim=1, keepdim=True)
            view_one -= view_one_mean  # zero mean
            view_two -= view_two_mean

            view_one, view_two = F.normalize(
                view_one), F.normalize(view_two)

            corr = torch.einsum('id,id->i', view_one,
                                view_two).mean()

            id_reg = torch.norm(view_one @ view_one.T - id_matrix) \
                + torch.norm(view_one @ view_one.T - id_matrix)
            # ===========> Use another number than 1 if do not normalize
            corr_loss += (1.0 - corr) + self.reg_param * id_reg
        return corr_loss

    def forward(self, intermediate_outputs: list = []):
        return torch.sum([w * (self.corr_loss(ft) + self.corr_loss(ft, False)) for w, ft in zip(self.weights, intermediate_outputs) if w > 0])


class QualityParameterizeLoss(nn.Module):
    def __init__(self, n_modality, **kwargs):
        super(QualityParameterizeLoss, self).__init__()
        self.n_modality = n_modality
        self.labels = torch.arange(0, 1.0001, 1.0 / (n_modality-1))

    def forward(self, quality_pred: list = []):
        b, k = quality_pred[0].size()
        device = quality_pred[0].device
        assert k == 1, "Quality predict a scale number ... "
        assert (torch.max(b) < 1) and (torch.max(b) >
                                       0), "Quality values not in range (0,1)"

        n_steps = len(quality_pred)
        n_base = b // self.n_modality
        labels = torch.cat([self.labels.unsqueeze(-1)] *
                           n_base, dim=1).reshape(b).to(device)  # b
        lb_step_size = labels / (n_steps-1.0) if n_steps > 1 else labels
        loss = 0.0
        if n_steps == 1:
            return (quality_pred[0].reshape(b) -
                    labels).abs().mean()

        for i in range(n_steps):
            # enforce the model to increse the econded quality
            step_lb = labels - i * lb_step_size
            loss += (quality_pred[i].reshape(b) -
                     step_lb).abs().mean().div(n_steps)
        return loss


class L1SuperResolutionLoss(nn.Module):
    def __init__(self, n_modality, **kwargs):
        super(L1SuperResolutionLoss, self).__init__()
        self.n_modality = n_modality

    def forward(self, intermediate_output: list = []):
        x_input = intermediate_output[0]
        b, c, h, w = x_input.size()
        n_base = b//self.n_modality
        x_input = torch.reshape(
            x_input, (self.n_modality, n_base, c, h, w))
        x_input_raw = x_input[0]  # select the raw as the input
        x_label = torch.cat([x_input_raw]*self.n_modality,
                            dim=0).clone().detach()
        loss = 0.0
        for i in range(1, len(intermediate_output)):
            loss += (intermediate_output[i]-x_label).abs().mean(
                dim=0).sum().div(len(intermediate_output)-1)
        return loss


class AttentionPairLoss(nn.Module):
    def __init__(self, n_modality, neg_sampling=False, neg_margin=0.08):
        super(AttentionPairLoss, self).__init__()
        self.neg_sampling = neg_sampling
        self.neg_margin = neg_margin
        self.n_modality = n_modality
        self.all_components = list(
            itertools.combinations(range(n_modality), 2))
        if self.neg_sampling:
            print(f"With negative searching with margin of {neg_margin}")

    def forward(self, inatt_maps,  target):
        batch_size,  n_heads, wh, wh = inatt_maps.size()
        att_maps = torch.reshape(inatt_maps, (batch_size, n_heads * wh * wh))
        n_base_logits = batch_size // self.n_modality
        att_maps = torch.reshape(
            att_maps, (self.n_modality, n_base_logits, -1))
        loss = None
        for (i, j) in self.all_components:
            modality_att_loss = (att_maps[i] - att_maps[j]).pow(2).sum(dim=-1)
            loss = (
                loss + modality_att_loss) if loss is not None else modality_att_loss

        loss = loss.div(len(self.all_components)).mean()
        if not self.neg_sampling:
            return loss
        else:
            # Normalizing
            att_maps = torch.reshape(
                inatt_maps, (batch_size, n_heads * wh * wh))
            att_map_norm = torch.nn.functional.normalize(
                att_maps, p=2.0, dim=1)
            # in range (-1,1)
            sim = torch.matmul(att_map_norm, att_map_norm.transpose(1, 0))
            target_map = 1 - (target.unsqueeze(
                0) - target.unsqueeze(0).T).to(torch.bool).to(torch.int)  # 0 : negative, 1: positive
            target_map *= 2  # 0 : negative, 2: positive
            sim -= target_map  # masking  every positive smaller than -1
            neg_pair_idx = sim.argmax(dim=1)
            neg_att_maps = att_maps[neg_pair_idx]
            # alsway postive
            neg_distance = (att_maps - neg_att_maps).pow(2).sum(dim=-1)
            # greater than margin is remove
            neg_loss = torch.clamp(
                self.neg_margin-neg_distance, min=0.0).mean()
            loss += neg_loss
            return loss


class CrossEntrpyWeightLoss(nn.Module):
    def __init__(self, classes, dim=-1, min_weight=0.2,):
        """Weighted loss with different 
        """
        super(CrossEntrpyWeightLoss, self).__init__()

        self.weimin_weightght = min_weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):

        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class CollOutputLoss(nn.Module):
    def __init__(self, alpha=0.5, n_modality=3, temperature=2):
        """
            REF: https://github.com/qq456cvb/Collaborative-Learning/blob/master/main.py
        """
        super(CollOutputLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.n_modality = n_modality
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.hard_criterion = torch.nn.CrossEntropyLoss()
        print("Number of views: ", n_modality)

    def forward(self, logits, targets):
        batch_size, n_logits = logits.size()
        assert batch_size % self.n_modality == 0, "Number of data modality are not equal"
        n_base_logits = batch_size // self.n_modality
        soft_lb = logits #.clone() #.detach()
        sum_logit = soft_lb.view(
            self.n_modality, n_base_logits, n_logits).sum(dim=0, keepdim=True)  # Sum over modalities
      
        sum_logit = torch.cat([sum_logit]*self.n_modality, dim=0).view(batch_size, n_logits)
        soft_lb = (sum_logit - logits) / (self.n_modality-1)
        hard_loss = self.hard_criterion(logits, targets)

        soft_loss = -(F.log_softmax(logits / self.temperature, dim=1) * F.softmax(soft_lb.clone().detach() / self.temperature, dim=1)).sum(dim=1).mean()
        loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss
        return loss

class CollInterLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=0.05, temperature=2, n_classes=2):
        super(CollInterLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta 
        self.temperature = temperature
        self.hard_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, intermediate_outputs: list, targets: Tensor):
        """intermediate_outputs: list [(batch_size x n_class), (batch_size x n_class), ..., (batch_size x n_class)]"""
        
        tt_loss = 0
        for i, inter_out in enumerate(intermediate_outputs):
            hard_loss = self.hard_criterion(inter_out, targets)
            soft_lb = torch.stack(intermediate_outputs[i:], dim=0).mean(dim=0) # batch_size x n_class
            soft_loss = -(F.log_softmax(inter_out / self.temperature, dim=1) * F.softmax(soft_lb.clone().detach() / self.temperature, dim=1)).sum(dim=1).mean()
            tt_loss += (self.alpha * hard_loss + self.beta * soft_loss)
        return tt_loss

class CollSimLoss(nn.Module):
    def __init__(self, alpha_pull=0.001, alpha_push=0.001, n_classes=2):
        super(CollSimLoss, self).__init__()
        self.identity_mat = torch.eye(n_classes).cuda()
       
        self.n_classes = n_classes
        self.alpha_pull = alpha_pull
        self.alpha_push = alpha_push

    def _std(self, x):
        b, c, w, h = x.size()
        return x.view(b, c, w*h).std(dim=-1)

    def _center_mean(self, x):
        return x.sub(x.mean(dim=0))

    def _sim_matrix(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, inputs: Tensor, intermediate_outputs: list, targets: Tensor):
        """intermediate_outputs: list [(BxCxWxH), (BxCxWxH), ..., (BxCxWxH)]"""
        one_hot = self.identity_mat[targets] # b x n_classes
        
        input_rep = self._center_mean(self._std(inputs))
        input_sim = self._sim_matrix(input_rep, input_rep)
        out_rep = self._center_mean(one_hot)
        output_sim = self._sim_matrix(out_rep, out_rep)

        tt_loss = 0
        for  i, inter_out in enumerate(intermediate_outputs):
            inter_rep = self._center_mean(self._std(inter_out))
            inter_sim =  self._sim_matrix(inter_rep, inter_rep)
            pull_push_loss = self.alpha_pull * torch.norm(inter_sim - output_sim) - self.alpha_push*torch.norm(inter_sim - input_sim)
            tt_loss += pull_push_loss/(i+1)
        return tt_loss

class CollKernelLoss(nn.Module):
    def __init__(self):
        super(CollKernelLoss, self).__init__()

    def forward(self, kernel_list: list):
        """kernel_list: list [(P1xP2xKxK), (P1xP2xKxK), ..., (P1xP2xKxK)]"""
        tt_loss = 0
        for  i, kernel in enumerate(kernel_list):
            f, c, k, _ = kernel.size()
            kernel = kernel.view(f, c*k*k)
            kernel_norm = kernel.sub(kernel.mean(dim=0)).div(kernel.std(dim=0)) # f x g
            cov_mat = torch.mm(kernel_norm, kernel_norm.transpose(0, 1))
            tt_loss += torch.norm(cov_mat.fill_diagonal_ (0.0))
        return tt_loss

