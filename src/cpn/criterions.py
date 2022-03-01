import torch.nn as nn
import torch


class CPNLoss(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.s = h['scale']

    def forward(self, outputs, targets):
        y = targets['label']  # B * 1 * N * 1
        mask = targets['mask']  # B * 1 * N * 1
        num_sample = torch.sum(mask)
        num_pos = torch.sum(y * mask)
        num_neg = num_sample - num_pos
        loss_collection = dict()
        preds = torch.sigmoid(outputs).clamp(0.001, 0.999)
        loss_map = -(y * torch.log(preds) + (1 - y) * torch.log(1 - preds)) * mask
        pos = torch.sum(loss_map * y) / num_pos
        neg = torch.sum(loss_map * (1 - y)) / num_neg
        loss = self.s * pos + (1 - self.s) * neg
        loss_collection['pos_loss'] = pos
        loss_collection['neg_loss'] = neg
        loss_collection['total_loss'] = loss
        return loss_collection
