import torch


class weighted_l1_Loss(torch.nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        if reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'mean':
            self.reduction = torch.mean

    def forward(self, input, target):
        loss = abs(input - target) * (2**target)
        loss = self.reduction(loss)
        return loss
