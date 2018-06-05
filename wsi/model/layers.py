import torch
from torch import nn


class CRF(nn.Module):
    """
    CRF
    """
    def __init__(self, num_nodes, iteration=10):
        super(CRF, self).__init__()
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.zeros(1, num_nodes, num_nodes))

    def forward(self, feats, logits):
        feats_norm = torch.norm(feats, p=2, dim=2, keepdim=True)
        pairwise_norm = torch.bmm(feats_norm,
                                  torch.transpose(feats_norm, 1, 2))
        pairwise_dot = torch.bmm(feats, torch.transpose(feats, 1, 2))
        pairwise_sim = pairwise_dot / pairwise_norm
        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()

        for i in range(self.iteration):
            probs = torch.transpose(logits.sigmoid(), 1, 2)
            pairwise_potential_E = torch.sum(
                probs * pairwise_potential - (1 - probs) * pairwise_potential,
                dim=2, keepdim=True)
            logits = unary_potential + pairwise_potential_E

        return logits

    def __repr__(self):
        return 'CRF(num_nodes={}, iteration={})'.format(
            self.num_nodes, self.iteration
        )
