import torch
import torch.nn.functional as F
from .base import BaseLoss


class PoseContrastiveLoss(BaseLoss):
    def __init__(self, temperature=0.07, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], labels: [n]
        embeddings = embeddings.squeeze(-1)  # Remove last dimension if present
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Remove self-pairs
        pos_mask.fill_diagonal_(0)

        # Compute loss
        numerator = torch.exp(sim_matrix)
        denominator = numerator.sum(dim=1, keepdim=True)
        log_probs = sim_matrix - torch.log(denominator)
        
        loss = -(pos_mask * log_probs).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        loss_avg = loss.mean()

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'pos_pairs': pos_mask.sum().detach().clone(),
            'mean_sim': sim_matrix.mean().detach().clone()
        })

        return loss_avg, self.info
