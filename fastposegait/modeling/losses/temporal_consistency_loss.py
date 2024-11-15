import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances
from .base import BaseLoss


class TemporalConsistencyLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        
    def forward(self, embeddings):
        temp_diff = embeddings[:, 1:] - embeddings[:, :-1]
        temp_loss = torch.mean(torch.norm(temp_diff, dim=-1))
        self.info.update(
            {
                "loss": temp_loss.detach().clone(),
            }
        )
        return temp_loss, self.info
