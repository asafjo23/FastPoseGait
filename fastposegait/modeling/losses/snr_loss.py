import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import SNRDistance
from .base import BaseLoss

class SNRLoss(BaseLoss):
    def __init__(
            self,
            pos_margin=0.0,
            neg_margin=0.2,
        ):
            super().__init__()
            self.distance = SNRDistance()
            self.loss_fn = losses.SignalToNoiseRatioContrastiveLoss(
                pos_margin=pos_margin,
                neg_margin=neg_margin,
                distance=self.distance
            )

    def forward(self, embeddings, labels):
        # embeddings shape: [N, C, P]
        N, C, P = embeddings.size()
        
        total_loss = 0
        for p in range(P):
            part_embeddings = embeddings[:, :, p]  # [N, C]
            part_embeddings = F.normalize(part_embeddings, p=2, dim=1)
            part_loss = self.loss_fn(part_embeddings, labels)
            total_loss += part_loss
            
        loss = total_loss / P
        self.info.update({
            "loss": loss.detach().clone(),
        })
        return loss, self.info
