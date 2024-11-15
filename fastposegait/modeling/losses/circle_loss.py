import torch
from pytorch_metric_learning import losses
from .base import BaseLoss


class CircleLoss(BaseLoss):
    def __init__(self, m=0.25, gamma=256):
        super().__init__()

        self.circle_loss = losses.CircleLoss(
            m=m,
            gamma=gamma
        )

    def forward(self, embeddings, labels):
        if len(embeddings.size()) == 3:  # [N, C, 1]
            embeddings = embeddings.squeeze(-1)

        loss = self.circle_loss(embeddings, labels)
        self.info.update({
            "loss": loss.detach().clone(),
        })
        return loss, self.info
