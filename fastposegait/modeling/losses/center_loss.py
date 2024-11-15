import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.regularizers import CenterInvariantRegularizer
from .base import BaseLoss

class CenterLoss(BaseLoss):
    def __init__(self, feat_dim=256, num_classes=100, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        
        self.center_loss = losses.NTXentLoss(
            temperature=0.07,
            distance=None,
            embedding_regularizer=CenterInvariantRegularizer()
        )

    def forward(self, embeddings, labels):
        if len(embeddings.size()) == 3:  # [N, C, 1]
            embeddings = embeddings.squeeze(-1)

        loss = self.center_loss(embeddings, labels)
        self.info.update({
            "loss": loss.detach().clone(),
        })
        return loss, self.info
