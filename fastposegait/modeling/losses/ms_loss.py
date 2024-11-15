import torch
from pytorch_metric_learning import losses
from .base import BaseLoss
from pytorch_metric_learning.distances import CosineSimilarity

class MSLoss(BaseLoss):
    def __init__(
            self,
            alpha=2.0,
            beta=50.0,
            base=0.0,
            embedding_size=256
    ):
        super().__init__()

        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=alpha,
            beta=beta,
            base=base
        )

        # Optional: Add cross-batch memory like in SupCon
        self.memory = losses.CrossBatchMemory(
            loss=self.loss_fn,
            embedding_size=embedding_size,
            memory_size=1024
        )

    def forward(self, embeddings, labels):
        if len(embeddings.size()) == 3:
            embeddings = embeddings.squeeze(-1)

        loss = self.memory(embeddings, labels)
        return loss, {"loss": loss.detach().clone()}
