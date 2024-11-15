import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances
from .base import BaseLoss

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity


class GaitSupConLoss(BaseLoss):
    def __init__(
            self,
            temperature=0.05,
            embedding_size=256
        ):
            super().__init__()
            
            self.loss_fn = losses.SupConLoss(
                temperature=temperature,
                distance=CosineSimilarity()
            )
            
            # Optional: Add cross-batch memory
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