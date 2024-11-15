import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from .base import BaseLoss

class NTXentLoss(BaseLoss):
    def __init__(
            self,
            temperature=0.07,
            embedding_size=256
        ):
            super().__init__()
            
            self.loss_fn = losses.NTXentLoss(
                temperature=temperature,
                distance=CosineSimilarity()
            )
            
            self.memory = losses.CrossBatchMemory(
                loss=self.loss_fn,
                embedding_size=embedding_size,
                memory_size=512
            )

    def forward(self, embeddings, labels):
        # embeddings shape: [N, C, P]
        N, C, P = embeddings.size()
        
        # Compute loss for each part
        total_loss = 0
        for p in range(P):
            part_embeddings = embeddings[:, :, p]  # [N, C]
            part_embeddings = F.normalize(part_embeddings, p=2, dim=1)
            part_loss = self.memory(part_embeddings, labels)
            total_loss += part_loss
            
        # Average over parts
        loss = total_loss / P
        self.info.update({
            "loss": loss.detach().clone(),
        })
        return loss, self.info
