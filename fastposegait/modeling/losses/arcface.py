import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances
from .base import BaseLoss

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity


class ArcFace(BaseLoss):
    def __init__(
            self, 
            embedding_size=256,
            num_classes=100,
            margin=28.6,
            scale=64
        ):
            super().__init__()
            
            self.arc_face = losses.ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=margin,
                scale=scale
            )
        
    def forward(self, embeddings, labels):
        if len(embeddings.size()) == 3:  # [N, C, P]
            embeddings = embeddings.squeeze(-1)

        loss = self.arc_face(embeddings, labels)
        self.info.update(
            {
                "loss": loss.detach().clone(),
            }
        )
        return loss, self.info

