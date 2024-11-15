import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from .base import BaseLoss


class TupletMarginLoss(BaseLoss):
    def __init__(self, margin=5.73, scale=64, combine_parts="mean"):
        super().__init__()
        self.combine_parts = combine_parts
        self.loss_fn = losses.TupletMarginLoss(
            margin=margin, scale=scale, distance=CosineSimilarity()
        )

    def forward(self, embeddings, labels):
        # embeddings shape: [N, C, P] where P is number of parts
        N, C, P = embeddings.shape

        if self.combine_parts == "mean":
            # Average across parts first
            embeddings = embeddings.mean(dim=-1)  # [N, C]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            loss = self.loss_fn(embeddings, labels)

        elif self.combine_parts == "concat":
            # Reshape to combine all parts features
            embeddings = embeddings.permute(0, 2, 1)  # [N, P, C]
            embeddings = embeddings.reshape(N, P * C)  # [N, P*C]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            loss = self.loss_fn(embeddings, labels)

        elif self.combine_parts == "separate":
            # Calculate loss for each part separately
            losses = []
            for p in range(P):
                part_emb = embeddings[:, :, p]  # [N, C]
                part_emb = F.normalize(part_emb, p=2, dim=1)
                part_loss = self.loss_fn(part_emb, labels)
                losses.append(part_loss)
            loss = torch.mean(torch.stack(losses))
        self.info.update({
            "loss": loss.detach().clone(),
        })
        return loss, self.info
