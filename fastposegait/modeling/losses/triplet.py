import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances
from .base import BaseLoss


class TripletLoss(BaseLoss):
    def __init__(self, margin=0.3, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        
        # Increased margin for better separation
        self.margin = margin * 2  # Double the margin
        
        # Distance metrics
        self.euclidean_distance = distances.LpDistance(
            normalize_embeddings=True,
            p=2,
            power=1
        )
        self.cosine_distance = distances.CosineSimilarity()
        
        # More aggressive mining strategy
        self.miner = miners.MultiSimilarityMiner(
            epsilon=0.1,
            distance=self.euclidean_distance
        )
        
        # Enhanced triplet loss with larger margin
        self.triplet_loss = losses.TripletMarginLoss(
            margin=self.margin,
            distance=self.euclidean_distance,
            reducer=reducers.AvgNonZeroReducer()
        )
        
        # Adjusted contrastive loss
        self.contrastive_loss = losses.ContrastiveLoss(
            pos_margin=0.1,  # Tighter positive margin
            neg_margin=self.margin,
            distance=self.euclidean_distance,
            reducer=reducers.AvgNonZeroReducer()
        )
        
        # Add circle loss for better separation
        self.circle_loss = losses.CircleLoss(
            m=0.4,
            gamma=80,
            distance=self.cosine_distance
        )

    def forward(self, embeddings, labels):
        embeddings = embeddings.permute(2, 0, 1).contiguous().float()
        p, n, c = embeddings.size()
        embeddings = embeddings.view(-1, c)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Repeat labels for each part
        labels = labels.repeat(p)
        
        # Get hard pairs using miner
        hard_pairs = self.miner(embeddings, labels)
        
        # Compute losses with mining
        loss_triplet = self.triplet_loss(embeddings, labels, hard_pairs) * 2.0
        loss_contrastive = self.contrastive_loss(embeddings, labels) * 1.5
        loss_circle = self.circle_loss(embeddings, labels)
        
        # Weighted combination
        total_loss = (0.5 * loss_triplet + 
                     0.3 * loss_contrastive +
                     0.2 * loss_circle)

        self.info.update({
            'loss': total_loss.detach().clone(),
            'triplet_loss': loss_triplet.detach().clone(),
            'contrastive_loss': loss_contrastive.detach().clone(),
            'circle_loss': loss_circle.detach().clone(),
        })

        return total_loss, self.info

