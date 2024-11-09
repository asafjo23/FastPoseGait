import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers, distances
from .base import BaseLoss


# class TripletLoss(BaseLoss):
#     def __init__(self, margin, loss_term_weight=1.0):
#         super(TripletLoss, self).__init__(loss_term_weight)
#         self.margin = margin
#
#     def forward(self, embeddings, labels):
#         # embeddings: [n, c, p], label: [n]
#         embeddings = embeddings.permute(
#             2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
#
#         ref_embed, ref_label = embeddings, labels
#         dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
#         mean_dist = dist.mean((1, 2))  # [p]
#         ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
#         dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
#         loss = F.relu(dist_diff + self.margin)
#
#         hard_loss = torch.max(loss, -1)[0]
#         loss_avg, loss_num = self.AvgNonZeroReducer(loss)
#
#         self.info.update({
#             'loss': loss_avg.detach().clone(),
#             'hard_loss': hard_loss.detach().clone(),
#             'loss_num': loss_num.detach().clone(),
#             'mean_dist': mean_dist.detach().clone()})
#
#         return loss_avg, self.info
#
#     def AvgNonZeroReducer(self, loss):
#         eps = 1.0e-9
#         loss_sum = loss.sum(-1)
#         loss_num = (loss != 0).sum(-1).float()
#
#         loss_avg = loss_sum / (loss_num + eps)
#         loss_avg[loss_num == 0] = 0
#         return loss_avg, loss_num
#
#     def ComputeDistance(self, x, y):
#         """
#             x: [p, n_x, c]
#             y: [p, n_y, c]
#         """
#         x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
#         y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
#         inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
#         dist = x2 + y2 - 2 * inner
#         dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
#         return dist
#
#     def Convert2Triplets(self, row_labels, clo_label, dist):
#         """
#             row_labels: tensor with size [n_r]
#             clo_label : tensor with size [n_c]
#         """
#         matches = (row_labels.unsqueeze(1) ==
#                    clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
#         diffenc = torch.logical_not(matches)  # [n_r, n_c]
#         p, n, _ = dist.size()
#         ap_dist = dist[:, matches].view(p, n, -1, 1)
#         an_dist = dist[:, diffenc].view(p, n, 1, -1)
#         return ap_dist, an_dist


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
            pos_margin=0.2,  # Tighter positive margin
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
        embeddings = embeddings.squeeze(-1)

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

