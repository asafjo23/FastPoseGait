import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, is_hard_loss=False, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.is_hard_loss = is_hard_loss

    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        
        if self.is_hard_loss:
            loss_return = self.batch_hard_triplet_loss(ap_dist,an_dist,dist)
        else:
            loss_return = self.batch_all_triplet_loss(ap_dist,an_dist,dist)

        return loss_return, self.info

    def batch_hard_triplet_loss(self, ap_dist, an_dist, dist):
        # Largest d(a,p) per anchor
        ap_dist_max = ap_dist.max(-1)[0]  # [p, n_r]
        # Smallest d(a,n) per anchor
        an_dist_min = an_dist.min(-1)[0]  # [p, n_r]

        # Create a validity mask to handle cases where there are no positives or negatives
        valid_pos = (ap_dist_max != float('-inf'))
        valid_neg = (an_dist_min != float('inf'))
        valid = valid_pos & valid_neg  # [p, n_r]

        # Compute the loss only for valid anchors
        loss = F.relu(ap_dist_max - an_dist_min + self.margin)  # [p, n_r]
        loss = loss * valid.float()  # Zero out invalid positions

        # Average over valid entries
        num_valid = valid.float().sum()
        if num_valid > 0:
            hard_loss = loss.sum() / num_valid
        else:
            hard_loss = torch.tensor(0.0, device=loss.device)

        self.info.update({
            'hard_loss': hard_loss.detach().clone()
        })
        return hard_loss

    def batch_all_triplet_loss(self,ap_dist,an_dist,dist):
        mean_dist = dist.mean((1, 2))  # [p]
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)
        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})
        return loss_avg
    
    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, col_labels, dist):
        """
            row_labels: tensor with size [n_r]
            col_labels: tensor with size [n_c]
            dist: tensor with size [p, n_r, n_c]
        """
        matches = (row_labels.unsqueeze(1) == col_labels.unsqueeze(0))  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n_r, n_c = dist.size()

        # Expand matches and diffenc to size [p, n_r, n_c]
        matches = matches.unsqueeze(0).expand(p, -1, -1)  # [p, n_r, n_c]
        diffenc = diffenc.unsqueeze(0).expand(p, -1, -1)  # [p, n_r, n_c]

        # For ap_dist, set non-positive distances to -inf (so max will ignore them)
        ap_dist = dist.clone()
        ap_dist[~matches] = float('-inf')

        # For an_dist, set non-negative distances to +inf (so min will ignore them)
        an_dist = dist.clone()
        an_dist[~diffenc] = float('inf')

        return ap_dist, an_dist
