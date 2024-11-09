import torch
import torch.nn.functional as F
from .base import BaseLoss


class ViTPoseLoss(BaseLoss):
    def __init__(self, temperature=0.07, margin=0.3, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.temperature = temperature
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], labels: [n]
        embeddings = embeddings.squeeze(-1)  # Remove last dimension
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute InfoNCE/Contrastive loss
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive and negative masks
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # Remove self-pairs
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        
        # Compute hard mining mask (find hard positives and negatives)
        with torch.no_grad():
            sim_matrix_detach = sim_matrix.detach()
            # Hard positives: positive pairs with low similarity
            hard_pos = (sim_matrix_detach < sim_matrix_detach.mean()) & pos_mask.bool()
            # Hard negatives: negative pairs with high similarity
            hard_neg = (sim_matrix_detach > sim_matrix_detach.mean()) & neg_mask.bool()
            mining_mask = (hard_pos | hard_neg).float()
            
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        infonce_loss = -(mining_mask * pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        
        # Triplet-like regularization
        pos_sim = (pos_mask * sim_matrix).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        neg_sim = (neg_mask * sim_matrix).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
        triplet_loss = F.relu(neg_sim - pos_sim + self.margin)
        
        # Combine losses
        total_loss = infonce_loss.mean() + 0.5 * triplet_loss.mean()
        
        # Update info dictionary
        self.info.update({
            'loss': total_loss.detach().clone(),
            'infonce_loss': infonce_loss.mean().detach().clone(),
            'triplet_loss': triplet_loss.mean().detach().clone(),
            'pos_sim': pos_sim.mean().detach().clone(),
            'neg_sim': neg_sim.mean().detach().clone(),
            'hard_pairs': mining_mask.sum().detach().clone()
        })
        
        return total_loss, self.info

    def _get_hard_pairs(self, sim_matrix, pos_mask, neg_mask, k=3):
        """Helper function to mine hard pairs"""
        batch_size = sim_matrix.size(0)
        
        # For each anchor, find k hardest positives and negatives
        pos_sim = sim_matrix.masked_fill(~pos_mask.bool(), float('-inf'))
        neg_sim = sim_matrix.masked_fill(~neg_mask.bool(), float('-inf'))
        
        # Get hardest positives (lowest similarity)
        _, hard_pos_idx = pos_sim.topk(k=min(k, pos_sim.size(1)), dim=1, largest=False)
        # Get hardest negatives (highest similarity)
        _, hard_neg_idx = neg_sim.topk(k=min(k, neg_sim.size(1)), dim=1, largest=True)
        
        hard_pos_mask = torch.zeros_like(pos_mask)
        hard_neg_mask = torch.zeros_like(neg_mask)
        
        # Create masks for hard pairs
        for i in range(batch_size):
            hard_pos_mask[i, hard_pos_idx[i]] = 1
            hard_neg_mask[i, hard_neg_idx[i]] = 1 