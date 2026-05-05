import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss (Khosla et al., 2020)"""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalize features to map them to the unit hypersphere
        features = F.normalize(features, p=2, dim=1)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask to identify items of the same class
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask-out self-contrast cases (an item should not be pulled to itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Compute mean of log-likelihood over positive cases
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6, 1, mask_pos_pairs
        )  # prevent division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos.mean()
        return loss
