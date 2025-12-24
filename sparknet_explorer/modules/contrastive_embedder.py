"""
Contrastive Embedder Loss - Teach embedder meaningful representations.

Goal: Similar inputs → similar embeddings, different inputs → different embeddings.

This gives the embedder a clear learning signal instead of being frozen
or adversarially trying to confuse the curiosity module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEmbedderLoss(nn.Module):
    """
    Contrastive loss for state embedder.

    Uses position similarity as supervision signal:
    - Close positions in 2D → embeddings should be similar
    - Far positions in 2D → embeddings should be different
    """

    def __init__(self, temperature=0.1, margin=0.5, buffer_size=256):
        """
        Args:
            temperature: Softmax temperature for contrastive loss
            margin: Distance threshold - positions closer than this are "similar"
            buffer_size: Number of recent (position, embedding) pairs to store
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.buffer_size = buffer_size

        # Buffer of recent positions and embeddings
        self.position_buffer = []
        self.embedding_buffer = []

    def add_sample(self, position, embedding):
        """
        Add a (position, embedding) pair to the buffer.

        Args:
            position: 2D position tensor (detached)
            embedding: State embedding tensor (with gradients)
        """
        # Store detached position, but keep embedding for gradient flow
        self.position_buffer.append(position.detach().cpu())
        self.embedding_buffer.append(embedding)

        # Trim buffer if too large
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            self.embedding_buffer.pop(0)

    def compute_loss(self, current_position, current_embedding):
        """
        Compute contrastive loss for current sample against buffer.

        Args:
            current_position: Current 2D position (batch, 2)
            current_embedding: Current embedding (batch, embed_dim)

        Returns:
            Contrastive loss tensor (scalar)
        """
        if len(self.position_buffer) < 16:
            # Not enough samples yet
            return torch.tensor(0.0, device=current_embedding.device)

        # Sample from buffer (use recent samples)
        n_samples = min(32, len(self.position_buffer))
        indices = list(range(len(self.position_buffer) - n_samples, len(self.position_buffer)))

        # Stack buffer samples
        buffer_positions = torch.stack([self.position_buffer[i] for i in indices])
        buffer_embeddings = torch.stack([self.embedding_buffer[i].squeeze(0) for i in indices])

        # Move to same device
        device = current_embedding.device
        buffer_positions = buffer_positions.to(device)
        buffer_embeddings = buffer_embeddings.to(device)

        # Current position (squeeze batch dim if present)
        curr_pos = current_position.squeeze(0) if current_position.dim() > 1 else current_position
        curr_emb = current_embedding.squeeze(0) if current_embedding.dim() > 1 else current_embedding

        # Compute position distances (in 2D space)
        pos_distances = torch.norm(buffer_positions - curr_pos, dim=-1)  # (n_samples,)

        # Compute embedding similarities (cosine)
        curr_emb_norm = F.normalize(curr_emb.unsqueeze(0), dim=-1)  # (1, embed_dim)
        buffer_emb_norm = F.normalize(buffer_embeddings, dim=-1)    # (n_samples, embed_dim)
        emb_similarities = torch.mm(curr_emb_norm, buffer_emb_norm.t()).squeeze(0)  # (n_samples,)

        # Labels: 1 if positions are close, 0 if far
        labels = (pos_distances < self.margin).float()

        # Need both positive and negative samples for contrastive learning
        n_positive = labels.sum().item()
        n_negative = (1 - labels).sum().item()

        if n_positive < 2 or n_negative < 2:
            # Not enough contrast
            return torch.tensor(0.0, device=device)

        # Contrastive loss:
        # - Similar positions (label=1): maximize embedding similarity
        # - Different positions (label=0): minimize embedding similarity

        # Scale similarities by temperature
        scaled_sim = emb_similarities / self.temperature

        # Loss for positive pairs: -log(sigmoid(sim))
        # Loss for negative pairs: -log(1 - sigmoid(sim))
        positive_loss = -F.logsigmoid(scaled_sim) * labels
        negative_loss = -F.logsigmoid(-scaled_sim) * (1 - labels)

        # Average loss
        loss = (positive_loss.sum() + negative_loss.sum()) / n_samples

        return loss

    def get_stats(self):
        """Get buffer statistics."""
        return {
            'buffer_size': len(self.position_buffer),
            'buffer_capacity': self.buffer_size
        }

    def reset(self):
        """Clear the buffer."""
        self.position_buffer = []
        self.embedding_buffer = []
