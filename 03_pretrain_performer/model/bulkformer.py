"""
BulkFormer -- Performer-based foundation model for bulk RNA-seq.

Architecture:
    Input: [B, G] gene expression vector (continuous values)
      |
      v
    PositionalExprEmbedding (REE) + ESM2 gene identity + AutoEncoder sample embedding
      |
      v
    x_proj MLP (mixing layer)
      |
      v
    N x GBFormer blocks:
        1. LayerNorm
        2. GCNConv (gene-gene graph message passing)
        3. Binning: sort genes by learned scores -> split into bins
        4. Local Performer attention (one per bin)
        5. Unsort (restore original gene order)
        6. Global Performer attention (across all genes)
      |
      v
    LayerNorm
      |
      v
    Prediction head MLP -> [B, G] reconstructed expression

Adapted from: https://github.com/alwalt/BioFM
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from performer_pytorch import Performer

model_params = {
    "dim": 320,
    "bins": 10,
    "gb_repeat": 1,
    "p_repeat": 2,
    "bin_head": 8,
    "full_head": 4,
    "gene_length": 19357,
}


class PositionalExprEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE).

    Fixed (non-trainable) embedding that converts continuous gene expression
    values into sinusoidal rotation features. Masked positions (-10) are zeroed.
    """

    def __init__(self, dim):
        super().__init__()
        self.mask_token_id = -10
        self.inv_freq = nn.Parameter(
            1.0 / (100 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=False,
        )

    def forward(self, x):
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x = torch.einsum("bi,j->bij", x, self.inv_freq)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        x[x_mask_idx[:, 0], x_mask_idx[:, 1]] = 0
        return x


class GBFormer(nn.Module):
    """
    Graph-Bin-Former encoder block.

    Combines GCN message passing, expression-bin-conditional local Performer
    attention, and global Performer attention.
    """

    def __init__(self, dim, gene_length, bin_head=4, full_head=4, bins=10, p_repeat=1):
        super().__init__()
        self.dim = dim
        self.gene_length = gene_length
        self.bins = bins
        self.p_repeat = p_repeat
        self.bin_head = bin_head
        self.full_head = full_head

        self.g = GCNConv(dim, dim, cached=True, add_self_loops=False)

        self.which_b = nn.Sequential(nn.Linear(self.dim, 1))

        self.b = nn.ModuleList(
            [
                Performer(
                    dim=self.dim,
                    heads=self.bin_head,
                    depth=1,
                    dim_head=self.dim // self.bin_head,
                    attn_dropout=0.2,
                    ff_dropout=0.2,
                )
                for _ in range(self.bins)
            ]
        )

        self.f = nn.Sequential(
            *[
                Performer(
                    dim=self.dim,
                    heads=self.full_head,
                    depth=1,
                    dim_head=self.dim // self.full_head,
                )
                for _ in range(self.p_repeat)
            ]
        )

        self.layernorm = nn.LayerNorm(self.dim)

    def forward(self, x, graph):
        b, g, e = x.shape
        x = self.layernorm(x)
        x = x + self.g(x, graph)

        if self.bins > 0:
            which_b = self.which_b(x).squeeze(-1)
            order = torch.sort(which_b, dim=1, descending=True)[1]
            order = order.unsqueeze(-1).repeat(1, 1, e)
            n = (g - 1) // self.bins + 1

            x = x.gather(1, order)
            xs = torch.split(x, n, dim=1)
            xs = [layer(x_bin) for x_bin, layer in zip(xs, self.b)]
            xs = torch.cat(xs, dim=1)

            x = torch.empty_like(xs)
            x = x.scatter_(1, order, xs)

        x = self.f(x)
        return x


class BulkFormer(nn.Module):
    """
    BulkFormer: Performer-based foundation model for bulk RNA-seq.

    Combines expression embeddings (REE), gene identity embeddings (ESM2),
    sample-level autoencoder embeddings, GCN graph convolution, binned
    local attention, and global Performer attention.
    """

    def __init__(
        self,
        dim,
        graph,
        gene_emb,
        gene_length,
        bin_head=4,
        full_head=4,
        bins=10,
        gb_repeat=3,
        p_repeat=1,
    ):
        super().__init__()
        self.dim = dim
        self.gene_length = gene_length
        self.bins = bins
        self.bin_head = bin_head
        self.full_head = full_head
        self.gb_repeat = gb_repeat
        self.p_repeat = p_repeat
        self.graph = graph

        self.gene_emb = nn.Parameter(gene_emb)

        self.gene_emb_proj = nn.Sequential(
            nn.Linear(self.gene_emb.shape[1], 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

        self.expr_emb = PositionalExprEmbedding(self.dim)

        self.x_proj = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

        self.gb_formers = nn.ModuleList(
            [
                GBFormer(
                    self.dim,
                    self.gene_length,
                    self.bin_head,
                    self.full_head,
                    self.bins,
                    self.p_repeat,
                )
                for _ in range(self.gb_repeat)
            ]
        )

        self.layernorm = nn.LayerNorm(self.dim)

        self.ae_enc = nn.Sequential(
            nn.Linear(self.gene_length, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, 1),
            nn.ReLU(),
        )

    def forward(self, x, repr_layers=None):
        b, g = x.shape

        x = (
            self.expr_emb(x)
            + self.gene_emb_proj(self.gene_emb)
            + self.ae_enc(x).unsqueeze(1)
        )
        x = self.x_proj(x)

        hidden = {}
        for idx, layer in enumerate(self.gb_formers):
            x = layer(x, self.graph)
            if repr_layers and idx in repr_layers:
                hidden[idx] = x

        x = self.layernorm(x)
        if repr_layers and idx in repr_layers:
            hidden[idx] = x

        x = self.head(x).squeeze(-1)

        if repr_layers:
            return x, hidden
        return x
