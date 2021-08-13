import os,sys
import torch
import torch.nn as nn
from perceiver_pytorch import PerceiverIO
from nestedtensor import NestedTensor

class LArceiverDetection(nn.Module):
    def __init__( self,
                  *,
                  depth,
                  dim,
                  queries_dim,
                  num_detection_queries,
                  logits_dim = None,
                  num_latents = 512,
                  latent_dim = 512,
                  cross_heads = 1,
                  latent_heads = 8,
                  cross_dim_head = 64,
                  latent_dim_head = 64,
                  weight_tie_layers = False,
                  self_per_cross_attn = 1 ):
        """
        Parameters:
          depth: depth of the processing portion
          dim: dimension of sequence to be encoded
          queries_dim: dimension of decoder queries
          logits_dim: dimension of final logits
          num_latents: number of latents, or induced set points, or centroids. different papers giving it different names
          latent_dim: latent dimension
          cross_heads: number of heads for cross attention (1 in paper)
          latent_heads: number of heads for latent self attention (8 in paper)
          cross_dim_head: number of dimensions per cross attention head (64)
          latent_dim_head: number of dimensions per latent self attention head (64)
          weight_tie_layers: whether to weight tie layers in the processor (optional, as indicated in the diagram)(false)
          self_per_cross_attn: number of self attention blocks per cross attention (2)
        """
        super().__init__()

        # the perciever
        self.perceiver = PerceiverIO( depth=depth,
                                      dim=dim,
                                      queries_dim=queries_dim,
                                      logits_dim=logits_dim,
                                      num_latents=num_latents,
                                      latent_dim=latent_dim,
                                      cross_heads=cross_heads,
                                      latent_heads=latent_heads,
                                      cross_dim_head=cross_dim_head,
                                      latent_dim_head=latent_dim_head,
                                      weight_tie_layers=weight_tie_layers,
                                      self_per_cross_attn=self_per_cross_attn )

        # we define detection queries, just like the DETR architecture
        self.num_detection_queries = num_detection_queries
        self.query_embed = nn.Embedding(num_detection_queries, queries_dim)

    def forward(self, samples):

        bs, seqlen, _ = samples.shape
        batch_query_embed = self.query_embed.weight.repeat(bs, 1, 1)        
        out = self.perceiver( samples, queries=batch_query_embed )
        return out

