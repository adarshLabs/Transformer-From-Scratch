import torch
import torch.nn as nn
from dataclasses import dataclass
try:
    from .gpt_decoder_block import GPTDecoderBlock
except ImportError:
    from gpt_decoder_block import GPTDecoderBlock
from attention.masking import causal_mask, padding_mask, combined_mask
import torch.nn.functional as F


@dataclass
class GPT2Config:
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    vocab_size: int = 50257
    block_size: int = 256
    expansion_factor: int = 4
    dropout: float = 0.1
    padding_token: int = 0

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                embed_dim=config.d_model,
                num_heads=config.n_heads, 
                expansion_factor=config.expansion_factor, 
                dropout=config.dropout
            ) 
            for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _build_mask(self, input_ids):
        B, S = input_ids.shape

        causal = causal_mask(S)
        padding = padding_mask(input_ids, padding_token=self.config.padding_token)
        return combined_mask(causal, padding)
    
    def forward(self, input_ids, target=None):

        embedding = self.token_embedding(input_ids)
        x = self.drop(embedding)
        mask = self._build_mask(input_ids)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1, top_k=50):
        pass