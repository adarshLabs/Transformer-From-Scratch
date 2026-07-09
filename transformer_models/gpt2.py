import torch.nn as nn
from dataclasses import dataclass

try:
    from .gpt_decoder_block import GPTDecoderBlock
except ImportError:
    from gpt_decoder_block import GPTDecoderBlock

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
        pass
    
